"""Stateful Track-ID-based directional line crossing counter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from people_flow.counting.geometry import DirectedLine, LineSide, Point, euclidean_distance
from people_flow.errors import CountingError
from people_flow.tracking.track_record import TrackRecord

EventType = Literal["enter", "exit"]


@dataclass(frozen=True, slots=True)
class LineCrossingEvent:
    """Auditable evidence for one accepted Track-ID crossing."""

    event_id: str
    track_id: int
    event_type: EventType
    frame_id: int
    timestamp_ms: float
    previous_side: str
    current_side: str
    foot_x: float
    foot_y: float
    confidence: float
    evidence_frame_path: str = ""

    def with_evidence_path(self, path: Path) -> LineCrossingEvent:
        """Return an event carrying a portable evidence-frame path."""

        return replace(self, evidence_frame_path=path.as_posix())

    def as_row(self) -> dict[str, Any]:
        """Return a stable CSV-compatible mapping."""

        return asdict(self)


@dataclass(slots=True)
class _PendingCrossing:
    previous_side: LineSide
    current_side: LineSide


@dataclass(slots=True)
class _TrackState:
    observations: int
    last_seen_frame: int
    stable_side: LineSide | None
    reference_point: Point
    reference_distance: float
    last_nonzero_point: Point | None
    pending: _PendingCrossing | None = None
    last_event_frame: int | None = None


class LineCounter:
    """Count each Track ID at most once per direction across a finite line."""

    def __init__(
        self,
        line: DirectedLine,
        *,
        min_track_age: int = 5,
        min_displacement_pixels: float = 15.0,
        cooldown_frames: int = 30,
        max_track_gap_frames: int = 30,
    ) -> None:
        if min_track_age < 2:
            raise CountingError("min_track_age must be at least 2")
        if min_displacement_pixels < 0:
            raise CountingError("min_displacement_pixels must be non-negative")
        if cooldown_frames < 0:
            raise CountingError("cooldown_frames must be non-negative")
        if max_track_gap_frames < 0:
            raise CountingError("max_track_gap_frames must be non-negative")
        self.line = line
        self.min_track_age = min_track_age
        self.min_displacement_pixels = min_displacement_pixels
        self.cooldown_frames = cooldown_frames
        self.max_track_gap_frames = max_track_gap_frames
        self._states: dict[int, _TrackState] = {}
        self._counted_directions: dict[int, set[EventType]] = {}
        self._event_sequence = 0
        self._last_frame_id = 0
        self.total_enter = 0
        self.total_exit = 0

    def process_frame(
        self,
        records: Sequence[TrackRecord],
        *,
        frame_id: int,
    ) -> list[LineCrossingEvent]:
        """Update all tracks for one frame and return newly accepted crossings."""

        if frame_id <= self._last_frame_id:
            raise CountingError(
                f"LineCounter frame IDs must increase: previous={self._last_frame_id}, "
                f"current={frame_id}"
            )
        self._last_frame_id = frame_id
        self._prune_stale_tracks(frame_id)
        seen_track_ids: set[int] = set()
        events: list[LineCrossingEvent] = []
        for record in records:
            if record.frame_id != frame_id:
                raise CountingError(
                    f"Track record frame {record.frame_id} does not match current frame {frame_id}"
                )
            if record.track_id in seen_track_ids:
                raise CountingError(
                    f"Track ID {record.track_id} appears more than once in frame {frame_id}"
                )
            seen_track_ids.add(record.track_id)
            event = self._process_record(record)
            if event is not None:
                events.append(event)
        return events

    def _process_record(self, record: TrackRecord) -> LineCrossingEvent | None:
        point = (record.foot_x, record.foot_y)
        side = self.line.side_of(point)
        state = self._states.get(record.track_id)
        if state is None:
            stable_side = side if side is not LineSide.ON_LINE else None
            self._states[record.track_id] = _TrackState(
                observations=1,
                last_seen_frame=record.frame_id,
                stable_side=stable_side,
                reference_point=point,
                reference_distance=abs(self.line.signed_distance(point)),
                last_nonzero_point=point if stable_side is not None else None,
            )
            return None

        state.observations += 1
        state.last_seen_frame = record.frame_id
        if side is LineSide.ON_LINE:
            return None
        if state.stable_side is None:
            self._commit_side(state, side, point)
            return None
        if side is state.stable_side:
            state.pending = None
            signed_distance = abs(self.line.signed_distance(point))
            if signed_distance > state.reference_distance:
                state.reference_point = point
                state.reference_distance = signed_distance
            state.last_nonzero_point = point
            return None

        if state.pending is None:
            if state.last_nonzero_point is None or not self.line.intersects_motion(
                state.last_nonzero_point, point
            ):
                self._commit_side(state, side, point)
                return None
            state.pending = _PendingCrossing(state.stable_side, side)

        state.last_nonzero_point = point
        displacement = euclidean_distance(state.reference_point, point)
        if state.observations < self.min_track_age:
            return None
        if displacement < self.min_displacement_pixels:
            return None
        if (
            state.last_event_frame is not None
            and record.frame_id - state.last_event_frame < self.cooldown_frames
        ):
            return None

        event_type = _as_event_type(self.line.event_type_for(side))
        counted = self._counted_directions.setdefault(record.track_id, set())
        if event_type in counted:
            self._commit_side(state, side, point)
            return None

        self._event_sequence += 1
        event = LineCrossingEvent(
            event_id=f"event_{self._event_sequence:06d}",
            track_id=record.track_id,
            event_type=event_type,
            frame_id=record.frame_id,
            timestamp_ms=record.timestamp_ms,
            previous_side=state.stable_side.value,
            current_side=side.value,
            foot_x=record.foot_x,
            foot_y=record.foot_y,
            confidence=record.confidence,
        )
        counted.add(event_type)
        state.last_event_frame = record.frame_id
        if event_type == "enter":
            self.total_enter += 1
        else:
            self.total_exit += 1
        self._commit_side(state, side, point)
        return event

    def _commit_side(self, state: _TrackState, side: LineSide, point: Point) -> None:
        state.stable_side = side
        state.reference_point = point
        state.reference_distance = abs(self.line.signed_distance(point))
        state.last_nonzero_point = point
        state.pending = None

    def _prune_stale_tracks(self, frame_id: int) -> None:
        stale_ids = [
            track_id
            for track_id, state in self._states.items()
            if frame_id - state.last_seen_frame > self.max_track_gap_frames
        ]
        for track_id in stale_ids:
            del self._states[track_id]


def _as_event_type(value: str) -> EventType:
    if value == "enter":
        return "enter"
    if value == "exit":
        return "exit"
    raise CountingError(f"Unknown line crossing event type: {value}")
