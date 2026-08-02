"""Track-ID-based polygon occupancy and dwell-time state machine."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from people_flow.counting.dwell_time import DwellTimeRecord, RoiSummary, build_roi_summary
from people_flow.counting.geometry import PolygonRegion
from people_flow.errors import CountingError
from people_flow.tracking.track_record import TrackRecord


@dataclass(frozen=True, slots=True)
class RoiFrameSnapshot:
    """Observed ROI membership and transitions for one video frame."""

    frame_id: int
    timestamp_ms: float
    roi_name: str
    occupancy: int
    track_ids: tuple[int, ...]
    entered_track_ids: tuple[int, ...]
    exited_track_ids: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable mapping for output adapters."""

        return asdict(self)


@dataclass(slots=True)
class _TrackRoiState:
    last_seen_frame: int
    last_seen_timestamp_ms: float
    is_inside: bool = False
    first_entry_frame: int | None = None
    first_entry_timestamp_ms: float | None = None
    current_entry_timestamp_ms: float | None = None
    entry_count: int = 0
    exit_count: int = 0
    accumulated_dwell_ms: float = 0.0


class RoiCounter:
    """Measure observed occupancy and per-ID dwell inside one polygon ROI."""

    def __init__(self, region: PolygonRegion, *, max_track_gap_frames: int = 30) -> None:
        if max_track_gap_frames < 0:
            raise CountingError("max_track_gap_frames must be non-negative")
        self.region = region
        self.max_track_gap_frames = max_track_gap_frames
        self.total_enter = 0
        self.total_exit = 0
        self.maximum_occupancy = 0
        self.occupancy_sum = 0
        self.peak_time: float | None = None
        self.peak_frame_id: int | None = None
        self.processed_frames = 0
        self._last_frame_id = 0
        self._last_timestamp_ms = -math.inf
        self._states: dict[int, _TrackRoiState] = {}

    def process_frame(
        self,
        records: Sequence[TrackRecord],
        *,
        frame_id: int,
        timestamp_ms: float,
    ) -> RoiFrameSnapshot:
        """Update ROI state from one ordered frame of Track records."""

        if frame_id <= self._last_frame_id:
            raise CountingError(
                f"RoiCounter frame IDs must increase: previous={self._last_frame_id}, "
                f"current={frame_id}"
            )
        if not math.isfinite(timestamp_ms) or timestamp_ms < self._last_timestamp_ms:
            raise CountingError(
                "RoiCounter timestamps must be finite and non-decreasing: "
                f"previous={self._last_timestamp_ms}, current={timestamp_ms}"
            )
        self._expire_stale_tracks(frame_id)
        self._last_frame_id = frame_id
        self._last_timestamp_ms = timestamp_ms

        seen_track_ids: set[int] = set()
        inside_track_ids: list[int] = []
        entered_track_ids: list[int] = []
        exited_track_ids: list[int] = []
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
            is_inside = self.region.contains((record.foot_x, record.foot_y))
            state = self._states.get(record.track_id)
            if state is None:
                state = _TrackRoiState(
                    last_seen_frame=frame_id,
                    last_seen_timestamp_ms=timestamp_ms,
                )
                self._states[record.track_id] = state

            if is_inside and not state.is_inside:
                self._enter(state, frame_id=frame_id, timestamp_ms=timestamp_ms)
                entered_track_ids.append(record.track_id)
            elif not is_inside and state.is_inside:
                self._exit(state, timestamp_ms=timestamp_ms)
                exited_track_ids.append(record.track_id)

            state.last_seen_frame = frame_id
            state.last_seen_timestamp_ms = timestamp_ms
            if is_inside:
                inside_track_ids.append(record.track_id)

        inside = tuple(sorted(inside_track_ids))
        entered = tuple(sorted(entered_track_ids))
        exited = tuple(sorted(exited_track_ids))
        occupancy = len(inside)
        self.processed_frames += 1
        self.occupancy_sum += occupancy
        if self.peak_frame_id is None or occupancy > self.maximum_occupancy:
            self.maximum_occupancy = occupancy
            self.peak_frame_id = frame_id
            self.peak_time = timestamp_ms
        return RoiFrameSnapshot(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            roi_name=self.region.name,
            occupancy=occupancy,
            track_ids=inside,
            entered_track_ids=entered,
            exited_track_ids=exited,
        )

    def dwell_records(self) -> tuple[DwellTimeRecord, ...]:
        """Return finalized dwell values through each Track's last observation."""

        records: list[DwellTimeRecord] = []
        for track_id, state in sorted(self._states.items()):
            if state.first_entry_frame is None or state.first_entry_timestamp_ms is None:
                continue
            total_dwell_ms = state.accumulated_dwell_ms
            if state.is_inside:
                if state.current_entry_timestamp_ms is None:
                    raise CountingError(f"Track {track_id} has inconsistent active dwell state")
                total_dwell_ms += max(
                    0.0,
                    state.last_seen_timestamp_ms - state.current_entry_timestamp_ms,
                )
            records.append(
                DwellTimeRecord(
                    track_id=track_id,
                    roi_name=self.region.name,
                    first_entry_frame=state.first_entry_frame,
                    first_entry_timestamp_ms=state.first_entry_timestamp_ms,
                    last_seen_frame=state.last_seen_frame,
                    last_seen_timestamp_ms=state.last_seen_timestamp_ms,
                    entry_count=state.entry_count,
                    exit_count=state.exit_count,
                    total_dwell_ms=total_dwell_ms,
                    total_dwell_seconds=total_dwell_ms / 1000.0,
                    is_inside_at_end=state.is_inside,
                )
            )
        return tuple(records)

    def build_summary(self, *, processing_fps: float) -> RoiSummary:
        """Build aggregate occupancy and dwell metrics for processed frames."""

        dwell_records = self.dwell_records()
        return build_roi_summary(
            roi_name=self.region.name,
            total_enter=self.total_enter,
            total_exit=self.total_exit,
            maximum_occupancy=self.maximum_occupancy,
            occupancy_sum=self.occupancy_sum,
            peak_time=self.peak_time,
            peak_frame_id=self.peak_frame_id,
            dwell_records=dwell_records,
            processed_frames=self.processed_frames,
            processing_fps=processing_fps,
        )

    def _enter(self, state: _TrackRoiState, *, frame_id: int, timestamp_ms: float) -> None:
        state.is_inside = True
        state.current_entry_timestamp_ms = timestamp_ms
        state.entry_count += 1
        if state.first_entry_frame is None:
            state.first_entry_frame = frame_id
            state.first_entry_timestamp_ms = timestamp_ms
        self.total_enter += 1

    def _exit(self, state: _TrackRoiState, *, timestamp_ms: float) -> None:
        if state.current_entry_timestamp_ms is None:
            raise CountingError("Cannot exit an ROI without an active entry timestamp")
        state.accumulated_dwell_ms += max(0.0, timestamp_ms - state.current_entry_timestamp_ms)
        state.current_entry_timestamp_ms = None
        state.is_inside = False
        state.exit_count += 1
        self.total_exit += 1

    def _expire_stale_tracks(self, frame_id: int) -> None:
        for state in self._states.values():
            if not state.is_inside:
                continue
            if frame_id - state.last_seen_frame <= self.max_track_gap_frames:
                continue
            if state.current_entry_timestamp_ms is None:
                raise CountingError("Active ROI state is missing its entry timestamp")
            state.accumulated_dwell_ms += max(
                0.0,
                state.last_seen_timestamp_ms - state.current_entry_timestamp_ms,
            )
            state.current_entry_timestamp_ms = None
            state.is_inside = False
