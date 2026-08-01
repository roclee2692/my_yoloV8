"""Convert persistent Ultralytics tracking results into typed records."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from people_flow.datasets.video_source import Frame
from people_flow.tracking.track_record import TrackRecord


class TrackingDetector(Protocol):
    """Minimal detector contract used by the model-free integration test."""

    def track(
        self,
        frame: Frame,
        *,
        tracker: str,
        classes: Sequence[int],
        confidence: float,
        iou: float,
        imgsz: int,
    ) -> Any:
        """Return an Ultralytics-compatible tracking result."""


class TrackerRunner:
    """Run one configured tracker and retain IDs through detector persistence."""

    def __init__(
        self,
        detector: TrackingDetector,
        *,
        tracker: str,
        classes: tuple[int, ...],
        confidence: float,
        iou: float,
        imgsz: int,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.classes = classes
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz

    def process(self, frame: Frame, *, frame_id: int, timestamp_ms: float) -> list[TrackRecord]:
        """Return person-only records for one frame; empty detections are valid."""

        results = self.detector.track(
            frame,
            tracker=self.tracker,
            classes=self.classes,
            confidence=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
        )
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or getattr(boxes, "id", None) is None:
            return []

        ids = boxes.id.int().cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.float().cpu().tolist()
        coordinates = boxes.xyxy.float().cpu().tolist()
        records: list[TrackRecord] = []
        for track_id, class_id, confidence, bbox in zip(
            ids, classes, confidences, coordinates, strict=True
        ):
            if int(class_id) != 0:
                continue
            if len(bbox) != 4:
                continue
            bbox_values = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )
            records.append(
                TrackRecord.from_bbox(
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    track_id=int(track_id),
                    class_id=int(class_id),
                    confidence=float(confidence),
                    bbox=bbox_values,
                )
            )
        return records
