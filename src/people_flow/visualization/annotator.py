"""OpenCV annotations for tracked people."""

from __future__ import annotations

import cv2

from people_flow.datasets.video_source import Frame
from people_flow.tracking.track_record import TrackRecord


def annotate_frame(frame: Frame, records: list[TrackRecord]) -> Frame:
    """Return a copy with person boxes, stable IDs, confidence, and foot points."""

    annotated = frame.copy()
    for record in records:
        start = (round(record.x1), round(record.y1))
        end = (round(record.x2), round(record.y2))
        foot = (round(record.foot_x), round(record.foot_y))
        color = (46, 204, 113)
        cv2.rectangle(annotated, start, end, color, 2)
        cv2.circle(annotated, foot, 4, (0, 165, 255), -1)
        label = f"person ID {record.track_id} {record.confidence:.2f}"
        text_y = max(18, start[1] - 8)
        cv2.putText(
            annotated,
            label,
            (start[0], text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated
