"""OpenCV annotations for tracked people."""

from __future__ import annotations

import cv2
import numpy as np

from people_flow.counting.geometry import DirectedLine, PolygonRegion
from people_flow.datasets.video_source import Frame
from people_flow.tracking.track_record import TrackRecord


def annotate_frame(
    frame: Frame,
    records: list[TrackRecord],
    *,
    counting_line: DirectedLine | None = None,
    total_enter: int = 0,
    total_exit: int = 0,
    roi_region: PolygonRegion | None = None,
    roi_occupancy: int = 0,
) -> Frame:
    """Return a copy with tracks and optional directional-counting state."""

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

    if roi_region is not None:
        vertices = np.asarray(roi_region.points, dtype=np.int32).reshape((-1, 1, 2))
        roi_color = (0, 215, 255)
        cv2.polylines(annotated, [vertices], True, roi_color, 3, cv2.LINE_AA)
        anchor_x = max(12, min(round(point[0]) for point in roi_region.points))
        anchor_y = max(48, min(round(point[1]) for point in roi_region.points) - 10)
        cv2.putText(
            annotated,
            f"ROI {roi_region.name}: {roi_occupancy}",
            (anchor_x, anchor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            roi_color,
            2,
            cv2.LINE_AA,
        )
    if counting_line is not None:
        p1 = (round(counting_line.p1[0]), round(counting_line.p1[1]))
        p2 = (round(counting_line.p2[0]), round(counting_line.p2[1]))
        line_color = (255, 80, 180)
        cv2.arrowedLine(
            annotated,
            p1,
            p2,
            line_color,
            3,
            cv2.LINE_AA,
            tipLength=0.03,
        )
        cv2.circle(annotated, p1, 6, line_color, -1)
        cv2.circle(annotated, p2, 6, line_color, -1)
        caption = (
            f"ENTER {total_enter}  EXIT {total_exit}  enter_side={counting_line.enter_side.value}"
        )
        (text_width, text_height), _baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(annotated, (12, 10), (28 + text_width, 24 + text_height), (20, 20, 20), -1)
        cv2.putText(
            annotated,
            caption,
            (20, 20 + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            line_color,
            2,
            cv2.LINE_AA,
        )
    return annotated
