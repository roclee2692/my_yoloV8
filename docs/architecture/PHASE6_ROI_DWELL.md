# Phase 6 polygon ROI occupancy and dwell time

## Scope

Phase 6 adds polygon ROI membership, per-frame occupancy, observed enter/exit transitions, and per-Track dwell time. It reuses the Phase 3 Track stream and bottom-center foot point. It does not implement Ground Truth filtering, counting-error metrics, training, a dashboard, or LLM reporting.

## Data flow

```text
TrackRecord stream
  -> bottom-center foot point
  -> validated simple PolygonRegion
  -> RoiCounter per-Track state
       ├── current observed membership -> occupancy.csv
       ├── entry/exit intervals -> dwell_times.csv
       └── aggregate occupancy/dwell -> summary.json
  -> ROI polygon and current occupancy on annotated.mp4
```

Line counting and ROI analysis are independent consumers of the same ordered Track records. They can be enabled separately or together without duplicating detection or tracking.

## Polygon rules

- At least three ordered vertices are required.
- Consecutive duplicate vertices, zero-area polygons, non-finite coordinates, and self-intersections are rejected.
- Concave simple polygons are supported.
- A foot point exactly on an edge or vertex is considered inside.
- Only the bbox bottom-center foot point determines membership.

## Membership and missing-Track semantics

`occupancy.csv` represents observations, not extrapolation. Its `track_ids` and `occupancy` fields include only Tracks observed inside the ROI in that frame.

A Track first observed inside is recorded as an ROI entry because it newly joins the observed membership set. When an inside Track is later observed outside, an exit is recorded. A missing Track does not create a fabricated exit:

1. If the same ID returns within `max_track_gap_frames`, its active dwell interval continues and no duplicate entry is produced.
2. If the gap exceeds the threshold, its dwell interval closes at its last observed timestamp.
3. A later inside observation starts a new visit.
4. Timeout closure does not increment `total_exit`, because no outside observation proves an exit.
5. Video end closes reported dwell at each Track's last observation and does not invent an exit event.

This preserves the distinction between observed transitions and uncertainty caused by tracker loss.

## Output schemas

`occupancy.csv` contains one row per processed frame:

```text
frame_id,timestamp_ms,roi_name,occupancy,track_ids,entered_track_ids,exited_track_ids
```

The three Track-ID columns contain compact JSON arrays so empty and multiple-ID states remain unambiguous.

`dwell_times.csv` contains one row for each Track that entered the ROI:

```text
track_id,roi_name,first_entry_frame,first_entry_timestamp_ms,last_seen_frame,
last_seen_timestamp_ms,entry_count,exit_count,total_dwell_ms,total_dwell_seconds,
is_inside_at_end
```

`summary.json` includes all required Phase 6 values:

```text
total_enter,total_exit,maximum_occupancy,average_occupancy,peak_time,
average_dwell_seconds,median_dwell_seconds,processed_frames,processing_fps
```

It also records `roi_name`, `peak_frame_id`, `tracks_with_dwell`, and `inside_at_end`. `peak_time` is the source-video timestamp in milliseconds for the first frame that reaches the maximum occupancy. Empty dwell sets produce zero averages instead of NaN.

## Verification boundary

Unit tests cover boundary inclusion, membership transitions, multiple visits, short-gap recovery, long-gap closure, dwell accumulation, peak/average occupancy, median dwell, duplicate IDs, and frame ordering. A generated three-frame MP4 integration test proves all CSV/JSON outputs without a model or GPU. Real GPU output remains a functional smoke test until Phase 7 supplies Ground Truth.
