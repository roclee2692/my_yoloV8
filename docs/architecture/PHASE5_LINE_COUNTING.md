# Phase 5 directional line counting

## Scope

Phase 5 adds directional line-crossing counts to the existing person detection and tracking pipeline. It does not add ROI occupancy, dwell time, Ground Truth evaluation, model training, or a web dashboard.

## Data flow

```text
TrackRecord stream
  -> bottom-center foot point
  -> per-Track side and age state
  -> finite directed-line intersection
  -> age, displacement, cooldown, and direction de-duplication gates
  -> enter/exit event
  -> events.csv + evidence/<event>.jpg + annotated video totals
```

The exact same `LineCounter` accepts any ordered `TrackRecord` stream. Phase 7 can therefore feed MOT Ground Truth records through this implementation instead of maintaining a second counting algorithm.

## Geometry and direction

The directed line is defined by `p1 -> p2`. The normalized two-dimensional cross product classifies a point as `positive`, `negative`, or `on_line`. With the default `enter_side: positive`, a stable transition from the negative side to the positive side is `enter`, and the reverse transition is `exit`. Changing `enter_side` swaps those meanings without changing the line coordinates.

A crossing is accepted only when the observed foot-point motion intersects the finite `p1-p2` segment. Crossing the mathematical infinite extension outside either endpoint does not count.

## State and duplicate protection

Each live Track ID retains its age, last observation, last stable non-zero side, stable reference foot point, and a pending side transition. The counter applies these gates:

1. `min_track_age` rejects immature trajectories.
2. `min_displacement_pixels` measures movement from the last stable-side reference to the current foot point.
3. `cooldown_frames` suppresses rapid subsequent events for the same Track ID.
4. A `(track_id, event_type)` pair can be emitted only once, preventing repeated same-direction counts.
5. State survives gaps up to `max_track_gap_frames`; a longer gap starts a new local trajectory and cannot infer a crossing across the missing interval.
6. `on_line` observations do not erase the last stable side, allowing a transition that lands exactly on the line to resolve on the next stable side.

Track IDs are supplied by ByteTrack or BoT-SORT. If a tracker assigns a different ID after a long occlusion, the counter cannot know that the IDs represent the same person; this remains an explicit tracking limitation.

## Outputs

When counting is enabled, `events.csv` is created even if no event occurs. Its schema is:

```text
event_id,track_id,event_type,frame_id,timestamp_ms,previous_side,current_side,
foot_x,foot_y,confidence,evidence_frame_path
```

Every emitted event is paired with a JPEG evidence frame. The CSV stores a path relative to the run directory so that a run can be moved as a unit. The annotated video shows the directed line and cumulative enter/exit totals.

Generated media and CSV files remain under ignored `runs/` directories. Reusing an output directory requires explicit `--overwrite`; stale evidence frames are removed only inside that selected run directory.

## Verification boundary

Model-free unit tests cover side classification, direction swapping, finite-segment intersection, minimum age and displacement, cooldown, short-gap recovery, long-gap reset, and same-direction de-duplication. A generated-video integration test proves event CSV and evidence-image output. Real GPU processing is a smoke test only until Phase 7 supplies Ground Truth.
