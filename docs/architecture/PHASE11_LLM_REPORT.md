# Phase 11 structured LLM report

## Scope and decision

Phase 11 adds a reproducible Markdown report pipeline for completed people-flow
runs. The reporting layer accepts exactly three structured files:

- `summary.json`;
- `runtime_metrics.json`;
- `evaluation.json`.

It does not open `annotated.mp4`, the original video, evidence frames, `tracks.csv`,
or any other detection input. The generated `report_metadata.json` records
`raw_video_read: false` and a SHA-256 fingerprint of the bounded evidence payload.

## Architecture

```text
three validated JSON objects
    -> cross-file evidence reconciliation
    -> bounded narrative request
       -> deterministic offline fallback, or
       -> explicit OpenAI-compatible endpoint
    -> number-free narrative validation
    -> deterministic metric/table renderer
    -> report.md + report_metadata.json
    -> optional Dashboard preview/download
```

The LLM never owns numeric claims. It may return only a strict JSON narrative whose
text contains no numeric digits. Exact model, tracker, frame, count, occupancy,
dwell, runtime, latency, memory, and error values are inserted by the deterministic
renderer from validated evidence. A response that contains digits, additional fields,
Markdown wrappers, or malformed JSON is rejected.

This design prevents an LLM from silently inventing a new count or performance value.

## Evidence validation

`ReportEvidence` rejects a report before prompting when:

- any of the three files is absent or malformed;
- input and processed frames differ;
- summary and runtime frame counts differ;
- summary and runtime processing FPS differ;
- Ground Truth is unavailable but an error metric is populated with zero or another value;
- Ground Truth is available but required error metrics are missing.

When Ground Truth is unavailable, every report contains the exact statement:

> 本次结果没有人工或数据集 Ground Truth 验证，仅为模型预测结果。

Unknown accuracy metrics remain unavailable and are never presented as zero.

## Offline reproducible command

The deterministic provider exists for CI and environments with no configured LLM. It
is explicitly recorded as `llm_used: false` and is not presented as an external model
call.

```powershell
people-flow report `
  --run-dir runs\experiments\C_yolo26n_botsort `
  --provider deterministic
```

Equivalent module and script commands:

```powershell
python -m people_flow.reporting.cli `
  --run-dir runs\experiments\C_yolo26n_botsort

python scripts\generate_llm_report.py `
  --run-dir runs\experiments\C_yolo26n_botsort
```

## Explicit OpenAI-compatible provider

No provider, endpoint, model, or API key is silently selected. The secret remains in
an environment variable and is never stored in config, metadata, logs, or Git.

```powershell
$env:PEOPLE_FLOW_LLM_API_KEY = "<set locally; never commit>"

people-flow report `
  --run-dir runs\experiments\C_yolo26n_botsort `
  --provider openai-compatible `
  --endpoint https://example.invalid/v1/chat/completions `
  --model your-model-name `
  --api-key-env PEOPLE_FLOW_LLM_API_KEY `
  --overwrite
```

The endpoint must be an explicit absolute HTTP(S) URL without embedded credentials.
Missing connection settings or a missing environment variable fails before network
access. Provider responses are validated before either report output is replaced.

## Report structure

`report.md` contains:

1. `Executive Summary`;
2. video and run overview;
3. directional line and ROI results with distinct definitions;
4. peak occupancy and peak time;
5. dwell-time summary;
6. runtime performance;
7. Ground Truth and error availability;
8. recommended next steps;
9. further questions;
10. limitations, assumptions, conclusion, and confidence statement.

Supporting exact values use Markdown tables. Narrative interpretation appears next to
the relevant table rather than hiding the takeaway in metadata.

## Evidence presentation decision

The three permitted inputs contain aggregate run metrics but no frame-level time series.
The Markdown report therefore uses exact audit tables and adjacent interpretation. It
intentionally does not create a chart from invented intermediate values and does not
read `occupancy.csv` merely to make a visual. Time-series visualization remains in the
Phase 10 Dashboard, whose evidence contract explicitly includes frame-level occupancy.
## Actual Phase 11 validation

The existing `C_yolo26n_botsort` Phase 8 output generated a real local report and
metadata. The evidence fingerprint was:

```text
1ff2faab538fcc069d5bbd6488341a59db3c9424f0456f7e1047af0f91b04aeb
```

The run used the deterministic offline provider because no external endpoint or API
key was supplied. Therefore `report_metadata.json` truthfully records:

```json
{
  "provider": "deterministic",
  "model": null,
  "llm_used": false,
  "ground_truth_available": false,
  "raw_video_read": false
}
```

No live external LLM result is claimed. The optional network adapter is covered with a
mock transport in CI, including explicit endpoint, environment-key lookup, response
schema validation, and failure before network access when the key is absent.

## Known limitations

- The reusable network adapter currently targets a chat-completions-compatible JSON
  contract; provider-specific streaming or tool calls are outside this phase.
- Narrative text is Chinese-first and intentionally cannot contain numeric digits.
- The report is a snapshot of one completed run, not a live monitoring document.
- Report accuracy cannot exceed the supplied evaluation evidence.
- Current sample evidence has no Ground Truth, so its report cannot answer which model
  is more accurate.
