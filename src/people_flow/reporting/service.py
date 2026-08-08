"""Phase 11 report orchestration and reproducible output metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from people_flow.errors import ReportGenerationError
from people_flow.reporting.evidence import ReportEvidence, load_report_evidence
from people_flow.reporting.narrative import (
    OpenAICompatibleNarrativeClient,
    OpenAICompatibleSettings,
    deterministic_narrative,
)
from people_flow.reporting.renderer import ReportProvenance, render_markdown_report

ProviderName = Literal["deterministic", "openai-compatible"]


@dataclass(frozen=True, slots=True)
class ReportResult:
    """Persisted Phase 11 report and provenance metadata."""

    report_path: Path
    metadata_path: Path
    provider: ProviderName
    model: str | None
    evidence_fingerprint: str
    llm_used: bool


def generate_report(
    run_dir: Path,
    *,
    output: Path | None = None,
    metadata_output: Path | None = None,
    provider: ProviderName = "deterministic",
    endpoint: str | None = None,
    model: str | None = None,
    api_key_env: str = "PEOPLE_FLOW_LLM_API_KEY",
    timeout_seconds: float = 60.0,
    overwrite: bool = False,
    generated_at: datetime | None = None,
) -> ReportResult:
    """Generate a report from three structured inputs and no raw media."""

    resolved_run = run_dir.expanduser().resolve()
    report_path = (output or (resolved_run / "report.md")).expanduser().resolve()
    metadata_path = (
        (metadata_output or report_path.with_name("report_metadata.json")).expanduser().resolve()
    )
    _validate_output_targets(report_path, metadata_path, overwrite=overwrite)
    evidence = load_report_evidence(resolved_run)
    if provider == "deterministic":
        narrative = deterministic_narrative(evidence)
        report_model = None
        llm_used = False
    else:
        if endpoint is None or model is None:
            raise ReportGenerationError(
                "openai-compatible provider requires --endpoint and --model"
            )
        client = OpenAICompatibleNarrativeClient(
            OpenAICompatibleSettings(
                endpoint=endpoint,
                model=model,
                api_key_env=api_key_env,
                timeout_seconds=timeout_seconds,
            )
        )
        narrative = client.generate(evidence)
        report_model = model
        llm_used = True
    timestamp = generated_at or datetime.now(timezone.utc)
    provenance = ReportProvenance(
        provider=provider,
        model=report_model,
        generated_at=timestamp,
        evidence_fingerprint=evidence.fingerprint,
    )
    report_text = render_markdown_report(evidence, narrative, provenance)
    metadata = _metadata_payload(
        evidence,
        provider=provider,
        model=report_model,
        llm_used=llm_used,
        generated_at=timestamp,
    )
    _write_text_atomic(report_path, report_text)
    try:
        _write_text_atomic(
            metadata_path,
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        )
    except ReportGenerationError:
        report_path.unlink(missing_ok=True)
        raise
    return ReportResult(
        report_path=report_path,
        metadata_path=metadata_path,
        provider=provider,
        model=report_model,
        evidence_fingerprint=evidence.fingerprint,
        llm_used=llm_used,
    )


def _metadata_payload(
    evidence: ReportEvidence,
    *,
    provider: ProviderName,
    model: str | None,
    llm_used: bool,
    generated_at: datetime,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "provider": provider,
        "model": model,
        "llm_used": llm_used,
        "generated_at": generated_at.astimezone(timezone.utc).isoformat(),
        "evidence_sha256": evidence.fingerprint,
        "input_files": ["summary.json", "runtime_metrics.json", "evaluation.json"],
        "ground_truth_available": evidence.evaluation.ground_truth_available,
        "raw_video_read": False,
    }


def _validate_output_targets(report: Path, metadata: Path, *, overwrite: bool) -> None:
    if report == metadata:
        raise ReportGenerationError("Report and metadata outputs must be different files")
    existing = [path for path in (report, metadata) if path.exists()]
    if existing and not overwrite:
        raise ReportGenerationError(f"Report output already exists: {existing}. Use --overwrite.")


def _write_text_atomic(path: Path, content: str) -> None:
    partial = path.with_name(f".{path.name}.part")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(content, encoding="utf-8", newline="\n")
        partial.replace(path)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise ReportGenerationError(f"Unable to write report output: {path}") from exc
