"""Tests for prompt boundaries and LLM output validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from people_flow.errors import ReportGenerationError
from people_flow.reporting.evidence import load_report_evidence
from people_flow.reporting.narrative import (
    OpenAICompatibleNarrativeClient,
    OpenAICompatibleSettings,
    build_user_prompt,
    parse_narrative,
)


def _valid_narrative() -> dict[str, object]:
    return {
        "executive_summary": ["运行产物完整。", "准确率结论仍需标注支持。"],
        "performance_interpretation": "当前吞吐更适合离线处理。",
        "counting_interpretation": "计数线和区域人数采用不同口径。",
        "limitations": ["报告没有读取原始视频。"],
        "conclusion": "系统能够生成可审计报告。",
        "confidence_statement": "运行性能可信，计数准确率尚未验证。",
    }


def test_prompt_contains_only_structured_evidence(report_run: Path) -> None:
    """The LLM prompt must not expose video, Track CSV, or filesystem paths."""

    prompt = build_user_prompt(load_report_evidence(report_run))

    assert '"summary"' in prompt
    assert '"runtime_metrics"' in prompt
    assert '"evaluation"' in prompt
    assert "annotated.mp4" not in prompt
    assert "tracks.csv" not in prompt
    assert str(report_run) not in prompt


def test_narrative_rejects_any_model_generated_digit() -> None:
    """All numeric claims remain owned by the deterministic renderer."""

    payload = _valid_narrative()
    payload["conclusion"] = "准确率达到九成以上，误差为零。"
    parse_narrative(json.dumps(payload, ensure_ascii=False))
    payload["conclusion"] = "准确率达到 90%。"

    with pytest.raises(ReportGenerationError, match="numeric digits"):
        parse_narrative(json.dumps(payload, ensure_ascii=False))


def test_openai_compatible_client_uses_explicit_endpoint_and_env_key(
    report_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional network adapter should validate its response and avoid stored keys."""

    evidence = load_report_evidence(report_run)
    monkeypatch.setenv("TEST_REPORT_KEY", "secret-value")
    captured: dict[str, object] = {}

    def fake_transport(request: object, timeout: float) -> bytes:
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["authorization"] = request.headers["Authorization"]  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        content = json.dumps(_valid_narrative(), ensure_ascii=False)
        return json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    client = OpenAICompatibleNarrativeClient(
        OpenAICompatibleSettings(
            endpoint="https://llm.example/v1/chat/completions",
            model="example-model",
            api_key_env="TEST_REPORT_KEY",
            timeout_seconds=12.0,
        ),
        transport=fake_transport,
    )
    narrative = client.generate(evidence)

    assert narrative.conclusion == "系统能够生成可审计报告。"
    assert captured == {
        "url": "https://llm.example/v1/chat/completions",
        "authorization": "Bearer secret-value",
        "timeout": 12.0,
    }


def test_openai_compatible_client_requires_key_without_network(report_run: Path) -> None:
    """A missing credential should fail before transport execution."""

    called = False

    def fake_transport(request: object, timeout: float) -> bytes:
        nonlocal called
        called = True
        return b"{}"

    client = OpenAICompatibleNarrativeClient(
        OpenAICompatibleSettings(
            endpoint="http://localhost:9999/v1/chat/completions",
            model="local-model",
            api_key_env="DEFINITELY_MISSING_REPORT_KEY",
        ),
        transport=fake_transport,
    )
    with pytest.raises(ReportGenerationError, match="is not set"):
        client.generate(load_report_evidence(report_run))
    assert called is False
