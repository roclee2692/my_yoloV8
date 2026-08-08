"""Bounded LLM narrative contract with no model-generated numeric claims."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from people_flow.errors import ReportGenerationError
from people_flow.reporting.evidence import ReportEvidence

HttpTransport = Callable[[urllib.request.Request, float], bytes]
_DIGIT_PATTERN = re.compile(r"\d")

SYSTEM_PROMPT = """你是人流分析报告助手。你只能解释用户提供的结构化 JSON。
禁止推断、补全或编造任何指标。所有精确数字由确定性报告渲染器添加，因此你的文本中禁止出现阿拉伯数字。
不要声称看过视频、图片、轨迹 CSV 或任何未提供的数据。
只返回符合指定结构的 JSON 对象，不要返回 Markdown 或代码围栏。"""


class ReportNarrative(BaseModel):
    """Number-free prose supplied by an LLM or deterministic fallback."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    executive_summary: tuple[str, ...] = Field(min_length=2, max_length=4)
    performance_interpretation: str = Field(min_length=1, max_length=1000)
    counting_interpretation: str = Field(min_length=1, max_length=1000)
    limitations: tuple[str, ...] = Field(min_length=1, max_length=6)
    conclusion: str = Field(min_length=1, max_length=1000)
    confidence_statement: str = Field(min_length=1, max_length=600)

    @model_validator(mode="after")
    def reject_numeric_claims(self) -> ReportNarrative:
        """Reserve every numeric claim for the deterministic renderer."""

        fields = (
            *self.executive_summary,
            self.performance_interpretation,
            self.counting_interpretation,
            *self.limitations,
            self.conclusion,
            self.confidence_statement,
        )
        if any(_DIGIT_PATTERN.search(value) for value in fields):
            raise ValueError("LLM narrative must not contain numeric digits")
        return self


@dataclass(frozen=True, slots=True)
class OpenAICompatibleSettings:
    """Connection settings without storing the API key itself."""

    endpoint: str
    model: str
    api_key_env: str = "PEOPLE_FLOW_LLM_API_KEY"
    timeout_seconds: float = 60.0

    def validate(self) -> None:
        """Reject ambiguous endpoints and embedded credentials."""

        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ReportGenerationError("LLM endpoint must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ReportGenerationError("LLM endpoint must not contain credentials")
        if not self.model.strip():
            raise ReportGenerationError("LLM model must not be blank")
        if not self.api_key_env.strip():
            raise ReportGenerationError("LLM API key environment variable must not be blank")
        if self.timeout_seconds <= 0.0:
            raise ReportGenerationError("LLM timeout must be positive")


class OpenAICompatibleNarrativeClient:
    """Minimal JSON client for an explicitly configured chat-completions endpoint."""

    def __init__(
        self,
        settings: OpenAICompatibleSettings,
        *,
        transport: HttpTransport | None = None,
    ) -> None:
        settings.validate()
        self._settings = settings
        self._transport = transport or _urlopen_transport

    def generate(self, evidence: ReportEvidence) -> ReportNarrative:
        """Request and validate a bounded narrative without exposing secrets."""

        api_key = os.environ.get(self._settings.api_key_env)
        if not api_key:
            raise ReportGenerationError(
                f"LLM API key environment variable is not set: {self._settings.api_key_env}"
            )
        request_body = {
            "model": self._settings.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(evidence)},
            ],
        }
        request = urllib.request.Request(
            self._settings.endpoint,
            data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            response_bytes = self._transport(request, self._settings.timeout_seconds)
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise ReportGenerationError(f"LLM request failed: {exc}") from exc
        try:
            payload: Any = json.loads(response_bytes.decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReportGenerationError("LLM response does not match the expected schema") from exc
        if not isinstance(content, str):
            raise ReportGenerationError("LLM response content must be text")
        return parse_narrative(content)


def build_user_prompt(evidence: ReportEvidence) -> str:
    """Build the only prompt payload, containing exactly three JSON objects."""

    schema = {
        "executive_summary": ["两到四条无数字摘要"],
        "performance_interpretation": "无数字解释",
        "counting_interpretation": "无数字解释",
        "limitations": ["至少一条无数字限制"],
        "conclusion": "无数字结论",
        "confidence_statement": "无数字可信度说明",
    }
    return (
        "请依据 evidence 生成中文叙述。不得使用阿拉伯数字。\n"
        f"输出结构：{json.dumps(schema, ensure_ascii=False)}\n"
        f"evidence：{json.dumps(evidence.prompt_payload(), ensure_ascii=False, sort_keys=True)}"
    )


def parse_narrative(content: str) -> ReportNarrative:
    """Parse strict JSON and reject Markdown wrappers or invented digits."""

    try:
        payload: Any = json.loads(content)
        return ReportNarrative.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ReportGenerationError(f"LLM narrative validation failed: {exc}") from exc


def deterministic_narrative(evidence: ReportEvidence) -> ReportNarrative:
    """Build an explicitly non-LLM fallback for offline validation and CI."""

    no_ground_truth = not evidence.evaluation.ground_truth_available
    summary = (
        "本次运行已形成完整的检测、跟踪、计数和区域统计产物。",
        (
            "当前证据只能支持功能和运行表现判断，不能支持准确率结论。"
            if no_ground_truth
            else "当前报告已关联数据集标注，可结合误差指标判断计数表现。"
        ),
    )
    limitations = ["报告只使用结构化汇总文件，没有读取原始视频。"]
    if no_ground_truth:
        limitations.append("当前样例缺少独立标注，预测结果不能视为真实人数。")
    return ReportNarrative(
        executive_summary=summary,
        performance_interpretation=(
            "处理吞吐低于源视频播放速率，当前配置更适合离线分析而非实时处理。"
            if evidence.runtime_metrics.processing_fps < evidence.runtime_metrics.video_fps
            else "处理吞吐达到源视频播放速率，仍应在目标硬件和更长视频上复核稳定性。"
        ),
        counting_interpretation=(
            "计数线与区域指标来自稳定轨迹和脚点规则，二者含义不同，不应混为同一总人数。"
        ),
        limitations=tuple(limitations),
        conclusion=("系统已经能够产出可审计的人流分析报告，但准确率结论需要关联标注后再形成。"),
        confidence_statement=(
            "对文件完整性和运行性能的可信度较高；对计数准确性的可信度受标注缺失限制。"
            if no_ground_truth
            else "结构化误差指标已通过同一计数算法生成，结论仍受场景代表性限制。"
        ),
    )


def _urlopen_transport(request: urllib.request.Request, timeout: float) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return cast(bytes, response.read())
