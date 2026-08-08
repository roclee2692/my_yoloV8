"""Deterministic Markdown rendering for validated evidence and LLM prose."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from people_flow.reporting.evidence import ReportEvidence
from people_flow.reporting.narrative import ReportNarrative

NO_GROUND_TRUTH_NOTICE = "本次结果没有人工或数据集 Ground Truth 验证，仅为模型预测结果。"


@dataclass(frozen=True, slots=True)
class ReportProvenance:
    """Non-secret report generation metadata."""

    provider: str
    model: str | None
    generated_at: datetime
    evidence_fingerprint: str


def render_markdown_report(
    evidence: ReportEvidence,
    narrative: ReportNarrative,
    provenance: ReportProvenance,
) -> str:
    """Render exact metrics deterministically and prose from validated narrative."""

    summary = evidence.summary
    runtime = evidence.runtime_metrics
    evaluation = evidence.evaluation
    duration_seconds = runtime.processed_frames / runtime.video_fps
    realtime_ratio = runtime.processing_fps / runtime.video_fps
    peak_time = "不可用" if summary.peak_time is None else f"{summary.peak_time / 1000.0:.2f} 秒"

    lines = [
        "# 固定摄像头人流分析报告",
        "",
        "## Executive Summary",
        "",
    ]
    lines.extend(f"- **结论。** {item}" for item in narrative.executive_summary)
    if not evaluation.ground_truth_available:
        lines.extend(("", f"> {NO_GROUND_TRUTH_NOTICE}"))
    lines.extend(
        (
            "",
            "## 视频和运行概况",
            "",
            "**本次结构化产物记录了一次完整处理，输入帧与处理帧一致。**",
            "",
            "| 项目 | 数值 |",
            "|---|---:|",
            f"| 模型 | {runtime.model} |",
            f"| 跟踪器 | {runtime.tracker} |",
            f"| 设备 | {runtime.device} |",
            f"| 输入 / 处理帧 | {runtime.input_frames} / {runtime.processed_frames} |",
            f"| 视频 FPS | {runtime.video_fps:.2f} |",
            f"| 视频时长 | {duration_seconds:.2f} 秒 |",
            "",
            "## 人流与区域结果",
            "",
            f"**{narrative.counting_interpretation}**",
            "",
            "| 指标 | 数值 | 口径 |",
            "|---|---:|---|",
            f"| 计数线进入 | {evaluation.predicted_enter} | 轨迹穿越有向线 |",
            f"| 计数线离开 | {evaluation.predicted_exit} | 轨迹穿越有向线 |",
            f"| 计数线总事件 | {evaluation.predicted_total} | 进入与离开之和 |",
            f"| ROI 进入 | {summary.total_enter} | 区域成员状态变化 |",
            f"| ROI 离开 | {summary.total_exit} | 区域成员状态变化 |",
            f"| 峰值人数 | {summary.maximum_occupancy} | ROI 同时在场人数 |",
            f"| 平均人数 | {summary.average_occupancy:.3f} | 逐帧 ROI 人数均值 |",
            f"| 高峰时刻 | {peak_time} | 相对视频起点 |",
            "",
            "## 停留时间",
            "",
            "**停留时间来自同一 Track 在 ROI 中的累计观测，不等同于完整现实停留时长。**",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| 有停留记录的 Track | {summary.tracks_with_dwell} |",
            f"| 平均停留时间 | {summary.average_dwell_seconds:.3f} 秒 |",
            f"| 中位停留时间 | {summary.median_dwell_seconds:.3f} 秒 |",
            f"| 视频结束时仍在 ROI | {summary.inside_at_end} |",
            "",
            "## 模型运行性能",
            "",
            f"**{narrative.performance_interpretation}**",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| 处理 FPS | {runtime.processing_fps:.3f} |",
            f"| 相对实时倍速 | {realtime_ratio:.3f}× |",
            f"| 平均延迟 | {runtime.average_latency_ms:.3f} ms |",
            f"| 延迟 P95 | {runtime.p95_latency_ms:.3f} ms |",
            f"| 峰值 GPU 显存 | {runtime.peak_gpu_memory_mb:.3f} MB |",
            f"| 总运行时间 | {runtime.total_runtime_seconds:.3f} 秒 |",
            "",
            "## Ground Truth 与计数误差",
            "",
        )
    )
    if evaluation.ground_truth_available:
        lines.extend(_ground_truth_metrics(evidence))
    else:
        lines.extend(
            (
                f"**{NO_GROUND_TRUTH_NOTICE}**",
                "",
                f"数据说明：{evaluation.reason}",
                "",
                "计数绝对误差、MAE、MAPE、ROI occupancy MAE 和 ID Switch 均不可用，",
                "不能将这些空值解释为零误差。",
            )
        )
    lines.extend(
        (
            "",
            "## 建议的下一步",
            "",
            "1. 使用带 MOT Ground Truth 的序列重新运行同一评测流程。",
            "2. 只有在误差归因证明检测质量是主要瓶颈后，再考虑模型微调。",
            "3. 在目标摄像机、分辨率和部署硬件上复核吞吐及失败案例。",
            "",
            "## Further Questions",
            "",
            "- 当前场景是否代表计划部署地点的遮挡、密度和视角？",
            "- 计数线与 ROI 几何是否由业务人员确认？",
            "- 是否已有可关联的 MOT Ground Truth 用于准确率结论？",
            "",
            "## 局限、假设与结论可信度",
            "",
        )
    )
    lines.extend(f"- {item}" for item in narrative.limitations)
    lines.extend(
        (
            f"- {narrative.confidence_statement}",
            "",
            f"**结论：** {narrative.conclusion}",
            "",
            (
                f"<!-- provider={provenance.provider}; model={provenance.model or 'none'}; "
                f"generated_at={provenance.generated_at.astimezone(timezone.utc).isoformat()}; "
                f"evidence_sha256={provenance.evidence_fingerprint} -->"
            ),
            "",
        )
    )
    return "\n".join(lines)


def _ground_truth_metrics(evidence: ReportEvidence) -> tuple[str, ...]:
    evaluation = evidence.evaluation
    mape = "不可用（Ground Truth 总数为零）"
    if evaluation.count_mape is not None:
        mape = f"{evaluation.count_mape:.3f}%"
    return (
        "**本次运行已关联 Ground Truth，以下误差可用于评测。**",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| Ground Truth 进入 / 离开 | {evaluation.ground_truth_enter} / "
        f"{evaluation.ground_truth_exit} |",
        f"| 进入绝对误差 | {evaluation.enter_absolute_error:.3f} |",
        f"| 离开绝对误差 | {evaluation.exit_absolute_error:.3f} |",
        f"| 总计数绝对误差 | {evaluation.total_count_absolute_error:.3f} |",
        f"| Count MAE | {evaluation.count_mae:.3f} |",
        f"| Count MAPE | {mape} |",
        f"| Occupancy MAE | {evaluation.occupancy_mae:.3f} |",
        f"| ID Switch | {evaluation.id_switches} |",
    )
