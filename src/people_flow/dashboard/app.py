"""Streamlit interface for configuring, running, and inspecting people-flow jobs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import cv2
import streamlit as st
import torch
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from people_flow.config import TrackerName
from people_flow.dashboard.data import (
    DashboardRunData,
    build_results_archive,
    discover_completed_runs,
    load_comparison_rows,
    load_run_dashboard,
)
from people_flow.dashboard.service import (
    DashboardRunRequest,
    Point,
    draw_geometry_preview,
    load_video_preview,
    map_display_point,
    persist_uploaded_mp4,
    run_dashboard_job,
    validate_run_id,
)
from people_flow.errors import PeopleFlowError

_APP_TITLE = "基于 YOLO26 与多目标跟踪的人流分析系统"


def _project_root() -> Path:
    configured = os.environ.get("PEOPLE_FLOW_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    current = Path.cwd().resolve()
    if (current / "pyproject.toml").is_file():
        return current
    return Path(__file__).resolve().parents[3]


def _available_samples(root: Path) -> tuple[Path, ...]:
    samples_dir = root / "data" / "samples"
    if not samples_dir.is_dir():
        return ()
    return tuple(sorted(samples_dir.glob("*.mp4")))


def _initialize_geometry(source: Path, *, width: int, height: int) -> None:
    source_key = str(source)
    if st.session_state.get("geometry_source") == source_key:
        return
    st.session_state.geometry_source = source_key
    st.session_state.line_points = [
        (0.08 * width, 0.69 * height),
        (0.86 * width, 0.69 * height),
    ]
    st.session_state.roi_points = [
        (0.08 * width, 0.14 * height),
        (0.86 * width, 0.14 * height),
        (0.86 * width, 0.90 * height),
        (0.08 * width, 0.90 * height),
    ]
    st.session_state.geometry_last_click = None


def _source_picker(root: Path) -> Path | None:
    samples = _available_samples(root)
    sample_names = {path.name: path for path in samples}
    selected_sample: Path | None = None
    if sample_names:
        selected_name = st.selectbox("内置测试视频", tuple(sample_names), index=0)
        selected_sample = sample_names[selected_name]
    else:
        st.info("data/samples 中暂无 MP4；可以在下方上传本地视频。")

    upload = st.file_uploader("上传 MP4", type=("mp4",), accept_multiple_files=False)
    if upload is None:
        return selected_sample
    return persist_uploaded_mp4(
        upload.name,
        upload.getvalue(),
        upload_dir=root / "data" / "uploads",
    )


def _geometry_editor(source: Path) -> tuple[tuple[Point, ...], tuple[Point, ...]]:
    preview = load_video_preview(source)
    _initialize_geometry(source, width=preview.width, height=preview.height)
    line_points: list[Point] = st.session_state.line_points
    roi_points: list[Point] = st.session_state.roi_points

    st.caption(
        f"源视频：{preview.width}×{preview.height} · {preview.fps:.2f} FPS · "
        f"{preview.frame_count} 帧。红线为计数线，绿色区域为 ROI。"
    )
    target = st.radio("当前绘制对象", ("计数线", "ROI"), horizontal=True)
    first, second, third, fourth = st.columns(4)
    if first.button("撤销计数线端点", width="stretch"):
        if line_points:
            line_points.pop()
        st.rerun()
    if second.button("清空计数线", width="stretch"):
        line_points.clear()
        st.rerun()
    if third.button("撤销 ROI 顶点", width="stretch"):
        if roi_points:
            roi_points.pop()
        st.rerun()
    if fourth.button("清空 ROI", width="stretch"):
        roi_points.clear()
        st.rerun()

    annotated = draw_geometry_preview(
        preview.frame_rgb,
        line_points=line_points,
        roi_points=roi_points,
    )
    display_width = min(preview.width, 900)
    display_height = max(1, round(preview.height * display_width / preview.width))
    resized = cv2.resize(annotated, (display_width, display_height), interpolation=cv2.INTER_AREA)
    clicked = streamlit_image_coordinates(
        Image.fromarray(resized),
        width=display_width,
        key=f"geometry-{source}",
    )
    if (
        isinstance(clicked, dict)
        and isinstance(clicked.get("x"), (int, float))
        and isinstance(clicked.get("y"), (int, float))
    ):
        signature = (target, clicked.get("x"), clicked.get("y"), clicked.get("unix_time"))
        if signature != st.session_state.get("geometry_last_click"):
            st.session_state.geometry_last_click = signature
            point = map_display_point(
                float(clicked["x"]),
                float(clicked["y"]),
                display_size=(display_width, display_height),
                source_size=(preview.width, preview.height),
            )
            if target == "计数线":
                if len(line_points) < 2:
                    line_points.append(point)
                else:
                    st.warning("计数线已有两个端点；请先撤销或清空。")
            elif len(roi_points) < 20:
                roi_points.append(point)
            else:
                st.warning("ROI 最多支持 20 个顶点。")
            st.rerun()

    st.write(f"计数线端点：{len(line_points)}/2；ROI 顶点：{len(roi_points)}（至少 3 个）")
    return tuple(line_points), tuple(roi_points)


def _render_run_tab(root: Path) -> None:
    st.subheader("配置并运行")
    st.caption(
        "界面仅组织参数并调用 people_flow.pipeline.run_pipeline；"
        "检测、跟踪与计数逻辑不在界面层实现。"
    )
    left, right = st.columns((1, 1))
    with left:
        source = _source_picker(root)
        model = st.selectbox("检测模型", ("yolo26n.pt", "yolov8n.pt"), index=0)
        tracker_value = st.selectbox(
            "多目标跟踪器",
            ("bytetrack.yaml", "botsort.yaml"),
            index=0,
        )
        confidence = st.slider("置信度阈值", 0.05, 0.95, 0.25, 0.05)
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.insert(0, "0")
        device = st.selectbox("计算设备", devices, index=0)
        default_run_id = "dashboard_yolo26_bytetrack"
        run_id = st.text_input("运行 ID", value=default_run_id)
        overwrite = st.checkbox("覆盖同名输出目录", value=False)

    if source is None:
        with right:
            st.warning("请上传 MP4 或把测试视频放入 data/samples 后再配置几何区域。")
        return
    with right:
        line_points, roi_points = _geometry_editor(source)

    if not st.button("开始处理", type="primary", width="stretch"):
        return
    try:
        validated_run_id = validate_run_id(run_id)
        request = DashboardRunRequest(
            source=source,
            output_dir=root / "runs" / "dashboard" / validated_run_id,
            model=model,
            tracker=cast(TrackerName, tracker_value),
            device=device,
            confidence=confidence,
            line_points=line_points,
            roi_points=roi_points,
            overwrite=overwrite,
            weights_dir=root / "weights",
        )
        request.to_run_config()
        progress = st.progress(0.0, text="正在初始化模型与视频…")

        def update_progress(processed: int, total: int) -> None:
            ratio = processed / total if total > 0 else 0.0
            progress.progress(
                min(max(ratio, 0.0), 1.0),
                text=f"已处理 {processed}/{total} 帧",
            )

        with st.status("核心管线运行中", expanded=True) as status:
            result = run_dashboard_job(request, progress_callback=update_progress)
            status.update(label="处理完成", state="complete", expanded=False)
        st.session_state.selected_run = str(result.output_dir)
        st.success(f"输出已写入：{result.output_dir}")
    except (PeopleFlowError, ValueError, OSError) as exc:
        st.error(str(exc))


def _relative_label(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _render_result(data: DashboardRunData) -> None:
    if data.ground_truth_available:
        st.success(data.ground_truth_message)
    else:
        st.warning(data.ground_truth_message)

    peak_time = "—" if data.peak_time_ms is None else f"{data.peak_time_ms / 1000.0:.2f} s"
    columns = st.columns(6)
    values = (
        ("当前 ROI 人数", str(data.current_occupancy)),
        ("计数线进入", str(data.total_enter)),
        ("计数线离开", str(data.total_exit)),
        ("峰值人数", str(data.maximum_occupancy)),
        ("峰值时刻", peak_time),
        ("处理 FPS", f"{data.processing_fps:.2f}"),
    )
    for column, (label, value) in zip(columns, values, strict=True):
        column.metric(label, value)

    st.video(str(data.annotated_video))
    flow = data.flow_series()
    chart_left, chart_right = st.columns((2, 1))
    with chart_left:
        st.markdown("#### 人流时间曲线")
        if flow:
            st.line_chart(
                flow,
                x="timestamp_seconds",
                y=("occupancy", "cumulative_enter", "cumulative_exit"),
            )
        else:
            st.info("此运行未启用 ROI，因此没有 occupancy.csv。")
    with chart_right:
        st.markdown("#### 停留时间分布")
        dwell_distribution = [
            {"区间": bucket, "人数": count} for bucket, count in data.dwell_histogram().items()
        ]
        if data.dwell_times:
            st.bar_chart(dwell_distribution, x="区间", y="人数")
        else:
            st.info("此运行没有可用的停留时间记录。")

    if data.dwell_times:
        st.markdown("#### 停留时间明细")
        st.dataframe(data.dwell_times, width="stretch", hide_index=True)
    archive = build_results_archive(data)
    st.download_button(
        "下载本次运行全部结果（ZIP）",
        data=archive,
        file_name=f"{data.output_dir.name}-results.zip",
        mime="application/zip",
    )


def _render_results_tab(root: Path) -> None:
    st.subheader("运行结果")
    runs = discover_completed_runs(root / "runs")
    if not runs:
        st.info("尚未发现完整运行结果。完成一次处理后，本页会从 CSV/JSON 产物生成视图。")
        return
    labels = {_relative_label(path, root): path for path in runs}
    preferred = st.session_state.get("selected_run")
    index = 0
    if preferred:
        for candidate_index, path in enumerate(labels.values()):
            if str(path) == preferred:
                index = candidate_index
                break
    selected_label = st.selectbox("选择运行", tuple(labels), index=index)
    try:
        data = load_run_dashboard(labels[selected_label])
        st.caption(
            f"数据源：{_relative_label(data.output_dir, root)} · "
            f"已处理 {data.processed_frames} 帧。页面不重新推理。"
        )
        _render_result(data)
    except (PeopleFlowError, ValueError, OSError) as exc:
        st.error(str(exc))


def _render_experiments_tab(root: Path) -> None:
    st.subheader("Phase 8 基线实验")
    comparison_path = root / "runs" / "experiments" / "comparison.csv"
    try:
        rows = load_comparison_rows(comparison_path)
    except PeopleFlowError as exc:
        st.error(str(exc))
        return
    if not rows:
        st.info("尚未生成 runs/experiments/comparison.csv。")
        return
    if not any(row.get("ground_truth_available", "").lower() == "true" for row in rows):
        st.warning(
            "这些基线实验没有 Ground Truth；计数误差与 ID Switch 指标不可用，"
            "不能据此判断准确率优劣。"
        )
    st.dataframe(rows, width="stretch", hide_index=True)
    fps_rows = [
        {
            "实验": row["experiment_id"],
            "处理 FPS": float(row["processing_fps"]),
        }
        for row in rows
    ]
    st.bar_chart(fps_rows, x="实验", y="处理 FPS")
    st.download_button(
        "下载实验对比 CSV",
        data=comparison_path.read_bytes(),
        file_name="comparison.csv",
        mime="text/csv",
    )


def run_app() -> None:
    """Render the complete Streamlit Dashboard."""

    st.set_page_config(page_title="People Flow Analytics", page_icon="🚶", layout="wide")
    st.title(_APP_TITLE)
    st.caption("People Flow Analytics with YOLO26 and Multi-Object Tracking")
    st.info("先配置视频与几何区域，再运行核心管线；所有指标均来自落盘的 CSV/JSON 文件。")
    root = _project_root()
    run_tab, results_tab, experiments_tab = st.tabs(("运行分析", "结果与下载", "实验对比"))
    with run_tab:
        _render_run_tab(root)
    with results_tab:
        _render_results_tab(root)
    with experiments_tab:
        _render_experiments_tab(root)


if __name__ == "__main__":
    run_app()
