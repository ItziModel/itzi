"""Standalone SWMM hotstart roundtrip test for the EA8b network."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import os
import tempfile
from typing import Any

import numpy as np
import pytest
import pyswmm
from pyswmm.toolkitapi import NodeResults, SimulationTime

from itzi import SwmmInputParser


SECONDS_PER_DAY = 24 * 3600
SPLIT_TIME = timedelta(hours=1, minutes=40)


def _record_snapshot(
    snapshots: dict[str, list[float] | list[list[float]]],
    node_objects: list,
    link_objects: list,
    swmm_model: Any,
    current_time: datetime,
    original_start: datetime,
) -> None:
    snapshots["elapsed_seconds"].append((current_time - original_start).total_seconds())
    snapshots["node_depths"].append([node.depth for node in node_objects])
    snapshots["node_heads"].append([node.head for node in node_objects])
    snapshots["node_total_inflow"].append([node.total_inflow for node in node_objects])
    snapshots["node_cumulative_inflow"].append([node.cumulative_inflow for node in node_objects])
    snapshots["node_volumes"].append([node.volume for node in node_objects])
    snapshots["node_overflow"].append(
        [swmm_model.getNodeResult(node.nodeid, NodeResults.overflow) for node in node_objects]
    )
    snapshots["link_flows"].append([link.flow for link in link_objects])
    snapshots["link_depths"].append([link.depth for link in link_objects])
    snapshots["link_volumes"].append([link.volume for link in link_objects])


def _empty_snapshots() -> dict[str, list[float] | list[list[float]]]:
    return {
        "elapsed_seconds": [],
        "node_depths": [],
        "node_heads": [],
        "node_total_inflow": [],
        "node_cumulative_inflow": [],
        "node_volumes": [],
        "node_overflow": [],
        "link_flows": [],
        "link_depths": [],
        "link_volumes": [],
    }


def _finalize_snapshots(
    snapshots: dict[str, list[float] | list[list[float]]],
    node_ids: tuple[str, ...],
    link_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...] | np.ndarray]:
    return {
        "node_ids": node_ids,
        "link_ids": link_ids,
        "elapsed_seconds": np.asarray(snapshots["elapsed_seconds"], dtype=np.float64),
        "node_depths": np.asarray(snapshots["node_depths"], dtype=np.float64),
        "node_heads": np.asarray(snapshots["node_heads"], dtype=np.float64),
        "node_total_inflow": np.asarray(snapshots["node_total_inflow"], dtype=np.float64),
        "node_cumulative_inflow": np.asarray(
            snapshots["node_cumulative_inflow"], dtype=np.float64
        ),
        "node_volumes": np.asarray(snapshots["node_volumes"], dtype=np.float64),
        "node_overflow": np.asarray(snapshots["node_overflow"], dtype=np.float64),
        "link_flows": np.asarray(snapshots["link_flows"], dtype=np.float64),
        "link_depths": np.asarray(snapshots["link_depths"], dtype=np.float64),
        "link_volumes": np.asarray(snapshots["link_volumes"], dtype=np.float64),
    }


def _close_swmm_simulation(swmm_sim: pyswmm.Simulation) -> None:
    try:
        swmm_sim.close()
    except Exception:
        model = swmm_sim._model
        for method_name in ("swmm_end", "swmm_report", "swmm_close"):
            try:
                getattr(model, method_name)()
            except Exception:
                pass


def _step_to_elapsed_seconds(model, target_seconds: float) -> float:
    elapsed_seconds = 0.0
    while elapsed_seconds < target_seconds:
        elapsed_days = model.swmm_step()
        assert elapsed_days > 0.0, "SWMM ended before reaching the requested split time"
        elapsed_seconds = elapsed_days * SECONDS_PER_DAY
    return elapsed_seconds


def _assert_reached_split_time(actual_seconds: float, target_seconds: float) -> None:
    assert actual_seconds >= target_seconds
    assert actual_seconds - target_seconds < 1.0


def _integrated_volume(elapsed_seconds: np.ndarray, flows: np.ndarray) -> float:
    return float(np.trapezoid(flows, elapsed_seconds))


def _format_focus_window(
    elapsed_reference: np.ndarray,
    elapsed_resumed: np.ndarray,
    reference_values: np.ndarray,
    resumed_values: np.ndarray,
    focus_index: int,
    label: str,
    window_size: int = 8,
) -> list[str]:
    lines = [f"{label} around max diff:"]
    start = max(0, focus_index - window_size)
    end = min(len(elapsed_reference), focus_index + window_size + 1)
    lines.append("idx  t_ref  t_res  resumed  reference  diff")
    for idx in range(start, end):
        resumed_value = resumed_values[idx]
        reference_value = reference_values[idx]
        lines.append(
            f"{idx:4d} {elapsed_reference[idx]:6.1f} {elapsed_resumed[idx]:6.1f} "
            f"{resumed_value:8.5f} {reference_value:10.5f} "
            f"{(resumed_value - reference_value):9.5f}"
        )
    return lines


def _build_diagnostic_report(
    reference: dict[str, tuple[str, ...] | np.ndarray],
    resumed: dict[str, tuple[str, ...] | np.ndarray],
) -> str:
    node_ids = reference["node_ids"]
    link_ids = reference["link_ids"]
    elapsed_reference = np.asarray(reference["elapsed_seconds"])
    elapsed_resumed = np.asarray(resumed["elapsed_seconds"])
    focus_nodes = ("J0", "J1")
    focus_links = ("C0",)

    metric_defs = {
        "node_depths": node_ids,
        "node_heads": node_ids,
        "node_total_inflow": node_ids,
        "node_cumulative_inflow": node_ids,
        "node_volumes": node_ids,
        "node_overflow": node_ids,
        "link_flows": link_ids,
        "link_depths": link_ids,
        "link_volumes": link_ids,
    }

    lines = [
        "Standalone SWMM hotstart roundtrip diagnostics",
        "",
        "Global maxima:",
    ]
    for metric_name, ids in metric_defs.items():
        reference_values = np.asarray(reference[metric_name])
        resumed_values = np.asarray(resumed[metric_name])
        diff = np.abs(resumed_values - reference_values)
        max_time_idx, max_entity_idx = np.unravel_index(np.argmax(diff), diff.shape)
        lines.append(
            "- "
            f"{metric_name}: max_diff={diff[max_time_idx, max_entity_idx]:.6f} "
            f"at t_ref={elapsed_reference[max_time_idx]:.1f}s "
            f"t_res={elapsed_resumed[max_time_idx]:.1f}s "
            f"id={ids[max_entity_idx]}"
        )

    lines.extend(["", "Focused nodes/links in first 60s after restart:"])
    early_window = (elapsed_reference >= SPLIT_TIME.total_seconds()) & (
        elapsed_reference <= SPLIT_TIME.total_seconds() + 60.0
    )
    for node_id in focus_nodes:
        node_idx = node_ids.index(node_id)
        for metric_name in (
            "node_depths",
            "node_total_inflow",
            "node_volumes",
            "node_overflow",
        ):
            reference_values = np.asarray(reference[metric_name])[early_window, node_idx]
            resumed_values = np.asarray(resumed[metric_name])[early_window, node_idx]
            diff = np.abs(resumed_values - reference_values)
            lines.append(
                f"- {node_id} {metric_name}: max_diff={diff.max():.6f} mean_diff={diff.mean():.6f}"
            )

    for link_id in focus_links:
        link_idx = link_ids.index(link_id)
        for metric_name in ("link_flows", "link_depths", "link_volumes"):
            reference_values = np.asarray(reference[metric_name])[early_window, link_idx]
            resumed_values = np.asarray(resumed[metric_name])[early_window, link_idx]
            diff = np.abs(resumed_values - reference_values)
            lines.append(
                f"- {link_id} {metric_name}: max_diff={diff.max():.6f} mean_diff={diff.mean():.6f}"
            )

    lines.extend(["", "Integrated flow volume after restart:"])
    c0_idx = link_ids.index("C0")
    j1_idx = node_ids.index("J1")
    c0_reference = np.asarray(reference["link_flows"])[:, c0_idx]
    c0_resumed = np.asarray(resumed["link_flows"])[:, c0_idx]
    j1_reference = np.asarray(reference["node_total_inflow"])[:, j1_idx]
    j1_resumed = np.asarray(resumed["node_total_inflow"])[:, j1_idx]
    lines.append(
        "- "
        f"C0 integrated flow volume: resumed={_integrated_volume(elapsed_resumed, c0_resumed):.3f} m3, "
        f"reference={_integrated_volume(elapsed_reference, c0_reference):.3f} m3, "
        f"diff={_integrated_volume(elapsed_resumed, c0_resumed) - _integrated_volume(elapsed_reference, c0_reference):.3f} m3"
    )
    lines.append(
        "- "
        f"J1 integrated total inflow: resumed={_integrated_volume(elapsed_resumed, j1_resumed):.3f} m3, "
        f"reference={_integrated_volume(elapsed_reference, j1_reference):.3f} m3, "
        f"diff={_integrated_volume(elapsed_resumed, j1_resumed) - _integrated_volume(elapsed_reference, j1_reference):.3f} m3"
    )

    lines.extend(["", "Detailed windows around largest discrepancies:"])
    detail_specs = (
        ("node_depths", "J0"),
        ("node_total_inflow", "J1"),
        ("node_volumes", "J0"),
        ("link_flows", "C0"),
    )
    for metric_name, entity_id in detail_specs:
        ids = node_ids if metric_name.startswith("node_") else link_ids
        entity_idx = ids.index(entity_id)
        reference_values = np.asarray(reference[metric_name])[:, entity_idx]
        resumed_values = np.asarray(resumed[metric_name])[:, entity_idx]
        max_time_idx = int(np.argmax(np.abs(resumed_values - reference_values)))
        lines.extend(
            _format_focus_window(
                elapsed_reference,
                elapsed_resumed,
                reference_values,
                resumed_values,
                max_time_idx,
                f"{entity_id} {metric_name}",
            )
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _rewrite_j1_ponded_area(inp_text: str, ponded_area: int) -> str:
    target = "J1               29.46        2.0          0       100        1"
    replacement = f"J1               29.46        2.0          0       100        {ponded_area}"
    assert target in inp_text, "Could not find J1 junction definition"
    return inp_text.replace(target, replacement, 1)


def _aligned_values(
    times_a: np.ndarray,
    values_a: np.ndarray,
    times_b: np.ndarray,
    values_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    common_times, idx_a, idx_b = np.intersect1d(
        times_a, times_b, assume_unique=False, return_indices=True
    )
    assert len(common_times) > 0, "No common elapsed times found for comparison"
    return values_a[idx_a], values_b[idx_b]


def _aligned_max_abs_diff(
    times_a: np.ndarray,
    values_a: np.ndarray,
    times_b: np.ndarray,
    values_b: np.ndarray,
) -> float:
    aligned_a, aligned_b = _aligned_values(times_a, values_a, times_b, values_b)
    return float(np.max(np.abs(aligned_a - aligned_b)))


def _run_ab_ponding_diagnostic(
    original_inp: Path,
    test_data_temp_path: str,
) -> dict[str, dict[str, float]]:
    base_text = original_inp.read_text()
    results: dict[str, dict[str, float]] = {}

    with tempfile.TemporaryDirectory(dir=test_data_temp_path) as tmp_dir:
        for label, ponded_area in (("ponding_on", 1), ("ponding_off", 0)):
            inp_path = Path(tmp_dir) / f"{label}.inp"
            inp_path.write_text(_rewrite_j1_ponded_area(base_text, ponded_area))

            reference = _run_uninterrupted(str(inp_path), SPLIT_TIME.total_seconds())
            resumed = _run_with_hotstart(str(inp_path), SPLIT_TIME.total_seconds())
            ref_times = np.asarray(reference["elapsed_seconds"])
            resumed_times = np.asarray(resumed["elapsed_seconds"])
            ref_node_ids = reference["node_ids"]
            resumed_node_ids = resumed["node_ids"]
            ref_link_ids = reference["link_ids"]
            resumed_link_ids = resumed["link_ids"]

            results[label] = {
                "j1_volume_max_diff": _aligned_max_abs_diff(
                    ref_times,
                    np.asarray(reference["node_volumes"])[:, ref_node_ids.index("J1")],
                    resumed_times,
                    np.asarray(resumed["node_volumes"])[:, resumed_node_ids.index("J1")],
                ),
                "j1_overflow_max_diff": _aligned_max_abs_diff(
                    ref_times,
                    np.asarray(reference["node_overflow"])[:, ref_node_ids.index("J1")],
                    resumed_times,
                    np.asarray(resumed["node_overflow"])[:, resumed_node_ids.index("J1")],
                ),
                "j1_total_inflow_max_diff": _aligned_max_abs_diff(
                    ref_times,
                    np.asarray(reference["node_total_inflow"])[:, ref_node_ids.index("J1")],
                    resumed_times,
                    np.asarray(resumed["node_total_inflow"])[:, resumed_node_ids.index("J1")],
                ),
                "j0_depth_max_diff": _aligned_max_abs_diff(
                    ref_times,
                    np.asarray(reference["node_depths"])[:, ref_node_ids.index("J0")],
                    resumed_times,
                    np.asarray(resumed["node_depths"])[:, resumed_node_ids.index("J0")],
                ),
                "c0_flow_max_diff": _aligned_max_abs_diff(
                    ref_times,
                    np.asarray(reference["link_flows"])[:, ref_link_ids.index("C0")],
                    resumed_times,
                    np.asarray(resumed["link_flows"])[:, resumed_link_ids.index("C0")],
                ),
                "c0_link_volume_max_diff": _aligned_max_abs_diff(
                    ref_times,
                    np.asarray(reference["link_volumes"])[:, ref_link_ids.index("C0")],
                    resumed_times,
                    np.asarray(resumed["link_volumes"])[:, resumed_link_ids.index("C0")],
                ),
            }

    return results


def _run_uninterrupted(
    inp_file: str, split_seconds: float
) -> dict[str, tuple[str, ...] | np.ndarray]:
    swmm_sim = pyswmm.Simulation(inp_file)
    swmm_model = swmm_sim._model
    node_objects = list(pyswmm.Nodes(swmm_sim))
    link_objects = list(pyswmm.Links(swmm_sim))
    node_ids = tuple(node.nodeid for node in node_objects)
    link_ids = tuple(link.linkid for link in link_objects)
    original_start = swmm_sim.start_time
    snapshots = _empty_snapshots()

    try:
        swmm_model.swmm_start()
        elapsed_seconds = _step_to_elapsed_seconds(swmm_model, split_seconds)
        _assert_reached_split_time(elapsed_seconds, split_seconds)

        _record_snapshot(
            snapshots,
            node_objects,
            link_objects,
            swmm_model,
            swmm_model.getCurrentSimulationTime(),
            original_start,
        )

        while True:
            elapsed_days = swmm_model.swmm_step()
            if elapsed_days <= 0.0:
                break
            _record_snapshot(
                snapshots,
                node_objects,
                link_objects,
                swmm_model,
                swmm_model.getCurrentSimulationTime(),
                original_start,
            )
    finally:
        _close_swmm_simulation(swmm_sim)

    return _finalize_snapshots(snapshots, node_ids, link_ids)


def _run_with_hotstart(
    inp_file: str, split_seconds: float
) -> dict[str, tuple[str, ...] | np.ndarray]:
    parser = SwmmInputParser(inp_file)
    original_start = parser.get_start_datetime()
    assert original_start is not None, "Failed to parse SWMM START_DATE/START_TIME"

    hotstart_file = tempfile.NamedTemporaryFile(suffix=".hsf", delete=False)
    hotstart_path = hotstart_file.name
    hotstart_file.close()

    phase_one_sim = pyswmm.Simulation(inp_file)
    phase_one_model = phase_one_sim._model
    try:
        phase_one_model.swmm_start()
        elapsed_seconds = _step_to_elapsed_seconds(phase_one_model, split_seconds)
        _assert_reached_split_time(elapsed_seconds, split_seconds)
        phase_one_model.swmm_save_hotstart(hotstart_path)
    finally:
        _close_swmm_simulation(phase_one_sim)

    resumed_sim = pyswmm.Simulation(inp_file)
    resumed_model = resumed_sim._model
    node_objects = list(pyswmm.Nodes(resumed_sim))
    link_objects = list(pyswmm.Links(resumed_sim))
    node_ids = tuple(node.nodeid for node in node_objects)
    link_ids = tuple(link.linkid for link in link_objects)
    snapshots = _empty_snapshots()

    try:
        resumed_model.swmm_use_hotstart(hotstart_path)
        resumed_model.setSimulationDateTime(
            SimulationTime.StartDateTime,
            original_start + timedelta(seconds=elapsed_seconds),
        )
        resumed_model.swmm_start()

        _record_snapshot(
            snapshots,
            node_objects,
            link_objects,
            resumed_model,
            resumed_model.getCurrentSimulationTime(),
            original_start,
        )

        while True:
            elapsed_days = resumed_model.swmm_step()
            if elapsed_days <= 0.0:
                break
            _record_snapshot(
                snapshots,
                node_objects,
                link_objects,
                resumed_model,
                resumed_model.getCurrentSimulationTime(),
                original_start,
            )
    finally:
        _close_swmm_simulation(resumed_sim)
        os.unlink(hotstart_path)

    return _finalize_snapshots(snapshots, node_ids, link_ids)


@pytest.mark.slow
@pytest.mark.forked  # Avoid pyswmm.errors.MultiSimulationError.
@pytest.mark.xfail(
    reason=(
        "EPA SWMM hotstart resume does not reproduce uninterrupted EA8b outputs exactly; "
        "this test documents the current standalone SWMM limitation"
    ),
)
def test_swmm_hotstart_roundtrip(test_data_path: str) -> None:
    """Verify SWMM-only hotstart resume matches uninterrupted EA8b output."""
    inp_file = Path(test_data_path) / "EA_test_8" / "b" / "test8b_drainage_ponding.inp"
    split_seconds = SPLIT_TIME.total_seconds()

    reference = _run_uninterrupted(str(inp_file), split_seconds)
    resumed = _run_with_hotstart(str(inp_file), split_seconds)

    assert resumed["node_ids"] == reference["node_ids"]
    assert resumed["link_ids"] == reference["link_ids"]

    np.testing.assert_allclose(
        resumed["elapsed_seconds"],
        reference["elapsed_seconds"],
        atol=1.0,
        rtol=0.0,
        err_msg="SWMM-only roundtrip mismatch in snapshot times",
    )

    for key in (
        "node_depths",
        "node_heads",
        "node_total_inflow",
        "link_flows",
        "link_depths",
    ):
        np.testing.assert_allclose(
            resumed[key],
            reference[key],
            err_msg=f"SWMM-only roundtrip mismatch for {key}",
        )


@pytest.mark.slow
@pytest.mark.forked  # Avoid pyswmm.errors.MultiSimulationError.
def test_swmm_hotstart_roundtrip_diagnostics(
    test_data_path: str,
    test_data_temp_path: str,
) -> None:
    """Write a focused SWMM-only diagnostic report for the EA8b hotstart split."""
    inp_file = Path(test_data_path) / "EA_test_8" / "b" / "test8b_drainage_ponding.inp"
    split_seconds = SPLIT_TIME.total_seconds()

    reference = _run_uninterrupted(str(inp_file), split_seconds)
    resumed = _run_with_hotstart(str(inp_file), split_seconds)

    report_path = Path(test_data_temp_path) / "swmm_hotstart_roundtrip_diagnostic.txt"
    report_path.write_text(_build_diagnostic_report(reference, resumed))

    assert report_path.exists()


@pytest.mark.slow
@pytest.mark.forked  # Avoid pyswmm.errors.MultiSimulationError.
def test_swmm_hotstart_roundtrip_ponding_ab(
    test_data_path: str,
    test_data_temp_path: str,
) -> None:
    """Compare standalone SWMM hotstart divergence with J1 ponding enabled vs disabled."""
    original_inp = Path(test_data_path) / "EA_test_8" / "b" / "test8b_drainage_ponding.inp"
    results = _run_ab_ponding_diagnostic(original_inp, test_data_temp_path)

    report_lines = [
        "SWMM hotstart ponding A/B diagnostic",
        "",
    ]
    for label in ("ponding_on", "ponding_off"):
        report_lines.append(f"{label}:")
        for key, value in results[label].items():
            report_lines.append(f"- {key}: {value:.6f}")
        report_lines.append("")

    comparison_path = Path(test_data_temp_path) / "swmm_hotstart_ponding_ab.txt"
    comparison_path.write_text("\n".join(report_lines).rstrip() + "\n")

    assert comparison_path.exists()
