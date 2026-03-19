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
from pyswmm.simulation import _sim_state_instance
from pyswmm.toolkitapi import NodeResults, SimulationTime

from itzi import SwmmInputParser


SECONDS_PER_DAY = 24 * 3600
SPLIT_TIME = timedelta(hours=1, minutes=40)
SWMM_J1_TOTAL_INFLOW_MAX_DIFF = 0.25
SWMM_C0_FLOW_MAX_DIFF = 0.25
SWMM_J1_INTEGRATED_INFLOW_MAX_DIFF = 2.0


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
    finally:
        _sim_state_instance.sim_is_instantiated = False


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


def _rewrite_allow_ponding(inp_text: str, enabled: bool) -> str:
    target = "ALLOW_PONDING        YES"
    replacement = f"ALLOW_PONDING        {'YES' if enabled else 'NO'}"
    assert target in inp_text, "Could not find ALLOW_PONDING option"
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
        for label, allow_ponding in (("ponding_on", True), ("ponding_off", False)):
            inp_path = Path(tmp_dir) / f"{label}.inp"
            inp_path.write_text(_rewrite_allow_ponding(base_text, allow_ponding))

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
def test_swmm_hotstart_roundtrip_guardrails(test_data_path: str) -> None:
    """Keep a standalone SWMM guardrail on the known resume-sensitive quantities.

    SWMM hotstart is not restart-exact for the EA8b network, especially around the
    ponded J1 state, so the exact-comparison test above remains xfailed. This test
    keeps a narrower regression guardrail on the resumed-vs-uninterrupted metrics
    that were useful during the investigation, without relying on large diagnostic
    artifacts.
    """
    inp_file = Path(test_data_path) / "EA_test_8" / "b" / "test8b_drainage_ponding.inp"
    split_seconds = SPLIT_TIME.total_seconds()

    reference = _run_uninterrupted(str(inp_file), split_seconds)
    resumed = _run_with_hotstart(str(inp_file), split_seconds)

    ref_times = np.asarray(reference["elapsed_seconds"])
    resumed_times = np.asarray(resumed["elapsed_seconds"])
    ref_node_ids = reference["node_ids"]
    resumed_node_ids = resumed["node_ids"]
    ref_link_ids = reference["link_ids"]
    resumed_link_ids = resumed["link_ids"]

    j1_total_inflow_max_diff = _aligned_max_abs_diff(
        ref_times,
        np.asarray(reference["node_total_inflow"])[:, ref_node_ids.index("J1")],
        resumed_times,
        np.asarray(resumed["node_total_inflow"])[:, resumed_node_ids.index("J1")],
    )
    c0_flow_max_diff = _aligned_max_abs_diff(
        ref_times,
        np.asarray(reference["link_flows"])[:, ref_link_ids.index("C0")],
        resumed_times,
        np.asarray(resumed["link_flows"])[:, resumed_link_ids.index("C0")],
    )
    j1_integrated_inflow_diff = abs(
        _integrated_volume(
            resumed_times,
            np.asarray(resumed["node_total_inflow"])[:, resumed_node_ids.index("J1")],
        )
        - _integrated_volume(
            ref_times,
            np.asarray(reference["node_total_inflow"])[:, ref_node_ids.index("J1")],
        )
    )

    assert j1_total_inflow_max_diff < SWMM_J1_TOTAL_INFLOW_MAX_DIFF
    assert c0_flow_max_diff < SWMM_C0_FLOW_MAX_DIFF
    assert j1_integrated_inflow_diff < SWMM_J1_INTEGRATED_INFLOW_MAX_DIFF


@pytest.mark.slow
@pytest.mark.forked  # Avoid pyswmm.errors.MultiSimulationError.
def test_swmm_hotstart_roundtrip_ponding_ab(
    test_data_path: str,
    test_data_temp_path: str,
) -> None:
    """Verify that disabling global ponding reduces standalone SWMM resume divergence."""
    original_inp = Path(test_data_path) / "EA_test_8" / "b" / "test8b_drainage_ponding.inp"
    results = _run_ab_ponding_diagnostic(original_inp, test_data_temp_path)

    assert (
        results["ponding_off"]["j1_volume_max_diff"] < results["ponding_on"]["j1_volume_max_diff"]
    )
    assert (
        results["ponding_off"]["j1_overflow_max_diff"]
        < results["ponding_on"]["j1_overflow_max_diff"]
    )
    assert results["ponding_off"]["j0_depth_max_diff"] < results["ponding_on"]["j0_depth_max_diff"]
    assert results["ponding_off"]["c0_flow_max_diff"] < results["ponding_on"]["c0_flow_max_diff"]
