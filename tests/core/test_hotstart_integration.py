"""Integration tests for hotstart round-trip and scheduler state restoration.

This file implements phases 4 and 7 from the hotstart testing plan:
- Phase 4: Round-Trip Resume Tests
- Phase 7: Scheduler/Runtime State Tests
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest

from itzi.simulation_builder import SimulationBuilder
from itzi.data_containers import SimulationConfig, SurfaceFlowParameters
from itzi.providers.memory_output import MemoryRasterOutputProvider, MemoryVectorOutputProvider
from itzi.const import InfiltrationModelType, TemporalType

if TYPE_CHECKING:
    from itzi.simulation import Simulation


# Total simulation duration for all tests
TOTAL_DURATION_SECONDS = 60


def run_simulation_to_end(simulation: Simulation, skip_initialize: bool = False) -> None:
    """Run a simulation from its current state to end_time.

    Args:
        simulation: The simulation to run
        skip_initialize: If True, skip initialize() (for hotstarted simulations)
    """
    if not skip_initialize:
        simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()


def create_sim_config(
    start_time: datetime,
    end_time: datetime,
    helpers,
) -> SimulationConfig:
    """Create a SimulationConfig with rainfall and Green-Ampt infiltration."""
    return SimulationConfig(
        start_time=start_time,
        end_time=end_time,
        record_step=timedelta(seconds=30),
        temporal_type=TemporalType.RELATIVE,
        input_map_names=helpers.make_input_map_names(
            dem="z",
            friction="n",
            water_depth="start_h",
            rain="rain",
        ),
        output_map_names=helpers.make_output_map_names(
            "out_hotstart",
            ["water_depth", "qx", "qy", "volume_error"],
        ),
        surface_flow_parameters=SurfaceFlowParameters(hmin=0.0001, dtmax=0.3, cfl=0.2),
        infiltration_model=InfiltrationModelType.GREEN_AMPT,
    )


def build_simulation(
    sim_config: SimulationConfig,
    domain_5by5,
    hotstart_bytes: bytes | None = None,
    raster_output_provider=None,
) -> Simulation:
    """Build a simulation with optional hotstart.

    Args:
        sim_config: Simulation configuration
        domain_5by5: Domain fixture
        hotstart_bytes: Optional hotstart archive bytes
    """
    raster_output = raster_output_provider or MemoryRasterOutputProvider(
        {"out_map_names": sim_config.output_map_names}
    )

    builder = (
        SimulationBuilder(sim_config, domain_5by5.arr_mask, np.float32)
        .with_domain_data(domain_5by5.domain_data)
        .with_raster_output_provider(raster_output)
        .with_vector_output_provider(MemoryVectorOutputProvider({}))
    )

    if hotstart_bytes is not None:
        builder.with_hotstart(hotstart_bytes)

    simulation = builder.build()

    # Only set input arrays if NOT restoring from hotstart
    # Hotstart restores raster state, so we don't want to overwrite it
    if hotstart_bytes is None:
        # Set input arrays
        simulation.set_array("dem", domain_5by5.arr_dem_flat)
        simulation.set_array("friction", domain_5by5.arr_n)
        simulation.set_array("water_depth", domain_5by5.arr_start_h)
        simulation.set_array("rain", domain_5by5.arr_rain)

        # Set Green-Ampt infiltration parameters
        simulation.set_array("hydraulic_conductivity", domain_5by5.arr_inf)
        simulation.set_array(
            "capillary_pressure", np.full(domain_5by5.domain_data.shape, 100.0, dtype=np.float32)
        )
        simulation.set_array(
            "effective_porosity", np.full(domain_5by5.domain_data.shape, 0.4, dtype=np.float32)
        )
        simulation.set_array(
            "soil_water_content", np.full(domain_5by5.domain_data.shape, 0.3, dtype=np.float32)
        )

    return simulation


def capture_simulation_snapshot(simulation: Simulation) -> dict:
    return {
        "sim_time": simulation.sim_time,
        "dt": simulation.dt,
        "time_steps_counters": dict(simulation.time_steps_counters),
        "old_domain_volume": simulation.old_domain_volume,
        "next_ts": {key: value for key, value in simulation.next_ts.items()},
        "accum_update_time": {key: value for key, value in simulation.accum_update_time.items()},
        "raster_state": {
            key: simulation.raster_domain.get_array(key).copy()
            for key in simulation.raster_domain.k_all
        },
    }


def run_with_hotstart_checkpoints(
    simulation: Simulation,
    checkpoints: list[tuple[str, datetime]],
) -> dict[str, dict]:
    pending_checkpoints = sorted(checkpoints, key=lambda item: item[1])
    captured: dict[str, dict] = {}

    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
        while pending_checkpoints and simulation.sim_time >= pending_checkpoints[0][1]:
            name, _ = pending_checkpoints.pop(0)
            captured[name] = {
                "snapshot": capture_simulation_snapshot(simulation),
                "hotstart_bytes": simulation.create_hotstart().getvalue(),
            }
    simulation.finalize()
    return captured


def assert_final_state_matches(simulation: Simulation, reference: Simulation) -> None:
    for key in ["water_depth", "qe", "qs"]:
        arr_resumed = simulation.raster_domain.get_array(key)
        arr_reference = reference.raster_domain.get_array(key)
        np.testing.assert_allclose(arr_resumed, arr_reference, err_msg=f"Final {key} mismatch")


def assert_state_differs(simulation: Simulation, reference: Simulation) -> None:
    mismatch_found = False
    for key in ["water_depth", "qe", "qs"]:
        arr_resumed = simulation.raster_domain.get_array(key)
        arr_reference = reference.raster_domain.get_array(key)
        if not np.allclose(arr_resumed, arr_reference):
            mismatch_found = True
            break
    assert mismatch_found, "Expected resumed state to differ from the archived-cadence reference"


def get_unique_record_times(
    output_provider: MemoryRasterOutputProvider,
    output_key: str = "water_depth",
) -> list[datetime | timedelta]:
    unique_times: list[datetime | timedelta] = []
    for sim_time, _ in output_provider.output_maps_dict[output_key]:
        if not unique_times or sim_time != unique_times[-1]:
            unique_times.append(sim_time)
    return unique_times


@pytest.fixture(scope="module")
def base_time() -> datetime:
    """Base time for all simulations."""
    return datetime(2000, 1, 1, 0, 0, 0)


@pytest.fixture(scope="module")
def uninterrupted_simulation(baseline_hotstart_run) -> Simulation:
    """Reuse the shared 60-second baseline run as the final-state reference."""
    return baseline_hotstart_run["simulation"]


@pytest.fixture(scope="module")
def baseline_hotstart_run(domain_5by5, helpers, base_time) -> dict:
    end_time = base_time + timedelta(seconds=TOTAL_DURATION_SECONDS)
    sim_config = create_sim_config(
        start_time=base_time,
        end_time=end_time,
        helpers=helpers,
    )
    simulation = build_simulation(sim_config, domain_5by5)
    checkpoints = run_with_hotstart_checkpoints(
        simulation,
        [
            ("split_10", base_time + timedelta(seconds=10)),
            ("split_30", base_time + timedelta(seconds=30)),
            ("split_50", base_time + timedelta(seconds=50)),
        ],
    )
    return {
        "sim_config": sim_config,
        "simulation": simulation,
        "checkpoints": checkpoints,
    }


@pytest.fixture(scope="module")
def extended_reference_simulation(domain_5by5, helpers, base_time) -> dict:
    sim_config = create_sim_config(
        start_time=base_time,
        end_time=base_time + timedelta(seconds=90),
        helpers=helpers,
    )
    simulation = build_simulation(sim_config, domain_5by5)
    run_simulation_to_end(simulation)
    return {"sim_config": sim_config, "simulation": simulation}


@pytest.fixture(scope="module")
def record_step_hotstart_run(domain_5by5, helpers, base_time) -> dict:
    end_time = base_time + timedelta(seconds=70)
    sim_config = create_sim_config(
        start_time=base_time,
        end_time=end_time,
        helpers=helpers,
    )
    simulation = build_simulation(sim_config, domain_5by5)
    checkpoints = run_with_hotstart_checkpoints(
        simulation,
        [("split_34_2", base_time + timedelta(seconds=34.2))],
    )
    return {
        "sim_config": sim_config,
        "simulation": simulation,
        "checkpoints": checkpoints,
    }


# @pytest.mark.skip(reason="Hotstart final state comparison fails due to final results diverging "
#     "by ~0.05% which exceeds the default tolerance, possibly due to surfaceFlow dt being calculated from restored arrays.")
@pytest.mark.parametrize("split_seconds", [10, 30, 50])
def test_roundtrip_state_restoration_and_match(
    domain_5by5,
    helpers,
    base_time,
    baseline_hotstart_run,
    uninterrupted_simulation: Simulation,
    split_seconds: int,
) -> None:
    """Test state restoration and final match for different split points.

    This test:
    1. Runs simulation to split point and creates hotstart
    2. Verifies all states are properly restored (rasters, scheduler, continuity)
    3. Runs resumed simulation to end
    4. Verifies final results match the uninterrupted reference
    """
    end_time = base_time + timedelta(seconds=TOTAL_DURATION_SECONDS)

    checkpoint = baseline_hotstart_run["checkpoints"][f"split_{split_seconds}"]
    saved_snapshot = checkpoint["snapshot"]
    hotstart_bytes = checkpoint["hotstart_bytes"]

    # Step 2: Create simulation B with hotstart and verify state restoration
    sim_b_config = create_sim_config(
        start_time=base_time,
        end_time=end_time,
        helpers=helpers,
    )
    sim_b = build_simulation(sim_b_config, domain_5by5, hotstart_bytes=hotstart_bytes)

    # Verify raster state was restored
    for key, arr_saved in saved_snapshot["raster_state"].items():
        arr_restored = sim_b.raster_domain.get_array(key)
        np.testing.assert_allclose(
            arr_restored,
            arr_saved,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"Raster state {key} not restored correctly at split_time={split_seconds}s",
        )

    # Verify scheduler state was restored
    assert sim_b.sim_time == saved_snapshot["sim_time"], (
        f"sim_time not restored: {sim_b.sim_time} != {saved_snapshot['sim_time']}"
    )
    assert sim_b.dt == saved_snapshot["dt"], (
        f"dt not restored: {sim_b.dt} != {saved_snapshot['dt']}"
    )
    assert sim_b.time_steps_counters == saved_snapshot["time_steps_counters"], (
        "time_steps_counters not restored: "
        f"{sim_b.time_steps_counters} != {saved_snapshot['time_steps_counters']}"
    )
    assert np.isclose(sim_b.old_domain_volume, saved_snapshot["old_domain_volume"]), (
        "old_domain_volume not restored: "
        f"{sim_b.old_domain_volume} != {saved_snapshot['old_domain_volume']}"
    )

    # Verify next_ts schedule was restored
    for key, value in saved_snapshot["next_ts"].items():
        assert key in sim_b.next_ts, f"next_ts key '{key}' missing in restored simulation"
        assert sim_b.next_ts[key] == value, (
            f"next_ts[{key}] not restored: {sim_b.next_ts[key]} != {value}"
        )

    # Verify accum_update_time was restored
    for key, value in saved_snapshot["accum_update_time"].items():
        assert key in sim_b.accum_update_time, f"accum_update_time key '{key}' missing"
        assert sim_b.accum_update_time[key] == value, (
            f"accum_update_time[{key}] not restored: {sim_b.accum_update_time[key]} != {value}"
        )

    # Verify nextstep is consistent with next_ts (scheduler invariant)
    expected_nextstep = min(sim_b.next_ts.values())
    assert sim_b.nextstep == expected_nextstep, (
        f"nextstep not consistent with next_ts: {sim_b.nextstep} != {expected_nextstep}"
    )

    # Step 3: Run resumed simulation to end
    # Skip initialize() because hotstart already restored all necessary state
    # Calling initialize() would overwrite restored state (old_domain_volume, accum arrays, etc.)
    run_simulation_to_end(sim_b, skip_initialize=True)

    # Step 4: Verify final results match uninterrupted reference
    # Use qe/qs (internal flow arrays) instead of qx/qy (output arrays computed on-the-fly)
    assert_final_state_matches(sim_b, uninterrupted_simulation)

    # Verify simulation reached end time
    assert sim_b.sim_time == end_time, (
        f"Resumed simulation did not reach end time: {sim_b.sim_time} != {end_time}"
    )


def test_resume_allows_output_provider_change(
    domain_5by5,
    baseline_hotstart_run,
    uninterrupted_simulation: Simulation,
) -> None:
    sim_config = baseline_hotstart_run["sim_config"]
    hotstart_bytes = baseline_hotstart_run["checkpoints"]["split_30"]["hotstart_bytes"]

    resumed_output = MemoryRasterOutputProvider({"out_map_names": sim_config.output_map_names})
    sim_b = build_simulation(
        sim_config,
        domain_5by5,
        hotstart_bytes=hotstart_bytes,
        raster_output_provider=resumed_output,
    )

    run_simulation_to_end(sim_b, skip_initialize=True)

    assert resumed_output.output_maps_dict["water_depth"]
    assert len(resumed_output.output_maps_dict["water_depth"]) >= 1
    assert_final_state_matches(sim_b, uninterrupted_simulation)
    assert sim_b.sim_time == sim_config.end_time


def test_resume_allows_output_map_name_change(
    domain_5by5,
    helpers,
    baseline_hotstart_run,
    uninterrupted_simulation: Simulation,
) -> None:
    sim_a_config = baseline_hotstart_run["sim_config"]
    hotstart_bytes = baseline_hotstart_run["checkpoints"]["split_30"]["hotstart_bytes"]

    resumed_output_map_names = helpers.make_output_map_names(
        "out_resume",
        ["water_depth", "qx", "qy", "volume_error"],
    )
    sim_b_config = sim_a_config.model_copy(update={"output_map_names": resumed_output_map_names})
    resumed_output = MemoryRasterOutputProvider({"out_map_names": resumed_output_map_names})
    sim_b = build_simulation(
        sim_b_config,
        domain_5by5,
        hotstart_bytes=hotstart_bytes,
        raster_output_provider=resumed_output,
    )

    run_simulation_to_end(sim_b, skip_initialize=True)

    assert sim_b.report.out_map_names == resumed_output_map_names
    assert resumed_output.out_map_names == resumed_output_map_names
    assert resumed_output.output_maps_dict["water_depth"]
    assert sim_a_config.output_map_names["water_depth"] != resumed_output_map_names["water_depth"]
    assert_final_state_matches(sim_b, uninterrupted_simulation)


def test_resume_allows_end_time_extension(
    domain_5by5,
    base_time,
    baseline_hotstart_run,
    extended_reference_simulation,
) -> None:
    extended_end_time = base_time + timedelta(seconds=90)

    hotstart_bytes = baseline_hotstart_run["checkpoints"]["split_30"]["hotstart_bytes"]
    sim_b_config = extended_reference_simulation["sim_config"]
    sim_b = build_simulation(sim_b_config, domain_5by5, hotstart_bytes=hotstart_bytes)

    run_simulation_to_end(sim_b, skip_initialize=True)

    assert sim_b.sim_time == extended_end_time
    assert_final_state_matches(sim_b, extended_reference_simulation["simulation"])


def test_resume_applies_new_record_step_cadence(
    domain_5by5,
    base_time,
    record_step_hotstart_run,
) -> None:
    end_time = base_time + timedelta(seconds=70)
    original_config = record_step_hotstart_run["sim_config"]
    checkpoint = record_step_hotstart_run["checkpoints"]["split_34_2"]
    hotstart_bytes = checkpoint["hotstart_bytes"]

    resumed_record_step = timedelta(seconds=10)
    resumed_config = original_config.model_copy(update={"record_step": resumed_record_step})
    resumed_output = MemoryRasterOutputProvider({"out_map_names": resumed_config.output_map_names})
    sim_b = build_simulation(
        resumed_config,
        domain_5by5,
        hotstart_bytes=hotstart_bytes,
        raster_output_provider=resumed_output,
    )

    run_simulation_to_end(sim_b, skip_initialize=True)

    resumed_record_times = get_unique_record_times(resumed_output)
    resumed_offset = checkpoint["snapshot"]["sim_time"] - base_time
    expected_record_times = [
        resumed_offset + resumed_record_step,
        resumed_offset + (2 * resumed_record_step),
        resumed_offset + (3 * resumed_record_step),
    ]
    expected_record_times = [
        time for time in expected_record_times if time < (end_time - base_time)
    ]

    assert resumed_record_times[: len(expected_record_times)] == expected_record_times

    assert_state_differs(sim_b, record_step_hotstart_run["simulation"])


def test_resume_applies_new_dtinf_to_hydrology_schedule(
    domain_5by5,
    base_time,
    baseline_hotstart_run,
) -> None:
    hotstart_bytes = baseline_hotstart_run["checkpoints"]["split_30"]["hotstart_bytes"]
    original_config = baseline_hotstart_run["sim_config"]
    resumed_dtinf = 5.0
    resumed_config = original_config.model_copy(update={"dtinf": resumed_dtinf})

    sim_b = build_simulation(resumed_config, domain_5by5, hotstart_bytes=hotstart_bytes)

    assert sim_b.hydrology_model.dt == timedelta(seconds=resumed_dtinf)
    assert sim_b.next_ts["hydrology"] == sim_b.sim_time + timedelta(seconds=resumed_dtinf)
