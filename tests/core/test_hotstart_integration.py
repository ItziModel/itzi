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


def run_to_split_and_create_hotstart(
    sim_config: SimulationConfig,
    domain_5by5,
    split_time: datetime,
    raster_output_provider=None,
) -> tuple[Simulation, bytes]:
    simulation = build_simulation(
        sim_config,
        domain_5by5,
        raster_output_provider=raster_output_provider,
    )
    simulation.initialize()
    while simulation.sim_time < split_time:
        simulation.update()
    return simulation, simulation.create_hotstart().getvalue()


def assert_final_state_matches(simulation: Simulation, reference: Simulation) -> None:
    for key in ["water_depth", "qe", "qs"]:
        arr_resumed = simulation.raster_domain.get_array(key)
        arr_reference = reference.raster_domain.get_array(key)
        np.testing.assert_allclose(arr_resumed, arr_reference, err_msg=f"Final {key} mismatch")


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
def uninterrupted_simulation(domain_5by5, helpers, base_time) -> Simulation:
    """Run the full simulation once for reference.

    This simulation runs from t=0 to t=60s without interruption.
    """
    end_time = base_time + timedelta(seconds=TOTAL_DURATION_SECONDS)
    sim_config = create_sim_config(
        start_time=base_time,
        end_time=end_time,
        helpers=helpers,
    )
    simulation = build_simulation(sim_config, domain_5by5)
    run_simulation_to_end(simulation)
    return simulation


# @pytest.mark.skip(reason="Hotstart final state comparison fails due to final results diverging "
#     "by ~0.05% which exceeds the default tolerance, possibly due to surfaceFlow dt being calculated from restored arrays.")
@pytest.mark.parametrize("split_seconds", [10, 30, 50])
def test_roundtrip_state_restoration_and_match(
    domain_5by5,
    helpers,
    base_time,
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
    split_time = base_time + timedelta(seconds=split_seconds)
    end_time = base_time + timedelta(seconds=TOTAL_DURATION_SECONDS)

    # Step 1: Run simulation A to split point and create hotstart
    sim_a_config = create_sim_config(
        start_time=base_time,
        end_time=end_time,
        helpers=helpers,
    )
    sim_a, hotstart_bytes = run_to_split_and_create_hotstart(
        sim_a_config,
        domain_5by5,
        split_time,
    )

    # Save state for comparison
    saved_sim_time = sim_a.sim_time
    saved_dt = sim_a.dt
    saved_counters = dict(sim_a.time_steps_counters)
    saved_old_domain_volume = sim_a.old_domain_volume
    saved_next_ts = {k: v for k, v in sim_a.next_ts.items()}
    saved_accum_update_time = {k: v for k, v in sim_a.accum_update_time.items()}

    # Save raster state for comparison
    saved_raster_state = {}
    for key in sim_a.raster_domain.k_all:
        saved_raster_state[key] = sim_a.raster_domain.get_array(key).copy()

    # Step 2: Create simulation B with hotstart and verify state restoration
    sim_b_config = create_sim_config(
        start_time=base_time,
        end_time=end_time,
        helpers=helpers,
    )
    sim_b = build_simulation(sim_b_config, domain_5by5, hotstart_bytes=hotstart_bytes)

    # Verify raster state was restored
    for key in saved_raster_state:
        arr_restored = sim_b.raster_domain.get_array(key)
        arr_saved = saved_raster_state[key]
        np.testing.assert_allclose(
            arr_restored,
            arr_saved,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"Raster state {key} not restored correctly at split_time={split_seconds}s",
        )

    # Verify scheduler state was restored
    assert sim_b.sim_time == saved_sim_time, (
        f"sim_time not restored: {sim_b.sim_time} != {saved_sim_time}"
    )
    assert sim_b.dt == saved_dt, f"dt not restored: {sim_b.dt} != {saved_dt}"
    assert sim_b.time_steps_counters == saved_counters, (
        f"time_steps_counters not restored: {sim_b.time_steps_counters} != {saved_counters}"
    )
    assert np.isclose(sim_b.old_domain_volume, saved_old_domain_volume), (
        f"old_domain_volume not restored: {sim_b.old_domain_volume} != {saved_old_domain_volume}"
    )

    # Verify next_ts schedule was restored
    for key in saved_next_ts:
        assert key in sim_b.next_ts, f"next_ts key '{key}' missing in restored simulation"
        assert sim_b.next_ts[key] == saved_next_ts[key], (
            f"next_ts[{key}] not restored: {sim_b.next_ts[key]} != {saved_next_ts[key]}"
        )

    # Verify accum_update_time was restored
    for key in saved_accum_update_time:
        assert key in sim_b.accum_update_time, f"accum_update_time key '{key}' missing"
        assert sim_b.accum_update_time[key] == saved_accum_update_time[key], (
            f"accum_update_time[{key}] not restored: "
            f"{sim_b.accum_update_time[key]} != {saved_accum_update_time[key]}"
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
    helpers,
    base_time,
    uninterrupted_simulation: Simulation,
) -> None:
    split_time = base_time + timedelta(seconds=30)
    end_time = base_time + timedelta(seconds=TOTAL_DURATION_SECONDS)

    sim_config = create_sim_config(
        start_time=base_time,
        end_time=end_time,
        helpers=helpers,
    )
    initial_output = MemoryRasterOutputProvider({"out_map_names": sim_config.output_map_names})
    _, hotstart_bytes = run_to_split_and_create_hotstart(
        sim_config,
        domain_5by5,
        split_time,
        raster_output_provider=initial_output,
    )

    resumed_output = MemoryRasterOutputProvider({"out_map_names": sim_config.output_map_names})
    sim_b = build_simulation(
        sim_config,
        domain_5by5,
        hotstart_bytes=hotstart_bytes,
        raster_output_provider=resumed_output,
    )

    run_simulation_to_end(sim_b, skip_initialize=True)

    assert initial_output is not resumed_output
    assert resumed_output.output_maps_dict["water_depth"]
    assert len(resumed_output.output_maps_dict["water_depth"]) >= 1
    assert_final_state_matches(sim_b, uninterrupted_simulation)
    assert sim_b.sim_time == end_time


def test_resume_allows_output_map_name_change(
    domain_5by5,
    helpers,
    base_time,
    uninterrupted_simulation: Simulation,
) -> None:
    split_time = base_time + timedelta(seconds=30)
    end_time = base_time + timedelta(seconds=TOTAL_DURATION_SECONDS)

    sim_a_config = create_sim_config(
        start_time=base_time,
        end_time=end_time,
        helpers=helpers,
    )
    _, hotstart_bytes = run_to_split_and_create_hotstart(sim_a_config, domain_5by5, split_time)

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


def test_resume_allows_end_time_extension(domain_5by5, helpers, base_time) -> None:
    split_time = base_time + timedelta(seconds=30)
    original_end_time = base_time + timedelta(seconds=TOTAL_DURATION_SECONDS)
    extended_end_time = base_time + timedelta(seconds=90)

    sim_a_config = create_sim_config(
        start_time=base_time,
        end_time=original_end_time,
        helpers=helpers,
    )
    _, hotstart_bytes = run_to_split_and_create_hotstart(sim_a_config, domain_5by5, split_time)

    sim_b_config = create_sim_config(
        start_time=base_time,
        end_time=extended_end_time,
        helpers=helpers,
    )
    sim_b = build_simulation(sim_b_config, domain_5by5, hotstart_bytes=hotstart_bytes)

    reference = build_simulation(sim_b_config, domain_5by5)
    run_simulation_to_end(reference)
    run_simulation_to_end(sim_b, skip_initialize=True)

    assert sim_b.sim_time == extended_end_time
    assert_final_state_matches(sim_b, reference)


def test_resume_applies_new_record_step_cadence(domain_5by5, helpers, base_time) -> None:
    split_time = base_time + timedelta(seconds=34.2)
    end_time = base_time + timedelta(seconds=70)
    original_config = create_sim_config(
        start_time=base_time,
        end_time=end_time,
        helpers=helpers,
    )
    sim_a, hotstart_bytes = run_to_split_and_create_hotstart(
        original_config,
        domain_5by5,
        split_time,
    )

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
    resumed_offset = sim_a.sim_time - base_time
    expected_record_times = [
        resumed_offset + resumed_record_step,
        resumed_offset + (2 * resumed_record_step),
        resumed_offset + (3 * resumed_record_step),
    ]
    expected_record_times = [
        time for time in expected_record_times if time < (end_time - base_time)
    ]

    assert resumed_record_times[: len(expected_record_times)] == expected_record_times

    reference = build_simulation(original_config, domain_5by5)
    run_simulation_to_end(reference)
    assert_final_state_matches(sim_b, reference)
