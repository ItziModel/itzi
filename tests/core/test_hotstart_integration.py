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
) -> Simulation:
    """Build a simulation with optional hotstart.

    Args:
        sim_config: Simulation configuration
        domain_5by5: Domain fixture
        hotstart_bytes: Optional hotstart archive bytes
    """
    raster_output = MemoryRasterOutputProvider({"out_map_names": sim_config.output_map_names})

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
    sim_a = build_simulation(sim_a_config, domain_5by5)
    sim_a.initialize()

    # Run until split time
    while sim_a.sim_time < split_time:
        sim_a.update()

    # Create hotstart at split point
    hotstart_bytes = sim_a.create_hotstart()

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
    sim_b = build_simulation(sim_b_config, domain_5by5, hotstart_bytes=hotstart_bytes.getvalue())

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
    for key in ["water_depth", "qe", "qs"]:
        arr_resumed = sim_b.raster_domain.get_array(key)
        arr_uninterrupted = uninterrupted_simulation.raster_domain.get_array(key)
        np.testing.assert_allclose(
            arr_resumed,
            arr_uninterrupted,
            err_msg=f"Final {key} mismatch for split_time={split_seconds}s",
        )

    # Verify simulation reached end time
    assert sim_b.sim_time == end_time, (
        f"Resumed simulation did not reach end time: {sim_b.sim_time} != {end_time}"
    )
