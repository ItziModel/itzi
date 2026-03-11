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


@pytest.fixture(scope="module")
def sim_5by5(domain_5by5, helpers) -> Simulation:
    """Run a 5x5 simulation for 60s with 30s record step.

    Outputs: water_depth, water_surface_elevation, froude, v, vdir, qx, qy, volume_error
    """
    # Build SimulationConfig
    sim_config = SimulationConfig(
        start_time=datetime(2000, 1, 1, 0, 0, 0),
        end_time=datetime(2000, 1, 1, 0, 1, 0),  # 60 seconds
        record_step=timedelta(seconds=30),
        temporal_type=TemporalType.RELATIVE,
        input_map_names=helpers.make_input_map_names(
            dem="z",
            friction="n",
            water_depth="start_h",
        ),
        output_map_names=helpers.make_output_map_names(
            "out_5by5",
            [
                "water_depth",
                "water_surface_elevation",
                "froude",
                "v",
                "vdir",
                "qx",
                "qy",
                "volume_error",
            ],
        ),
        # Same values as original 5by5.ini
        surface_flow_parameters=SurfaceFlowParameters(hmin=0.0001, dtmax=0.3, cfl=0.2),
        infiltration_model=InfiltrationModelType.NULL,
    )

    # Create output provider
    raster_output = MemoryRasterOutputProvider({"out_map_names": sim_config.output_map_names})

    # Build simulation
    simulation = (
        SimulationBuilder(sim_config, domain_5by5.arr_mask, np.float32)
        .with_domain_data(domain_5by5.domain_data)
        .with_raster_output_provider(raster_output)
        .with_vector_output_provider(MemoryVectorOutputProvider({}))
        .build()
    )

    # Set input arrays
    simulation.set_array("dem", domain_5by5.arr_dem_flat)
    simulation.set_array("friction", domain_5by5.arr_n)
    simulation.set_array("water_depth", domain_5by5.arr_start_h)

    # Run simulation
    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()

    return simulation


class TestNumberOfOutput:
    """Test that the correct number of output maps are produced.
    60 seconds test with 30s record steps.
    Memory provider outputs: initial (t=0) + 2 record steps (t=30, t=60) = 3 outputs.
    Note: Unlike GRASS provider, memory provider does not generate separate max maps."""

    def test_water_depth_count(self, sim_5by5):
        """water_depth should have 3 outputs (initial + 2 record steps)."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["water_depth"]) == 3

    def test_water_surface_elevation_count(self, sim_5by5):
        """water_surface_elevation should have 3 outputs."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["water_surface_elevation"]) == 3

    def test_froude_count(self, sim_5by5):
        """froude should have 3 outputs."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["froude"]) == 3

    def test_v_count(self, sim_5by5):
        """v (velocity) should have 3 outputs (initial + 2 record steps)."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["v"]) == 3

    def test_vdir_count(self, sim_5by5):
        """vdir (velocity direction) should have 3 outputs."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["vdir"]) == 3

    def test_qx_count(self, sim_5by5):
        """qx should have 3 outputs."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["qx"]) == 3

    def test_qy_count(self, sim_5by5):
        """qy should have 3 outputs."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["qy"]) == 3

    def test_volume_error_count(self, sim_5by5):
        """volume_error should have 3 outputs."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict
        assert len(output_dict["volume_error"]) == 3


class TestFlowSymmetry:
    """Test that water depths at 4 symmetric control points around center are equal.

    On a 5x5 grid, the center is [2, 2]. The 4 symmetric neighbours are:
    - [1, 2] (above center)
    - [3, 2] (below center)
    - [2, 1] (left of center)
    - [2, 3] (right of center)

    After a dam-break, flow should spread symmetrically in all directions.
    """

    def test_flow_symmetry(self, sim_5by5):
        """Water depths at symmetric points should be equal after dam-break."""
        output_dict = sim_5by5.report.raster_provider.output_maps_dict

        # Get the last water_depth output (index -1 or 2)
        # The GRASS test uses water_depth_0002 which is the last time step
        _, h_array = output_dict["water_depth"][-1]

        # Sample at the 4 symmetric control points around center
        # Center is at [2, 2], neighbours are at [1,2], [3,2], [2,1], [2,3]
        h_above = h_array[1, 2]  # row 1, col 2
        h_below = h_array[3, 2]  # row 3, col 2
        h_left = h_array[2, 1]  # row 2, col 1
        h_right = h_array[2, 3]  # row 2, col 3

        # All 4 values should be approximately equal due to symmetry
        values = [h_above, h_below, h_left, h_right]
        assert np.allclose(values[:-1], values[1:]), (
            f"Symmetric points should have equal depths: "
            f"above={h_above:.6f}, below={h_below:.6f}, "
            f"left={h_left:.6f}, right={h_right:.6f}"
        )
