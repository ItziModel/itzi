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
def sim_5by5_wse(domain_5by5, helpers) -> Simulation:
    """Run a 5x5 simulation for 60s with 30s record step using WSE input.

    Uses high DEM (z=132) and initial water surface elevation (132.2 at center).
    Outputs: water_depth, water_surface_elevation
    """
    # Build SimulationConfig
    sim_config = SimulationConfig(
        start_time=datetime(2000, 1, 1, 0, 0, 0),
        end_time=datetime(2000, 1, 1, 0, 1, 0),  # 60 seconds
        record_step=timedelta(seconds=30),
        temporal_type=TemporalType.RELATIVE,
        input_map_names=helpers.make_input_map_names(
            dem="z_high",
            friction="n",
            water_surface_elevation="start_wse",  # This activates WSE input
        ),
        output_map_names=helpers.make_output_map_names(
            "out_5by5_wse",
            ["water_depth", "water_surface_elevation"],
        ),
        # Same values as 5by5_wse.ini
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

    # Set input arrays - use HIGH DEM and WSE
    simulation.set_array("dem", domain_5by5.arr_dem_high)
    simulation.set_array("friction", domain_5by5.arr_n)
    simulation.set_array("water_surface_elevation", domain_5by5.arr_start_wse)

    # Run simulation
    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()

    return simulation


class TestWSE:
    """Test that water surface elevation is correctly computed from high DEM + initial depth.

    The simulation starts with:
    - DEM at z=132m everywhere
    - WSE at 132.2m at center cell [2,2] (implying 0.2m depth)

    After simulation, we verify:
    - h_max ~ 0.2 (max water depth)
    - wse_max ~ 132.2 (max water surface elevation)
    - wse_min ~ 132 (min WSE, equal to DEM where no water)
    """

    def test_water_depth_max(self, sim_5by5_wse):
        """Maximum water depth should be approximately 0.2m."""
        output_dict = sim_5by5_wse.report.raster_provider.output_maps_dict

        # Get all water_depth arrays and compute max
        h_arrays = [arr for _, arr in output_dict["water_depth"]]
        h_max = np.max([np.nanmax(arr) for arr in h_arrays])

        assert np.isclose(h_max, 0.2, atol=0.00001), (
            f"Max water depth should be ~0.2m, got {h_max:.6f}"
        )

    def test_wse_max(self, sim_5by5_wse):
        """Maximum water surface elevation should be approximately 132.2m."""
        output_dict = sim_5by5_wse.report.raster_provider.output_maps_dict

        # Get all WSE arrays and compute max
        wse_arrays = [arr for _, arr in output_dict["water_surface_elevation"]]
        wse_max = np.max([np.nanmax(arr) for arr in wse_arrays])

        assert np.isclose(wse_max, 132.2, atol=0.00001), (
            f"Max WSE should be ~132.2m, got {wse_max:.6f}"
        )

    def test_wse_min(self, sim_5by5_wse):
        """Minimum water surface elevation should be approximately 132m (DEM level)."""
        output_dict = sim_5by5_wse.report.raster_provider.output_maps_dict

        # Get all WSE arrays and compute min
        wse_arrays = [arr for _, arr in output_dict["water_surface_elevation"]]
        wse_min = np.min([np.nanmin(arr) for arr in wse_arrays])

        assert np.isclose(wse_min, 132.0, atol=0.00001), (
            f"Min WSE should be ~132m, got {wse_min:.6f}"
        )
