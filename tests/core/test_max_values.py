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
def sim_5by5_max_values(domain_5by5, helpers) -> Simulation:
    """Run a 5x5 simulation for 2s with 1s record step.

    Outputs: water_depth, v
    Used for testing that max values are correctly computed.
    """
    # Build SimulationConfig
    sim_config = SimulationConfig(
        start_time=datetime(2000, 1, 1, 0, 0, 0),
        end_time=datetime(2000, 1, 1, 0, 0, 2),  # 2 seconds
        record_step=timedelta(seconds=1),
        temporal_type=TemporalType.RELATIVE,
        input_map_names=helpers.make_input_map_names(
            dem="z",
            friction="n",
            water_depth="start_h",
        ),
        output_map_names=helpers.make_output_map_names(
            "out_5by5_max_values",
            ["water_depth", "v"],
        ),
        # Same values as 5by5_max_values.ini
        surface_flow_parameters=SurfaceFlowParameters(hmin=0.000001, dtmax=1, cfl=0.8),
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


class TestMaxValues:
    """Test that the maximum values of h and v are properly calculated.

    The simulation tracks hmax and vmax internally. These should equal
    the element-wise maximum across all time-step outputs.
    """

    def test_water_depth_max(self, sim_5by5_max_values):
        """The internal hmax array should match max of all water_depth outputs."""
        output_dict = sim_5by5_max_values.report.raster_provider.output_maps_dict

        # Get all water_depth arrays from output
        h_arrays = [arr for _, arr in output_dict["water_depth"]]

        # Compute element-wise maximum across all time steps
        h_max_computed = np.maximum.reduce(h_arrays)

        # Get the internal hmax array
        h_max_internal = sim_5by5_max_values.get_array("hmax")

        # The overall max values should match
        assert np.isclose(np.nanmax(h_max_computed), np.nanmax(h_max_internal)), (
            f"hmax mismatch: computed max={np.nanmax(h_max_computed):.6f}, "
            f"internal max={np.nanmax(h_max_internal):.6f}"
        )

    def test_velocity_max(self, sim_5by5_max_values):
        """The internal vmax array should match max of all v outputs."""
        output_dict = sim_5by5_max_values.report.raster_provider.output_maps_dict

        # Get all v arrays from output
        v_arrays = [arr for _, arr in output_dict["v"]]

        # Compute element-wise maximum across all time steps
        v_max_computed = np.maximum.reduce(v_arrays)

        # Get the internal vmax array
        v_max_internal = sim_5by5_max_values.get_array("vmax")

        # The overall max values should match
        assert np.isclose(np.nanmax(v_max_computed), np.nanmax(v_max_internal)), (
            f"vmax mismatch: computed max={np.nanmax(v_max_computed):.6f}, "
            f"internal max={np.nanmax(v_max_internal):.6f}"
        )
