from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest

from itzi.simulation_builder import SimulationBuilder
from itzi.data_containers import SimulationConfig, SurfaceFlowParameters
from itzi.providers.memory_output import MemoryRasterOutputProvider, MemoryVectorOutputProvider
from itzi.providers.domain_data import DomainData
from itzi.const import InfiltrationModelType, TemporalType

if TYPE_CHECKING:
    from itzi.simulation import Simulation


def _run_center_pulse_simulation(
    domain_5by5,
    helpers,
    *,
    dx: float,
    dy: float,
    duration_s: float = 1.0,
) -> np.ndarray:
    rows, cols = domain_5by5.domain_data.shape
    domain_data = DomainData(
        north=rows * dy,
        south=0.0,
        east=cols * dx,
        west=0.0,
        rows=rows,
        cols=cols,
        crs_wkt="",
    )

    initial_volume = float(np.sum(domain_5by5.arr_start_h) * domain_5by5.domain_data.cell_area)
    arr_start_h = np.zeros(domain_data.shape, dtype=np.float32)
    arr_start_h[2, 2] = initial_volume / domain_data.cell_area

    sim_config = SimulationConfig(
        start_time=datetime(2000, 1, 1, 0, 0, 0),
        end_time=datetime(2000, 1, 1, 0, 0, 0) + timedelta(seconds=duration_s),
        record_step=timedelta(seconds=duration_s),
        temporal_type=TemporalType.RELATIVE,
        input_map_names=helpers.make_input_map_names(
            dem="z",
            friction="n",
            water_depth="start_h",
        ),
        output_map_names=helpers.make_output_map_names(
            f"out_5by5_rect_{int(dx)}x{int(dy)}",
            ["water_depth"],
        ),
        surface_flow_parameters=SurfaceFlowParameters(hmin=0.0001, dtmax=0.3, cfl=0.2),
        infiltration_model=InfiltrationModelType.NULL,
    )

    raster_output = MemoryRasterOutputProvider({"out_map_names": sim_config.output_map_names})
    simulation = (
        SimulationBuilder(sim_config, domain_5by5.arr_mask, np.float32)
        .with_domain_data(domain_data)
        .with_raster_output_provider(raster_output)
        .with_vector_output_provider(MemoryVectorOutputProvider({}))
        .build()
    )

    simulation.set_array("dem", domain_5by5.arr_dem_flat.copy())
    simulation.set_array("friction", domain_5by5.arr_n.copy())
    simulation.set_array("water_depth", arr_start_h.copy())

    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    final_depth = simulation.get_array("water_depth").copy()
    simulation.finalize()
    return final_depth


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


@pytest.mark.parametrize(
    ("dx", "dy", "wetter_axis"),
    [
        (20.0, 10.0, "y"),
        (10.0, 20.0, "x"),
    ],
)
def test_rectangular_cells_preserve_axis_symmetry_and_bias_early_spreading(
    domain_5by5, helpers, dx: float, dy: float, wetter_axis: str
):
    """Early-time spreading should stay symmetric within each axis and favor the shorter cell size."""
    h_array = _run_center_pulse_simulation(domain_5by5, helpers, dx=dx, dy=dy)

    h_above = h_array[1, 2]
    h_below = h_array[3, 2]
    h_left = h_array[2, 1]
    h_right = h_array[2, 3]

    assert np.isclose(h_above, h_below)
    assert np.isclose(h_left, h_right)

    if wetter_axis == "y":
        assert min(h_above, h_below) > max(h_left, h_right)
    else:
        assert min(h_left, h_right) > max(h_above, h_below)


def test_swapping_dx_and_dy_transposes_the_early_time_solution(domain_5by5, helpers):
    """Swapping the cell sizes should swap the x/y spreading pattern."""
    h_dx20_dy10 = _run_center_pulse_simulation(domain_5by5, helpers, dx=20.0, dy=10.0)
    h_dx10_dy20 = _run_center_pulse_simulation(domain_5by5, helpers, dx=10.0, dy=20.0)

    assert np.allclose(h_dx20_dy10, h_dx10_dy20.T)
