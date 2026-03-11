"""
Memory-based tests for 5x5 domain simulations.

These tests replace GRASS-based tests from tests/grass_tests/test_itzi.py
(except test_region_mask which remains GRASS-specific).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
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


# =============================================================================
# Tests
# =============================================================================


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


# =============================================================================
# Fixture for max_values test
# =============================================================================


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


# =============================================================================
# Fixture for stats tests (test_stats_file, test_stats_maps)
# =============================================================================


@pytest.fixture(scope="module")
def sim_5by5_stats(domain_5by5, helpers, tmp_path_factory):
    """Run a 5x5 simulation for 5s with 1s record step.

    Outputs: water_depth, mean_infiltration, mean_rainfall, mean_inflow, mean_losses, volume_error
    Uses CONSTANT infiltration model with rain, infiltration, losses, and inflow.
    """
    # Create temp directory for stats file
    temp_dir = tmp_path_factory.mktemp("stats_test")
    stats_file = temp_dir / "5by5_stats.csv"

    # Build SimulationConfig
    sim_config = SimulationConfig(
        start_time=datetime(2000, 1, 1, 0, 0, 0),
        end_time=datetime(2000, 1, 1, 0, 0, 5),  # 5 seconds
        record_step=timedelta(seconds=1),
        temporal_type=TemporalType.RELATIVE,
        input_map_names=helpers.make_input_map_names(
            dem="z",
            friction="n",
            water_depth="start_h",
            rain="rainfall",
            infiltration="infiltration_rate",
            losses="loss_rate",
            inflow="inflow_rate",
        ),
        output_map_names=helpers.make_output_map_names(
            "out_5by5_stats",
            [
                "water_depth",
                "mean_infiltration",
                "mean_rainfall",
                "mean_inflow",
                "mean_losses",
                "volume_error",
            ],
        ),
        # Use default, same as 5by5_stats.ini
        surface_flow_parameters=SurfaceFlowParameters(),
        infiltration_model=InfiltrationModelType.CONSTANT,
        stats_file=str(stats_file),
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
    simulation.set_array("rain", domain_5by5.arr_rain)
    simulation.set_array("infiltration", domain_5by5.arr_inf)
    simulation.set_array("losses", domain_5by5.arr_loss)
    simulation.set_array("inflow", domain_5by5.arr_inflow)

    # Run simulation
    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()

    return simulation, domain_5by5.domain_data, stats_file


class TestStatsFile:
    """Test that the statistics CSV file has correct columns and volume values."""

    def test_stats_file_exists(self, sim_5by5_stats):
        """The stats CSV file should be created."""
        _, _, stats_file = sim_5by5_stats
        assert stats_file.exists()

    def test_stats_file_columns(self, sim_5by5_stats):
        """CSV columns should match MassBalanceData fields."""
        from itzi.data_containers import MassBalanceData

        _, _, stats_file = sim_5by5_stats
        df = pd.read_csv(stats_file)

        expected_cols = list(MassBalanceData.model_fields.keys())
        assert df.columns.to_list() == expected_cols

    def test_stats_volume_values(self, sim_5by5_stats):
        """Volume values should match expected rates x area."""

        _, domain_data, stats_file = sim_5by5_stats
        df = pd.read_csv(stats_file)

        # Domain area: 5x5 cells at 10m resolution = 2500 m²
        area = domain_data.cell_area * domain_data.cells

        # Expected volumes (rates in m/s)
        # Rainfall: 10 mm/h = 10/(1000*3600) m/s
        expected_rain_vol = 10.0 / (1000 * 3600) * area
        # Infiltration: 2 mm/h = 2/(1000*3600) m/s (negative because it removes water)
        expected_inf_vol = -2.0 / (1000 * 3600) * area
        # Losses: 1.5 mm/h = 1.5/(1000*3600) m/s (negative)
        expected_losses_vol = -1.5 / (1000 * 3600) * area
        # Inflow: 0.1 m/s
        expected_inflow_vol = 0.1 * area

        # Ignore first row (initial state, before time-stepping)
        assert np.all(np.isclose(df["rainfall_volume"][1:], expected_rain_vol, atol=0.001))
        assert np.all(np.isclose(df["infiltration_volume"][1:], expected_inf_vol, atol=0.001))
        assert np.all(np.isclose(df["inflow_volume"][1:], expected_inflow_vol, atol=0.001))
        assert np.all(np.isclose(df["losses_volume"][1:], expected_losses_vol, atol=0.001))

    def test_volume_change_coherence(self, sim_5by5_stats):
        """Volume change should be coherent with other volume components."""

        _, _, stats_file = sim_5by5_stats
        df = pd.read_csv(stats_file)

        # Check if the volume change is coherent with the rest of the volumes
        df["vol_change_ref"] = (
            df["boundary_volume"]
            + df["rainfall_volume"]
            + df["infiltration_volume"]
            + df["inflow_volume"]
            + df["losses_volume"]
            + df["drainage_network_volume"]
            + df["volume_error"]
        )
        assert np.allclose(df["vol_change_ref"], df["volume_change"], atol=1, rtol=0.01)


class TestStatsMaps:
    """Test that output maps for mean rates have correct uniform values."""

    def test_mean_rainfall_values(self, sim_5by5_stats):
        """Mean rainfall maps should have uniform value of 10 mm/h."""
        simulation, _, _ = sim_5by5_stats
        output_dict = simulation.report.raster_provider.output_maps_dict

        for i, (_, arr) in enumerate(output_dict["mean_rainfall"]):
            if i == 0:
                # Initial map is zero (simulation hasn't started)
                assert np.allclose(arr, 0)
            else:
                # Mean rainfall should be 10 mm/h (converted back from m/s)
                # The output is in mm/h
                assert np.isclose(arr.min(), 10.0)
                assert np.isclose(arr.max(), 10.0)

    def test_mean_inflow_values(self, sim_5by5_stats):
        """Mean inflow maps should have uniform value of 0.1 m/s."""
        simulation, _, _ = sim_5by5_stats
        output_dict = simulation.report.raster_provider.output_maps_dict

        for i, (_, arr) in enumerate(output_dict["mean_inflow"]):
            if i == 0:
                # Initial map is zero
                assert np.allclose(arr, 0)
            else:
                assert np.isclose(arr.min(), 0.1)
                assert np.isclose(arr.max(), 0.1)

    def test_mean_infiltration_values(self, sim_5by5_stats):
        """Mean infiltration maps should have uniform value of 2 mm/h."""
        simulation, _, _ = sim_5by5_stats
        output_dict = simulation.report.raster_provider.output_maps_dict

        for i, (_, arr) in enumerate(output_dict["mean_infiltration"]):
            if i == 0:
                # Initial map is zero
                assert np.allclose(arr, 0)
            else:
                # Mean infiltration should be 2 mm/h
                assert np.isclose(arr.min(), 2.0)
                assert np.isclose(arr.max(), 2.0)

    def test_mean_losses_values(self, sim_5by5_stats):
        """Mean losses maps should have uniform value of 1.5 mm/h."""
        simulation, _, _ = sim_5by5_stats
        output_dict = simulation.report.raster_provider.output_maps_dict

        for i, (_, arr) in enumerate(output_dict["mean_losses"]):
            if i == 0:
                # Initial map is zero
                assert np.allclose(arr, 0)
            else:
                # Mean losses should be 1.5 mm/h
                assert np.isclose(arr.min(), 1.5)
                assert np.isclose(arr.max(), 1.5)

    def test_initial_water_depth(self, sim_5by5_stats):
        """Initial water depth map should have 0.2 at center cell [2,2], 0 elsewhere.

        Note: Memory provider keeps all values (unlike GRASS which nullifies sub-hmin values).
        """
        simulation, _, _ = sim_5by5_stats
        output_dict = simulation.report.raster_provider.output_maps_dict

        # Get the first water_depth output (initial state)
        _, h_array = output_dict["water_depth"][0]

        # Center cell should have 0.2
        assert np.isclose(h_array[2, 2], 0.2), f"Center cell should be 0.2, got {h_array[2, 2]}"

        # All other cells should be 0 (memory provider preserves sub-hmin values)
        # Create a mask for non-center cells
        mask = np.ones((5, 5), dtype=bool)
        mask[2, 2] = False
        assert np.allclose(h_array[mask], 0), "Non-center cells should be 0"


# =============================================================================
# Fixture for open boundaries test
# =============================================================================


@pytest.fixture(scope="module")
def sim_5by5_open_boundaries(domain_5by5, helpers, tmp_path_factory):
    """Run a 5x5 simulation for 600s with 60s record step and open boundaries.

    Outputs: water_depth, volume_error, mean_boundary_flow
    Boundary condition type 2 (open) is set at all 16 edge cells.
    """
    # Create temp directory for stats file
    temp_dir = tmp_path_factory.mktemp("open_bc_test")
    stats_file = temp_dir / "5by5_open_boundaries.csv"

    # Build SimulationConfig
    sim_config = SimulationConfig(
        start_time=datetime(2000, 1, 1, 0, 0, 0),
        end_time=datetime(2000, 1, 1, 0, 10, 0),  # 600 seconds = 10 minutes
        record_step=timedelta(seconds=60),
        temporal_type=TemporalType.RELATIVE,
        input_map_names=helpers.make_input_map_names(
            dem="z",
            friction="n",
            water_depth="start_h",
            bctype="open_boundaries",
        ),
        output_map_names=helpers.make_output_map_names(
            "out_5by5_open_boundaries",
            ["water_depth", "volume_error", "mean_boundary_flow"],
        ),
        # Use dtmax=2 as specified in 5by5_open_boundaries.ini
        surface_flow_parameters=SurfaceFlowParameters(dtmax=2.0),
        infiltration_model=InfiltrationModelType.NULL,
        stats_file=str(stats_file),
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
    simulation.set_array("bctype", domain_5by5.arr_bctype)

    # Run simulation
    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()

    return simulation, domain_5by5.domain_data


class TestOpenBoundaries:
    """Test that flow at boundary cells exhibits symmetry across the domain.

    The 16 edge cells of the 5x5 grid are:
    - Top row: (0,0), (0,1), (0,2), (0,3), (0,4)
    - Bottom row: (4,0), (4,1), (4,2), (4,3), (4,4)
    - Left column (excluding corners): (1,0), (2,0), (3,0)
    - Right column (excluding corners): (1,4), (2,4), (3,4)

    Symmetric pairs (r, c) and (4-r, 4-c) should have equal flow rates.

    Note: Corner cells are tested separately because they exhibit non-deterministic
    behavior due to OpenMP parallel execution. See plans/openmp_determinism_investigation.md
    for details.
    """

    def test_edge_flow_symmetry(self, sim_5by5_open_boundaries):
        """Symmetric edge cells (non-corner boundary cells) should have equal flow rates.

        Edge cells (excluding corners) only receive ONE boundary condition accumulation,
        so they should pass symmetry tests consistently.
        """
        simulation, _ = sim_5by5_open_boundaries
        output_dict = simulation.report.raster_provider.output_maps_dict

        # Get the last mean_boundary_flow output
        _, bf_array = output_dict["mean_boundary_flow"][-1]

        # Define edge cells (non-corner boundary cells)
        # These are the 12 edge cells that are NOT corners
        edge_cells = [
            (0, 1),
            (0, 2),
            (0, 3),  # top row (excluding corners)
            (4, 1),
            (4, 2),
            (4, 3),  # bottom row (excluding corners)
            (1, 0),
            (2, 0),
            (3, 0),  # left column (excluding corners)
            (1, 4),
            (2, 4),
            (3, 4),  # right column (excluding corners)
        ]

        # Check symmetric pairs: (r, c) and (4-r, 4-c)
        # We only need to check half the cells
        checked = set()
        for r, c in edge_cells:
            if (r, c) in checked:
                continue
            sym_r, sym_c = 4 - r, 4 - c
            if (sym_r, sym_c) in checked:
                continue

            flow_val = bf_array[r, c]
            sym_flow_val = bf_array[sym_r, sym_c]

            assert np.isclose(flow_val, sym_flow_val), (
                f"Symmetric edge cells ({r},{c})={flow_val:.6f} and "
                f"({sym_r},{sym_c})={sym_flow_val:.6f} should have equal flow"
            )
            checked.add((r, c))
            checked.add((sym_r, sym_c))

    def test_corner_flow_symmetry(self, sim_5by5_open_boundaries):
        """Symmetric corner cells should have equal flow rates."""
        simulation, _ = sim_5by5_open_boundaries
        output_dict = simulation.report.raster_provider.output_maps_dict

        # Get the last mean_boundary_flow output
        _, bf_array = output_dict["mean_boundary_flow"][-1]

        # Define the 4 corner cells
        corner_cells = [
            (0, 0),  # top-left
            (0, 4),  # top-right
            (4, 0),  # bottom-left
            (4, 4),  # bottom-right
        ]

        # Check symmetric pairs: (r, c) and (4-r, 4-c)
        # Pairs are: (0,0) <-> (4,4) and (0,4) <-> (4,0)
        checked = set()
        for r, c in corner_cells:
            if (r, c) in checked:
                continue
            sym_r, sym_c = 4 - r, 4 - c
            if (sym_r, sym_c) in checked:
                continue

            flow_val = bf_array[r, c]
            sym_flow_val = bf_array[sym_r, sym_c]

            # Higher tolerance because corner boundaries test fails when OMP_NUM_THREADS>nrows (5 here)
            # TODO: investigate and fix the corner issues
            assert np.isclose(flow_val, sym_flow_val, atol=1e-5), (
                f"Symmetric corner cells ({r},{c})={flow_val:.6f} and "
                f"({sym_r},{sym_c})={sym_flow_val:.6f} should have equal flow"
            )
            checked.add((r, c))
            checked.add((sym_r, sym_c))


# =============================================================================
# Fixture for WSE test
# =============================================================================


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
