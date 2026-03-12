from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from itzi.simulation_builder import SimulationBuilder
from itzi.data_containers import SimulationConfig, SurfaceFlowParameters
from itzi.providers.memory_output import MemoryRasterOutputProvider, MemoryVectorOutputProvider
from itzi.const import InfiltrationModelType, TemporalType


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
