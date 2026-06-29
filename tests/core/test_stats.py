from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from itzi.simulation_builder import SimulationBuilder
from itzi.data_containers import SimulationConfig, SurfaceFlowParameters
from itzi.providers.memory_input import MemoryRasterInputProvider, TimedRasterSlice
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
        dtinf=1.0,
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
        expected_inf_vol_first = -2.0 / (1000 * 3600) * domain_data.cell_area
        # Losses: 1.5 mm/h = 1.5/(1000*3600) m/s (negative)
        expected_losses_vol = -1.5 / (1000 * 3600) * area
        expected_losses_vol_first = -1.5 / (1000 * 3600) * domain_data.cell_area
        # Inflow: 0.1 m/s
        expected_inflow_vol = 0.1 * area

        # Ignore first row (initial state, before time-stepping)
        assert np.all(np.isclose(df["rainfall_volume"][1:], expected_rain_vol, atol=0.001))
        assert np.isclose(df["infiltration_volume"].iloc[1], expected_inf_vol_first, atol=0.001)
        assert np.all(np.isclose(df["infiltration_volume"][2:], expected_inf_vol, atol=0.001))
        assert np.all(np.isclose(df["inflow_volume"][1:], expected_inflow_vol, atol=0.001))
        assert np.isclose(df["losses_volume"].iloc[1], expected_losses_vol_first, atol=0.001)
        assert np.all(np.isclose(df["losses_volume"][2:], expected_losses_vol, atol=0.001))

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


def _make_timed_forcing_slices(
    domain_5by5,
    start_time: datetime,
    *,
    forcing_key: str,
    forcing_values_by_second: dict[int, float],
) -> list[TimedRasterSlice]:
    time_slice_seconds = tuple(sorted(forcing_values_by_second))
    forcing_slices: list[TimedRasterSlice] = []

    for index, seconds in enumerate(time_slice_seconds[:-1]):
        slice_start = start_time + timedelta(seconds=seconds)
        final_boundary = start_time + timedelta(seconds=time_slice_seconds[-1])
        if index == len(time_slice_seconds) - 2:
            slice_end = final_boundary
        else:
            slice_end = start_time + timedelta(seconds=time_slice_seconds[index + 1])
        forcing_slices.append(
            TimedRasterSlice(
                start_time=slice_start,
                end_time=slice_end,
                array=np.full(
                    domain_5by5.domain_data.shape,
                    forcing_values_by_second[seconds],
                    dtype=np.float32,
                ),
            )
        )

    if not forcing_slices:
        raise ValueError(f"timed forcing '{forcing_key}' must define at least two boundaries")
    return forcing_slices


def _run_timed_stats_simulation(
    domain_5by5,
    helpers,
    tmp_path,
    *,
    forcing_key: str,
    forcing_values_by_second: dict[int, float],
    infiltration_model: InfiltrationModelType = InfiltrationModelType.NULL,
    initial_water_depth: np.ndarray | None = None,
) -> pd.DataFrame:
    start_time = datetime(2000, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(seconds=20)
    stats_file = tmp_path / f"timed_stats_{forcing_key}.csv"
    forcing_slices = _make_timed_forcing_slices(
        domain_5by5,
        start_time,
        forcing_key=forcing_key,
        forcing_values_by_second=forcing_values_by_second,
    )
    sim_config = SimulationConfig(
        start_time=start_time,
        end_time=end_time,
        record_step=timedelta(seconds=10),
        temporal_type=TemporalType.RELATIVE,
        input_map_names={
            "dem": "dem",
            "friction": "friction",
            "water_depth": "water_depth",
            forcing_key: forcing_key,
        },
        output_map_names=helpers.make_output_map_names(
            f"out_timed_stats_{forcing_key}",
            ["water_depth"],
        ),
        surface_flow_parameters=SurfaceFlowParameters(hmin=0.0001, dtmax=20.0, cfl=0.2),
        infiltration_model=infiltration_model,
        stats_file=str(stats_file),
    )
    input_provider = MemoryRasterInputProvider(
        {
            "domain_data": domain_5by5.domain_data,
            "simulation_start_time": sim_config.start_time,
            "simulation_end_time": sim_config.end_time,
            "static_arrays": {
                "dem": domain_5by5.arr_dem_flat.copy(),
                "friction": domain_5by5.arr_n.copy(),
                "water_depth": (
                    np.zeros(domain_5by5.domain_data.shape, dtype=np.float32)
                    if initial_water_depth is None
                    else initial_water_depth.copy()
                ),
            },
            "timed_arrays": {forcing_key: forcing_slices},
        }
    )
    simulation = (
        SimulationBuilder(sim_config, domain_5by5.arr_mask, np.float32)
        .with_input_provider(input_provider)
        .with_raster_output_provider(
            MemoryRasterOutputProvider({"out_map_names": sim_config.output_map_names})
        )
        .with_vector_output_provider(MemoryVectorOutputProvider({}))
        .build()
    )

    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()
    return pd.read_csv(stats_file)


@pytest.mark.parametrize(
    (
        "forcing_key",
        "forcing_values_by_second",
        "volume_column",
        "expected_volume",
        "infiltration_model",
        "initial_water_depth",
    ),
    [
        (
            "inflow",
            {0: 0.0, 3: 0.1, 20: 0.1},
            "inflow_volume",
            1750.0,
            InfiltrationModelType.NULL,
            None,
        ),
        (
            "rain",
            {0: 0.0, 3: 360.0, 20: 360.0},
            "rainfall_volume",
            1.75,
            InfiltrationModelType.NULL,
            None,
        ),
        (
            "infiltration",
            {0: 0.0, 3: 360.0, 20: 360.0},
            "infiltration_volume",
            -1.75,
            InfiltrationModelType.CONSTANT,
            np.full((5, 5), 1.0, dtype=np.float32),
        ),
        (
            "losses",
            {0: 0.0, 3: 360.0, 20: 360.0},
            "losses_volume",
            -1.75,
            InfiltrationModelType.CONSTANT,
            np.full((5, 5), 1.0, dtype=np.float32),
        ),
    ],
)
def test_timed_input_stats_first_report_row_stays_interval_coherent(
    domain_5by5,
    helpers,
    tmp_path,
    forcing_key: str,
    forcing_values_by_second: dict[int, float],
    volume_column: str,
    expected_volume: float,
    infiltration_model: InfiltrationModelType,
    initial_water_depth: np.ndarray | None,
) -> None:
    df = _run_timed_stats_simulation(
        domain_5by5,
        helpers,
        tmp_path,
        forcing_key=forcing_key,
        forcing_values_by_second=forcing_values_by_second,
        infiltration_model=infiltration_model,
        initial_water_depth=initial_water_depth,
    )

    first_report_row = df.iloc[1]
    volume_change_ref = (
        first_report_row["boundary_volume"]
        + first_report_row["rainfall_volume"]
        + first_report_row["infiltration_volume"]
        + first_report_row["inflow_volume"]
        + first_report_row["losses_volume"]
        + first_report_row["drainage_network_volume"]
        + first_report_row["volume_error"]
    )

    assert np.isclose(first_report_row[volume_column], expected_volume, atol=1e-6, rtol=1e-6)
    assert np.isclose(first_report_row["volume_change"], volume_change_ref, atol=5e-4, rtol=1e-6)


class TestStatsMaps:
    """Test that mean-rate maps reflect the interval that was just simulated."""

    @staticmethod
    def _assert_center_only(arr: np.ndarray, expected_center_value: float) -> None:
        mask = np.ones(arr.shape, dtype=bool)
        mask[2, 2] = False
        assert np.isclose(arr[2, 2], expected_center_value)
        assert np.allclose(arr[mask], 0.0)

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
        """Mean infiltration maps should use the hydrology rates valid for each interval."""
        simulation, _, _ = sim_5by5_stats
        output_dict = simulation.report.raster_provider.output_maps_dict

        for i, (_, arr) in enumerate(output_dict["mean_infiltration"]):
            if i == 0:
                # Initial map is zero
                assert np.allclose(arr, 0)
            elif i == 1:
                # The first simulated interval used the t=0 hydrology rates, when only
                # the center cell had water available for infiltration.
                self._assert_center_only(arr, 2.0)
            else:
                # Mean infiltration should be 2 mm/h
                assert np.isclose(arr.min(), 2.0)
                assert np.isclose(arr.max(), 2.0)

    def test_mean_losses_values(self, sim_5by5_stats):
        """Mean losses maps should use the hydrology rates valid for each interval."""
        simulation, _, _ = sim_5by5_stats
        output_dict = simulation.report.raster_provider.output_maps_dict

        for i, (_, arr) in enumerate(output_dict["mean_losses"]):
            if i == 0:
                # Initial map is zero
                assert np.allclose(arr, 0)
            elif i == 1:
                # The first simulated interval used the t=0 hydrology rates, when only
                # the center cell had water available for losses.
                self._assert_center_only(arr, 1.5)
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
