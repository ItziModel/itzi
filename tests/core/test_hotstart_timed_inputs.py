"""Hotstart integration tests for provider-backed timed xarray inputs."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("xarray")

import xarray as xr

from itzi.const import InfiltrationModelType, TemporalType
from itzi.data_containers import SimulationConfig, SurfaceFlowParameters
from itzi.itzi_error import ItziFatal
from itzi.providers.memory_output import MemoryRasterOutputProvider, MemoryVectorOutputProvider
from itzi.providers.xarray_input import XarrayRasterInputProvider
from itzi.simulation_builder import SimulationBuilder

if TYPE_CHECKING:
    from itzi.simulation import Simulation


# Mark all tests in this module as cloud tests
pytestmark = pytest.mark.cloud

TIME_SLICE_SECONDS = (0, 10, 20, 30)
RAIN_MM_PER_HOUR = {
    0: 0.0,
    10: 180.0,
    20: 360.0,
    30: 540.0,
}


def _run_to_end(simulation: "Simulation", *, skip_initialize: bool = False) -> None:
    if not skip_initialize:
        simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()


def _assert_final_state_matches(resumed: "Simulation", reference: "Simulation") -> None:
    for key in ["water_depth", "qe", "qs"]:
        np.testing.assert_allclose(
            resumed.raster_domain.get_array(key),
            reference.raster_domain.get_array(key),
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"Final {key} mismatch",
        )


def _make_xarray_input_dataset(
    domain_5by5,
    start_time: datetime,
    *,
    use_relative_time: bool,
) -> tuple[xr.Dataset, dict[int, np.ndarray]]:
    coordinates = domain_5by5.domain_data.get_coordinates()
    shape = domain_5by5.domain_data.shape

    if use_relative_time:
        time_coords = np.array(
            [np.timedelta64(seconds, "s") for seconds in TIME_SLICE_SECONDS],
            dtype="timedelta64[s]",
        )
    else:
        time_coords = np.array(
            [
                np.datetime64(start_time + timedelta(seconds=seconds), "s")
                for seconds in TIME_SLICE_SECONDS
            ],
            dtype="datetime64[s]",
        )

    expected_rain_arrays: dict[int, np.ndarray] = {}
    rain_stack: list[np.ndarray] = []
    for seconds in TIME_SLICE_SECONDS:
        rain_mm_per_hour = RAIN_MM_PER_HOUR[seconds]
        rain_stack.append(np.full(shape, rain_mm_per_hour, dtype=np.float32))
        expected_rain_arrays[seconds] = np.full(
            shape,
            rain_mm_per_hour / (1000 * 3600),
            dtype=np.float32,
        )

    dataset = xr.Dataset(
        {
            "dem": (["y", "x"], domain_5by5.arr_dem_flat.copy()),
            "friction": (["y", "x"], domain_5by5.arr_n.copy()),
            "water_depth": (["y", "x"], domain_5by5.arr_start_h.copy()),
            "rain": (["time", "y", "x"], np.stack(rain_stack, axis=0)),
        },
        coords={
            "time": time_coords,
            "x": coordinates["x"],
            "y": coordinates["y"],
        },
        attrs={"crs_wkt": domain_5by5.domain_data.crs_wkt},
    )
    return dataset, expected_rain_arrays


def _make_simulation_config(
    start_time: datetime,
    end_time: datetime,
    *,
    temporal_type: TemporalType,
) -> SimulationConfig:
    return SimulationConfig(
        start_time=start_time,
        end_time=end_time,
        record_step=timedelta(seconds=10),
        temporal_type=temporal_type,
        input_map_names={
            "dem": "dem",
            "friction": "friction",
            "water_depth": "water_depth",
            "rain": "rain",
        },
        output_map_names={"water_depth": "out_hotstart_timed_inputs_water_depth"},
        surface_flow_parameters=SurfaceFlowParameters(hmin=0.0001, dtmax=0.3, cfl=0.2),
        infiltration_model=InfiltrationModelType.NULL,
    )


def _build_provider_simulation(
    sim_config: SimulationConfig,
    domain_5by5,
    dataset: xr.Dataset,
    *,
    hotstart_bytes: bytes | None = None,
) -> "Simulation":
    input_provider = XarrayRasterInputProvider(
        {
            "dataset": dataset,
            "input_map_names": sim_config.input_map_names,
            "simulation_start_time": sim_config.start_time,
            "simulation_end_time": sim_config.end_time,
        }
    )

    builder = (
        SimulationBuilder(sim_config, domain_5by5.arr_mask, np.float32)
        .with_input_provider(input_provider)
        .with_raster_output_provider(
            MemoryRasterOutputProvider({"out_map_names": sim_config.output_map_names})
        )
        .with_vector_output_provider(MemoryVectorOutputProvider({}))
    )
    if hotstart_bytes is not None:
        builder.with_hotstart(hotstart_bytes)
    return builder.build()


def _run_reference_with_hotstart_checkpoint(
    sim_config: SimulationConfig,
    domain_5by5,
    dataset: xr.Dataset,
    split_target_time: datetime,
) -> tuple[dict[str, datetime | np.ndarray], bytes, "Simulation"]:
    simulation = _build_provider_simulation(sim_config, domain_5by5, dataset)
    simulation.initialize()

    while simulation.sim_time < split_target_time:
        simulation.update()

    checkpoint = {
        "sim_time": simulation.sim_time,
        "rain": simulation.raster_domain.get_array("rain").copy(),
    }
    hotstart_bytes = simulation.create_hotstart().getvalue()

    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()

    return checkpoint, hotstart_bytes, simulation


def _assert_resume_with_timed_xarray_inputs(
    domain_5by5,
    *,
    use_relative_time: bool,
) -> None:
    start_time = datetime(2000, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(seconds=25)
    split_target_time = start_time + timedelta(seconds=12)
    temporal_type = TemporalType.RELATIVE if use_relative_time else TemporalType.ABSOLUTE

    dataset, expected_rain_arrays = _make_xarray_input_dataset(
        domain_5by5,
        start_time,
        use_relative_time=use_relative_time,
    )
    sim_config = _make_simulation_config(
        start_time,
        end_time,
        temporal_type=temporal_type,
    )
    checkpoint, hotstart_bytes, reference = _run_reference_with_hotstart_checkpoint(
        sim_config,
        domain_5by5,
        dataset,
        split_target_time,
    )

    saved_sim_time = checkpoint["sim_time"]
    second_slice_start = start_time + timedelta(seconds=10)
    second_slice_end = start_time + timedelta(seconds=20)
    assert second_slice_start <= saved_sim_time < second_slice_end

    resumed = _build_provider_simulation(
        sim_config,
        domain_5by5,
        dataset,
        hotstart_bytes=hotstart_bytes,
    )

    assert resumed.sim_time == saved_sim_time
    np.testing.assert_allclose(resumed.raster_domain.get_array("rain"), checkpoint["rain"])
    np.testing.assert_allclose(resumed.raster_domain.get_array("rain"), expected_rain_arrays[10])

    # TimedArray cache is rebuilt from the fresh provider during construction.
    # After resume, the first update must realign that cache to the restored clock.
    resumed.update()

    assert resumed.timed_arrays is not None
    rain_timed_array = resumed.timed_arrays["rain"]
    assert second_slice_start <= resumed.sim_time < second_slice_end
    assert rain_timed_array.arr_start == second_slice_start
    assert rain_timed_array.arr_end == second_slice_end
    np.testing.assert_allclose(resumed.raster_domain.get_array("rain"), expected_rain_arrays[10])
    assert not np.allclose(resumed.raster_domain.get_array("rain"), expected_rain_arrays[0])

    _run_to_end(resumed, skip_initialize=True)
    _assert_final_state_matches(resumed, reference)


def test_resume_with_relative_time_xarray_inputs(domain_5by5) -> None:
    _assert_resume_with_timed_xarray_inputs(
        domain_5by5,
        use_relative_time=True,
    )


def test_resume_with_absolute_time_xarray_inputs(domain_5by5) -> None:
    _assert_resume_with_timed_xarray_inputs(
        domain_5by5,
        use_relative_time=False,
    )


def test_build_fails_when_dem_input_has_only_nan_cells(domain_5by5) -> None:
    start_time = datetime(2000, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(seconds=25)
    dataset, _ = _make_xarray_input_dataset(
        domain_5by5,
        start_time,
        use_relative_time=False,
    )
    dataset["dem"] = xr.DataArray(
        np.full(domain_5by5.domain_data.shape, np.nan, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": dataset["y"], "x": dataset["x"]},
    )
    sim_config = _make_simulation_config(
        start_time,
        end_time,
        temporal_type=TemporalType.ABSOLUTE,
    )

    with pytest.raises(ItziFatal, match=r"input map <dem> contains only NULL/NaN cells"):
        _build_provider_simulation(sim_config, domain_5by5, dataset)


def test_build_warns_when_non_dem_input_has_only_nan_cells(domain_5by5, caplog) -> None:
    start_time = datetime(2000, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(seconds=25)
    dataset, _ = _make_xarray_input_dataset(
        domain_5by5,
        start_time,
        use_relative_time=False,
    )
    dataset["rain"] = xr.DataArray(
        np.full(domain_5by5.domain_data.shape, np.nan, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": dataset["y"], "x": dataset["x"]},
    )
    sim_config = _make_simulation_config(
        start_time,
        end_time,
        temporal_type=TemporalType.ABSOLUTE,
    )

    itzi_logger = logging.getLogger("itzi")
    with caplog.at_level(logging.WARNING, logger="itzi"):
        itzi_logger.addHandler(caplog.handler)
        try:
            simulation = _build_provider_simulation(sim_config, domain_5by5, dataset)
        finally:
            itzi_logger.removeHandler(caplog.handler)

    warning_messages = [record.message for record in caplog.records]
    assert any(
        "input map <rain> contains only NULL/NaN cells inside the active domain" in message
        for message in warning_messages
    )
    assert np.allclose(simulation.raster_domain.get_array("rain"), 0.0)
