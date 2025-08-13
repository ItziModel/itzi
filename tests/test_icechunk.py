import tempfile
from typing import Dict
from datetime import datetime, timedelta

import icechunk
import xarray as xr
import numpy as np
import pandas as pd
import pytest
import pyproj

from itzi.providers.icechunk_output import IcechunkRasterOutputProvider
from itzi.array_definitions import ARRAY_DEFINITIONS, ArrayCategory


@pytest.fixture
def temp_dir():
    return tempfile.TemporaryDirectory()


@pytest.fixture
def maps_dict():
    """A dict representing the arrays to be written to disk"""
    key_list = [
        arr_def.key for arr_def in ARRAY_DEFINITIONS if ArrayCategory.OUTPUT in arr_def.category
    ]
    rng = np.random.default_rng()
    arr_shape = (6, 9)
    return {key: rng.random(size=arr_shape, dtype=np.float32) for key in key_list}


@pytest.fixture
def icechunk_provider(maps_dict: Dict, temp_dir: tempfile.TemporaryDirectory):
    storage = icechunk.local_filesystem_storage(temp_dir.name)
    out_map_names = {key: f"itzi_test_{key}" for key in maps_dict.keys()}
    arr_shape = next(iter(maps_dict.values())).shape
    y_coords = np.linspace(start=1234, stop=1234 + arr_shape[0], num=arr_shape[0])
    x_coords = np.linspace(start=1234, stop=1234 + arr_shape[1], num=arr_shape[1])
    crs = pyproj.CRS.from_epsg(6372)  # Mexico LCC
    provider_config = {
        "out_map_names": out_map_names,
        "crs": crs,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "icechunk_storage": storage,
    }
    icechunk_p = IcechunkRasterOutputProvider()
    icechunk_p.initialize(provider_config)
    return icechunk_p


@pytest.mark.parametrize("start_year", [1, 123, 1978, 3456])
@pytest.mark.parametrize("time_step_s", [1, 60, 300, 3600])
def test_write_arrays_absolute(
    icechunk_provider: IcechunkRasterOutputProvider,
    temp_dir: tempfile.TemporaryDirectory,
    start_year: int,
    time_step_s: int,
    maps_dict: Dict,
):
    # Write timesteps
    time_steps_num = 3
    sim_time = datetime(year=start_year, month=1, day=1)
    reference_timestep = timedelta(seconds=time_step_s)
    expected_times = []
    for t in range(time_steps_num):
        sim_time += reference_timestep
        expected_times.append(sim_time)
        print(sim_time)
        icechunk_provider.write_arrays(maps_dict, sim_time)

    # Read the data
    storage = icechunk.local_filesystem_storage(temp_dir.name)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")

    ds = xr.open_zarr(session.store)
    da_time = ds["time"]
    assert da_time.shape == (time_steps_num,)
    timestep = da_time[1] - da_time[0]

    # Assert that the timestep is correct
    timestep_py = pd.to_timedelta(timestep.data).to_pytimedelta()
    assert timestep_py == reference_timestep

    # Assert that all individual timestamps are correct
    actual_times = [pd.to_datetime(t).to_pydatetime() for t in da_time.values]
    assert len(actual_times) == len(expected_times)
    for actual, expected in zip(actual_times, expected_times):
        assert actual == expected, f"Expected {expected}, got {actual}"


@pytest.mark.parametrize("start_seconds", [0, 10, 3600, 86400])
@pytest.mark.parametrize("time_step_s", [1, 60, 300, 3600])
def test_write_arrays_relative(
    icechunk_provider: IcechunkRasterOutputProvider,
    temp_dir: tempfile.TemporaryDirectory,
    start_seconds: int,
    time_step_s: int,
    maps_dict: Dict,
):
    """Test writing arrays with relative time (timedelta)"""
    # Write timesteps, with 1 minute in between
    time_steps_num = 3
    sim_time = timedelta(seconds=start_seconds)
    reference_timestep = timedelta(seconds=time_step_s)
    expected_times = []
    for t in range(time_steps_num):
        sim_time += reference_timestep
        expected_times.append(sim_time)
        print(sim_time)
        icechunk_provider.write_arrays(maps_dict, sim_time)

    # Read the data
    storage = icechunk.local_filesystem_storage(temp_dir.name)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")

    ds = xr.open_zarr(session.store)
    da_time = ds["time"]
    assert da_time.shape == (time_steps_num,)
    timestep = da_time[1] - da_time[0]

    # Assert that the timestep is correct
    timestep_py = pd.to_timedelta(timestep.data).to_pytimedelta()
    assert timestep_py == reference_timestep

    # Assert that all individual timestamps are correct
    actual_times = [pd.to_timedelta(t).to_pytimedelta() for t in da_time.values]
    assert len(actual_times) == len(expected_times)
    for actual, expected in zip(actual_times, expected_times):
        assert actual == expected, f"Expected {expected}, got {actual}"
