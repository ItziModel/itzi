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
    arr_shape = (13, 34)
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


# @pytest.mark.parametrize("temporal_type", ["absolute", "relative"])
def write_arrays(
    icechunk_provider: IcechunkRasterOutputProvider,
    temp_dir: tempfile.TemporaryDirectory,
    temporal_type: str,
    maps_dict: Dict,
):
    # Write timesteps, with 1 minute in between
    time_steps_num = 3
    if temporal_type == "relative":
        sim_time = timedelta(seconds=0)
    else:
        sim_time = datetime(year=34, month=1, day=1)

    for t in range(time_steps_num):
        sim_time += timedelta(minutes=t)
        print(sim_time)
        icechunk_provider.write_arrays(maps_dict, sim_time)

    # Read the data
    storage = icechunk.local_filesystem_storage(temp_dir.name)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    # z_store = zarr.open_group(session.store, mode="r")

    # Do the checks
    # arr_time = z_store['time']
    # print(arr_time.dtype)
    # print(np.min(arr_time))
    # print(np.max(arr_time))
    # print(type(arr_time))
    # print(np.array(arr_time))

    ds = xr.open_zarr(session.store)
    da_time = ds["time"]
    assert da_time.shape == (time_steps_num,)
    timestep = da_time[1] - da_time[0]
    # pydt = timestep.dt.to_pydatetime()
    print(f"{da_time=}")
    print(f"{timestep.data=}")
    # print(f"{pydt=}")
    # if temporal_type == "relative":
    #     assert isinstance(arr_time.data.dtype, np.dtypes.TimeDelta64DType)
    #     assert arr_time.data.dtype == 'timedelta64[ms]'
    # else:
    #     assert isinstance(arr_time.data.dtype, np.dtypes.DateTime64DType)
    #     assert arr_time.data.dtype == 'datetime64[ms]'


@pytest.mark.parametrize("year", [1, 123, 1978, 3456])
def test_write_arrays_absolute(
    icechunk_provider: IcechunkRasterOutputProvider,
    temp_dir: tempfile.TemporaryDirectory,
    year: int,
    maps_dict: Dict,
):
    # Write timesteps, with 1 minute in between
    time_steps_num = 3
    sim_time = datetime(year=year, month=1, day=1)
    reference_timestep = timedelta(minutes=1)
    for t in range(time_steps_num):
        sim_time += reference_timestep
        print(sim_time)
        icechunk_provider.write_arrays(maps_dict, sim_time)

    # Read the data
    storage = icechunk.local_filesystem_storage(temp_dir.name)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    # z_store = zarr.open_group(session.store, mode="r")

    # Do the checks
    # arr_time = z_store['time']
    # print(arr_time.dtype)
    # print(np.min(arr_time))
    # print(np.max(arr_time))
    # print(type(arr_time))
    # print(np.array(arr_time))

    ds = xr.open_zarr(session.store)
    da_time = ds["time"]
    assert da_time.shape == (time_steps_num,)
    timestep = da_time[1] - da_time[0]
    print(f"{da_time=}")
    print(f"{timestep.data=}")
    timestep_py = pd.to_timedelta(timestep.data).to_pytimedelta()
    assert timestep_py == reference_timestep

    assert False
