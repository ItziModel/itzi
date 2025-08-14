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


@pytest.fixture(scope="function")
def temp_dir():
    return tempfile.TemporaryDirectory()


@pytest.fixture(scope="module")
def maps_dict():
    """A dict representing the arrays to be written to disk"""
    key_list = [
        arr_def.key for arr_def in ARRAY_DEFINITIONS if ArrayCategory.OUTPUT in arr_def.category
    ]
    rng = np.random.default_rng()
    arr_shape = (6, 9)
    return {key: rng.random(size=arr_shape, dtype=np.float32) for key in key_list}


@pytest.fixture(scope="module")
def coordinates(maps_dict: Dict):
    """Generate x and y coordinates for the test arrays"""
    arr_shape = next(iter(maps_dict.values())).shape
    y_coords = np.linspace(start=1234, stop=1234 + arr_shape[0], num=arr_shape[0])
    x_coords = np.linspace(start=1234, stop=1234 + arr_shape[1], num=arr_shape[1])
    return {"x_coords": x_coords, "y_coords": y_coords}


@pytest.fixture(scope="module")
def crs():
    """CRS for the test data"""
    return pyproj.CRS.from_epsg(6372)  # Mexico LCC


@pytest.fixture(scope="module")
def out_var_names(maps_dict: Dict):
    """Output map names for the test arrays"""
    return list(maps_dict.keys())


@pytest.fixture
def icechunk_provider(
    temp_dir: tempfile.TemporaryDirectory, coordinates: Dict, crs: pyproj.CRS, out_var_names: list
):
    storage = icechunk.local_filesystem_storage(temp_dir.name)
    provider_config = {
        "out_var_names": out_var_names,
        "crs": crs,
        "x_coords": coordinates["x_coords"],
        "y_coords": coordinates["y_coords"],
        "icechunk_storage": storage,
    }
    icechunk_p = IcechunkRasterOutputProvider()
    icechunk_p.initialize(provider_config)
    return icechunk_p


@pytest.mark.parametrize("start_year", [1, 1978, 3456])
@pytest.mark.parametrize("time_step_s", [1, 60, 300])
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


@pytest.mark.parametrize(
    "start_seconds",
    [
        0,
        300,
    ],
)
@pytest.mark.parametrize("time_step_s", [1, 60, 3600])
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


def test_data_consistency(
    icechunk_provider: IcechunkRasterOutputProvider,
    temp_dir: tempfile.TemporaryDirectory,
    maps_dict: Dict,
    coordinates: Dict,
    crs: pyproj.CRS,
    out_var_names: list,
):
    """Test that data values and coordinates are correctly
    preserved when reading from zarr with successive writes."""
    # Create first maps dict (use the fixture data)
    maps_dict_1 = maps_dict

    # Create second maps dict with different data
    key_list = list(maps_dict.keys())
    rng = np.random.default_rng(seed=42)  # Use seed for reproducible different data
    arr_shape = next(iter(maps_dict.values())).shape
    maps_dict_2 = {key: rng.random(size=arr_shape, dtype=np.float32) for key in key_list}

    # Write first timestep
    sim_time_1 = datetime(year=2023, month=1, day=1, hour=12)
    icechunk_provider.write_arrays(maps_dict_1, sim_time_1)

    # Write second timestep with different data
    sim_time_2 = datetime(year=2023, month=1, day=1, hour=13)
    icechunk_provider.write_arrays(maps_dict_2, sim_time_2)

    # Read the data back
    storage = icechunk.local_filesystem_storage(temp_dir.name)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store)

    # Assert that we have 2 timesteps
    assert ds.sizes["time"] == 2

    # Assert that all expected data variables are present
    expected_var_names = set(out_var_names)
    actual_var_names = set(ds.data_vars.keys())
    assert expected_var_names.issubset(actual_var_names)

    # Assert that spatial coordinates are preserved
    assert "x" in ds.coords
    assert "y" in ds.coords

    # Verify x coordinates
    expected_x = coordinates["x_coords"]
    actual_x = ds.coords["x"].values
    assert np.allclose(actual_x, expected_x)

    # Verify y coordinates
    expected_y = coordinates["y_coords"]
    actual_y = ds.coords["y"].values
    assert np.allclose(actual_y, expected_y)

    # Assert that data values are preserved for each variable at both timesteps
    for zarr_var_name in out_var_names:
        if zarr_var_name in ds.data_vars:
            # Check first timestep data
            original_data_1 = maps_dict_1[zarr_var_name]
            actual_data_1 = ds[zarr_var_name].isel(time=0).values
            assert np.allclose(actual_data_1, original_data_1), (
                f"First timestep data mismatch for {zarr_var_name}"
            )

            # Check second timestep data
            original_data_2 = maps_dict_2[zarr_var_name]
            actual_data_2 = ds[zarr_var_name].isel(time=1).values
            assert np.allclose(actual_data_2, original_data_2), (
                f"Second timestep data mismatch for {zarr_var_name}"
            )

            # Ensure the two timesteps have different data (they should not be identical)
            assert not np.allclose(actual_data_1, actual_data_2), (
                f"Timesteps should have different data for {zarr_var_name}"
            )

    # Assert that CRS information is preserved
    crs_actual = pyproj.CRS.from_wkt(ds.attrs["crs_wkt"])
    assert crs == crs_actual

    # Verify that timestamps are correct
    expected_times = [sim_time_1, sim_time_2]
    actual_times = [pd.to_datetime(t).to_pydatetime() for t in ds["time"].values]
    assert len(actual_times) == len(expected_times)
    for actual, expected in zip(actual_times, expected_times):
        assert actual == expected, f"Expected {expected}, got {actual}"


def test_non_matching_repo(
    temp_dir: tempfile.TemporaryDirectory,
    maps_dict: Dict,
    coordinates: Dict,
    crs: pyproj.CRS,
    out_var_names: list,
):
    """Test that writing arrays to an existing repository
    that does not match the new data fails."""

    # Create and write original arrays (6x9 from fixture)
    storage = icechunk.local_filesystem_storage(temp_dir.name)
    provider_config_1 = {
        "out_var_names": out_var_names,
        "crs": crs,
        "x_coords": coordinates["x_coords"],
        "y_coords": coordinates["y_coords"],
        "icechunk_storage": storage,
    }
    icechunk_p1 = IcechunkRasterOutputProvider()
    icechunk_p1.initialize(provider_config_1)

    # Write original data
    sim_time_1 = datetime(year=2023, month=1, day=1, hour=12)
    icechunk_p1.write_arrays(maps_dict, sim_time_1)

    # Create arrays with different dimensions (4x7)
    new_arr_shape = (4, 7)  # Different from original (6, 9)

    # Create new coordinates for the different dimensions
    new_y_coords = np.linspace(start=5000, stop=5000 + new_arr_shape[0], num=new_arr_shape[0])
    new_x_coords = np.linspace(start=5000, stop=5000 + new_arr_shape[1], num=new_arr_shape[1])
    new_coordinates = {"x_coords": new_x_coords, "y_coords": new_y_coords}

    # Create new provider with different dimensions but same storage
    provider_config_2 = {
        "out_var_names": out_var_names,
        "crs": crs,
        "x_coords": new_coordinates["x_coords"],
        "y_coords": new_coordinates["y_coords"],
        "icechunk_storage": storage,  # Same storage as before
    }
    # Non-matching coordinates should raise ValueError
    with pytest.raises(ValueError):
        icechunk_p2 = IcechunkRasterOutputProvider()
        icechunk_p2.initialize(provider_config_2)
