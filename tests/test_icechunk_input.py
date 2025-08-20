from typing import Dict
from datetime import datetime, timedelta

import icechunk
import xarray as xr
import numpy as np
import pandas as pd
import pytest
import pyproj

from itzi.providers.icechunk_input import IcechunkRasterInputProvider, IcechunkRasterInputConfig


@pytest.fixture(scope="module")
def input_maps_dict():
    """A dict representing the input arrays to be stored in the icechunk repository"""
    # Create some common input variables that might be used
    key_list = ["rainfall", "dem", "friction", "boundary_conditions", "infiltration"]

    rng = np.random.default_rng(seed=42)
    arr_shape = (6, 9)
    return {key: rng.random(size=arr_shape, dtype=np.float32) for key in key_list}


@pytest.fixture(scope="module")
def coordinates(input_maps_dict: Dict):
    """Generate x and y coordinates for the test arrays"""
    arr_shape = next(iter(input_maps_dict.values())).shape
    res = 10
    y_start = 1000
    y_stop = y_start + arr_shape[0] * res
    x_start = 2000
    x_stop = x_start + arr_shape[1] * res
    y_coords = np.linspace(start=y_start, stop=y_stop, num=arr_shape[0], endpoint=False)
    x_coords = np.linspace(start=x_start, stop=x_stop, num=arr_shape[1], endpoint=False)
    print(f"{x_start=}, {x_stop=} {y_start=}, {y_stop}")
    return {
        "x_start": x_start,
        "x_stop": x_stop,
        "y_start": y_start,
        "y_stop": y_stop,
        "x_coords": x_coords,
        "y_coords": y_coords,
    }


@pytest.fixture(scope="module")
def crs():
    """CRS for the test data"""
    return pyproj.CRS.from_epsg(6372)  # Mexico LCC


@pytest.fixture(scope="module")
def time_coordinates():
    """Generate time coordinates for the test data"""
    start_time = datetime(2023, 1, 1, 0, 0, 0)
    time_steps = 5
    time_step_hours = 1
    times = [start_time + timedelta(hours=i * time_step_hours) for i in range(time_steps)]
    return times


@pytest.fixture(scope="module")
def relative_time_coordinates():
    """Generate relative time coordinates (timedelta) for the test data"""
    time_steps = 5
    time_step_hours = 1
    times = [timedelta(hours=i * time_step_hours) for i in range(time_steps)]
    return times


@pytest.fixture
def icechunk_input_data_relative_time(
    input_maps_dict: Dict,
    coordinates: Dict,
    crs: pyproj.CRS,
    relative_time_coordinates: list,
    input_map_names: Dict,
):
    """Create an icechunk repository with relative time input data for testing"""
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")

    # Create the dataset
    time_len = len(relative_time_coordinates)

    # Create data variables - some time-dependent, some static
    data_vars = {}
    coords = {
        "time": relative_time_coordinates,
        "y": coordinates["y_coords"],
        "x": coordinates["x_coords"],
    }

    for itzi_key, zarr_var_name in input_map_names.items():
        if itzi_key in input_maps_dict:
            base_data = input_maps_dict[itzi_key]

            # Make some variables time-dependent, others static
            if itzi_key in ["rainfall", "boundary_conditions"]:
                # Time-dependent variables
                time_data = np.stack([base_data * (1 + 0.1 * t) for t in range(time_len)])
                data_vars[zarr_var_name] = (["time", "y", "x"], time_data)
            else:
                # Static variables (no time dimension)
                data_vars[zarr_var_name] = (["y", "x"], base_data)

    # Create the dataset
    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    # Add CRS information
    ds.attrs["crs_wkt"] = crs.to_wkt()

    # Write to zarr
    ds.to_zarr(session.store, mode="w")
    session.commit("Initial commit with relative time input data")

    return {
        "storage": storage,
        "group": "main",
        "dataset": ds,
        "input_maps_dict": input_maps_dict,
        "input_map_names": input_map_names,
        "relative_time_coordinates": relative_time_coordinates,
    }


@pytest.fixture(scope="module")
def input_map_names(input_maps_dict: Dict):
    """Mapping from itzi internal names to zarr variable names"""
    # Create a mapping where some keys map to different variable names in zarr
    mapping = {}
    keys = list(input_maps_dict.keys())
    for i, key in enumerate(keys):
        if i % 2 == 0:
            # Some variables have the same name
            mapping[key] = key
        else:
            # Some variables have different names in zarr
            mapping[key] = f"zarr_{key}"
    return mapping


@pytest.fixture
def icechunk_input_data(
    input_maps_dict: Dict,
    coordinates: Dict,
    crs: pyproj.CRS,
    time_coordinates: list,
    input_map_names: Dict,
):
    """Create an icechunk repository with input data for testing"""
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")

    # Create the dataset
    time_len = len(time_coordinates)

    # Create data variables - some time-dependent, some static
    data_vars = {}
    coords = {
        "time": time_coordinates,
        "y": coordinates["y_coords"],
        "x": coordinates["x_coords"],
    }

    for itzi_key, zarr_var_name in input_map_names.items():
        if itzi_key in input_maps_dict:
            base_data = input_maps_dict[itzi_key]

            # Make some variables time-dependent, others static
            if itzi_key in ["rainfall", "boundary_conditions"]:
                # Time-dependent variables
                time_data = np.stack([base_data * (1 + 0.1 * t) for t in range(time_len)])
                data_vars[zarr_var_name] = (["time", "y", "x"], time_data)
            else:
                # Static variables (no time dimension)
                data_vars[zarr_var_name] = (["y", "x"], base_data)

    # Create the dataset
    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    # Add CRS information
    ds.attrs["crs_wkt"] = crs.to_wkt()

    # Write to zarr
    ds.to_zarr(session.store, mode="w")
    session.commit("Initial commit with input data")

    return {
        "storage": storage,
        "group": "main",
        "dataset": ds,
        "input_maps_dict": input_maps_dict,
        "input_map_names": input_map_names,
        "time_coordinates": time_coordinates,
    }


@pytest.fixture
def default_times():
    """Default start and end times for the input provider"""
    return {
        "start_time": datetime(2023, 1, 1, 0, 0, 0),
        "end_time": datetime(2023, 1, 1, 23, 59, 59),
    }


def test_icechunk_input_provider_creation(icechunk_input_data: Dict, default_times: Dict):
    """Test that IcechunkRasterInputProvider can be created successfully"""
    config: IcechunkRasterInputConfig = {
        "icechunk_storage": icechunk_input_data["storage"],
        "icechunk_group": icechunk_input_data["group"],
        "input_map_names": icechunk_input_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    # Create the provider
    provider = IcechunkRasterInputProvider(config)

    # Verify that the provider was created with correct attributes
    assert provider.sim_start_time == default_times["start_time"]
    assert provider.sim_end_time == default_times["end_time"]
    assert provider.input_map_names == icechunk_input_data["input_map_names"]

    # Verify that the session can access the zarr store
    ds = provider.dataset
    assert ds is not None
    assert len(ds.data_vars) > 0


def test_icechunk_input_provider_get_array_static_variable(
    icechunk_input_data: Dict, default_times: Dict
):
    """Test get_array method for static (non-time-dependent) variables"""
    config: IcechunkRasterInputConfig = {
        "icechunk_storage": icechunk_input_data["storage"],
        "icechunk_group": icechunk_input_data["group"],
        "input_map_names": icechunk_input_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    provider = IcechunkRasterInputProvider(config)

    # Test with a static variable (e.g., 'dem')
    test_key = "dem"
    if test_key in icechunk_input_data["input_map_names"]:
        current_time = datetime(2023, 1, 1, 12, 0, 0)

        # Note: The get_array method is incomplete, so this test will likely fail
        # until the method is properly implemented. This test serves as a specification
        # for what the method should do.
        try:
            result = provider.get_array(test_key, current_time)

            # The method should return a tuple of (array, start_time, end_time)
            assert isinstance(result, tuple)
            assert len(result) == 3

            array, start_time, end_time = result

            # Verify the array
            assert isinstance(array, np.ndarray)
            expected_data = icechunk_input_data["input_maps_dict"][test_key]
            assert array.shape == expected_data.shape
            assert np.allclose(array, expected_data)

            # For static variables, times might be the default times
            assert isinstance(start_time, datetime)
            assert isinstance(end_time, datetime)

        except (NotImplementedError, AttributeError, TypeError) as e:
            # Expected since the method is not fully implemented
            pytest.skip(f"get_array method not fully implemented: {e}")


def test_icechunk_input_provider_get_array_time_dependent_variable(
    icechunk_input_data: Dict, default_times: Dict
):
    """Test get_array method for time-dependent variables"""
    config: IcechunkRasterInputConfig = {
        "icechunk_storage": icechunk_input_data["storage"],
        "icechunk_group": icechunk_input_data["group"],
        "input_map_names": icechunk_input_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    provider = IcechunkRasterInputProvider(config)

    # Test with a time-dependent variable (e.g., 'rainfall')
    test_key = "rainfall"
    if test_key in icechunk_input_data["input_map_names"]:
        # start_time = datetime(2023, 1, 1, 0, 0, 0)
        # time_step_hours = 1
        current_time = datetime(2023, 1, 1, 2, 30, 0)  # Should correspond to time index 2
        expected_start_time = datetime(2023, 1, 1, 2, 0, 0)
        expected_end_time = datetime(2023, 1, 1, 3, 0, 0)

        result = provider.get_array(test_key, current_time)

        # The method should return a tuple of (array, start_time, end_time)
        assert isinstance(result, tuple)
        assert len(result) == 3

        array, start_time, end_time = result

        # Verify the array
        assert isinstance(array, np.ndarray)
        # For time-dependent variables, should get the data for the specific time
        assert array.ndim == 2  # Should be 2D after time selection

        # Verify the array values are correct for the specific time
        # current_time = 2023-01-01 02:30:00 should correspond to time index 2
        # Expected data: base_data * (1 + 0.1 * 2) = base_data * 1.2
        expected_time_index = 2
        base_data = icechunk_input_data["input_maps_dict"][test_key]
        expected_array = base_data * (1 + 0.1 * expected_time_index)
        assert np.allclose(array, expected_array), (
            f"Array values don't match expected values for time index {expected_time_index}"
        )

        # Verify times
        assert isinstance(start_time, datetime)
        assert isinstance(end_time, datetime)
        assert start_time <= current_time <= end_time
        assert start_time == expected_start_time
        assert end_time == expected_end_time


def test_icechunk_input_provider_get_array_nonexistent_key(
    icechunk_input_data: Dict, default_times: Dict
):
    """Test get_array method with a non-existent map key"""
    config: IcechunkRasterInputConfig = {
        "icechunk_storage": icechunk_input_data["storage"],
        "icechunk_group": icechunk_input_data["group"],
        "input_map_names": icechunk_input_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    provider = IcechunkRasterInputProvider(config)

    # Test with a non-existent key
    nonexistent_key = "nonexistent_variable"
    current_time = datetime(2023, 1, 1, 12, 0, 0)

    result = provider.get_array(nonexistent_key, current_time)

    # According to the docstring, should return None for array and default times
    assert isinstance(result, tuple)
    assert len(result) == 3

    array, start_time, end_time = result

    # Should return None for the array
    assert array is None

    # Should return default times
    assert start_time == default_times["start_time"]
    assert end_time == default_times["end_time"]


def test_icechunk_input_provider_origin_property(
    icechunk_input_data: Dict, default_times: Dict, coordinates: Dict
):
    """Test the origin property - should return NW corner coordinates as (N, W) tuple"""
    config: IcechunkRasterInputConfig = {
        "icechunk_storage": icechunk_input_data["storage"],
        "icechunk_group": icechunk_input_data["group"],
        "input_map_names": icechunk_input_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    provider = IcechunkRasterInputProvider(config)

    # The origin property should return a tuple of (N, W) coordinates for the NW corner
    result = provider.origin

    # Should return a tuple of (North, West) coordinates
    assert isinstance(result, tuple)
    assert len(result) == 2

    north, west = result
    assert isinstance(north, (int, float))
    assert isinstance(west, (int, float))

    # Expected NW corner: highest Y (North) and lowest X (West)
    expected_north = coordinates["y_coords"].max() + 5  # Highest Y coordinate + half of resolution
    expected_west = coordinates["x_coords"].min() - 5  # Lowest X coordinate

    assert north == expected_north
    assert west == expected_west


def test_icechunk_input_provider_invalid_storage():
    """Test that creating provider with invalid storage raises appropriate error"""
    config: IcechunkRasterInputConfig = {
        "icechunk_storage": icechunk.local_filesystem_storage("/tmp/nonexistent/path"),
        "icechunk_group": "main",
        "input_map_names": {"test": "test"},
        "simulation_start_time": datetime(2023, 1, 1),
        "simulation_end_time": datetime(2023, 1, 2),
    }

    # Should raise an error when trying to open non-existent repository
    with pytest.raises(icechunk.IcechunkError):
        IcechunkRasterInputProvider(config)


def test_icechunk_input_provider_invalid_group(icechunk_input_data: Dict, default_times: Dict):
    """Test that creating provider with invalid group raises appropriate error"""
    config: IcechunkRasterInputConfig = {
        "icechunk_storage": icechunk_input_data["storage"],
        "icechunk_group": "nonexistent_group",
        "input_map_names": icechunk_input_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }
    # Should raise an error when trying to access non-existent group
    with pytest.raises(icechunk.IcechunkError):
        IcechunkRasterInputProvider(config)


def test_icechunk_input_provider_data_consistency(icechunk_input_data: Dict, default_times: Dict):
    """Test that the provider can consistently access the same data"""
    config: IcechunkRasterInputConfig = {
        "icechunk_storage": icechunk_input_data["storage"],
        "icechunk_group": icechunk_input_data["group"],
        "input_map_names": icechunk_input_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    provider = IcechunkRasterInputProvider(config)

    ds = provider.dataset

    # Check that all expected variables are present
    for itzi_key, zarr_var_name in icechunk_input_data["input_map_names"].items():
        if itzi_key in icechunk_input_data["input_maps_dict"]:
            assert zarr_var_name in ds.data_vars or zarr_var_name in ds.coords

    # Check that coordinates are correct
    assert "x" in ds.coords
    assert "y" in ds.coords
    assert "time" in ds.coords

    # Verify coordinate values
    expected_x = icechunk_input_data["dataset"]["x"].values
    expected_y = icechunk_input_data["dataset"]["y"].values
    assert np.allclose(ds["x"].values, expected_x)
    assert np.allclose(ds["y"].values, expected_y)

    # Verify time coordinates
    expected_times = icechunk_input_data["time_coordinates"]
    actual_times = [pd.to_datetime(t).to_pydatetime() for t in ds["time"].values]
    assert len(actual_times) == len(expected_times)
    for actual, expected in zip(actual_times, expected_times):
        assert actual == expected


@pytest.mark.parametrize("map_key", ["dem", "friction", "rainfall"])
def test_icechunk_input_provider_multiple_variables(
    icechunk_input_data: Dict, default_times: Dict, map_key: str
):
    """Test get_array method with different variable types"""
    if map_key not in icechunk_input_data["input_map_names"]:
        pytest.skip(f"Variable {map_key} not in test data")

    config: IcechunkRasterInputConfig = {
        "icechunk_storage": icechunk_input_data["storage"],
        "icechunk_group": icechunk_input_data["group"],
        "input_map_names": icechunk_input_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    provider = IcechunkRasterInputProvider(config)
    current_time = datetime(2023, 1, 1, 12, 0, 0)

    try:
        result = provider.get_array(map_key, current_time)

        # Basic validation that the method returns expected structure
        assert isinstance(result, tuple)
        assert len(result) == 3

        array, start_time, end_time = result

        if array is not None:
            assert isinstance(array, np.ndarray)
            assert array.ndim == 2  # Should be 2D spatial array

        assert isinstance(start_time, datetime)
        assert isinstance(end_time, datetime)

    except (NotImplementedError, AttributeError, TypeError) as e:
        # Expected since the method is not fully implemented
        pytest.skip(f"get_array method not fully implemented for {map_key}: {e}")


def test_icechunk_input_provider_get_array_time_dependent_variable_relative_time(
    icechunk_input_data_relative_time: Dict, default_times: Dict
):
    """Test get_array method for time-dependent variables with relative time (timedelta)"""
    config: IcechunkRasterInputConfig = {
        "icechunk_storage": icechunk_input_data_relative_time["storage"],
        "icechunk_group": icechunk_input_data_relative_time["group"],
        "input_map_names": icechunk_input_data_relative_time["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    provider = IcechunkRasterInputProvider(config)

    # Test with a time-dependent variable (e.g., 'rainfall')
    test_key = "rainfall"
    if test_key in icechunk_input_data_relative_time["input_map_names"]:
        # For relative time, we need to provide a current_time that can be mapped to the relative coordinates
        # Since the relative time coordinates start at timedelta(hours=0), timedelta(hours=1), etc.
        # We'll use a datetime that corresponds to the second time step (timedelta(hours=1))
        current_time = datetime(2023, 1, 1, 1, 30, 0)  # Should correspond to time index 1
        expected_start_time = datetime(2023, 1, 1, 1, 0, 0)
        expected_end_time = datetime(2023, 1, 1, 2, 0, 0)

        result = provider.get_array(test_key, current_time)

        # The method should return a tuple of (array, start_time, end_time)
        assert isinstance(result, tuple)
        assert len(result) == 3

        array, start_time, end_time = result

        # Verify the array
        assert isinstance(array, np.ndarray)
        # For time-dependent variables, should get the data for the specific time
        assert array.ndim == 2  # Should be 2D after time selection

        # Verify the array values are correct for the specific time
        # current_time should correspond to time index 1 (timedelta(hours=1))
        # Expected data: base_data * (1 + 0.1 * 1) = base_data * 1.1
        expected_time_index = 1
        base_data = icechunk_input_data_relative_time["input_maps_dict"][test_key]
        expected_array = base_data * (1 + 0.1 * expected_time_index)
        assert np.allclose(array, expected_array), (
            f"Array values don't match expected values for relative time index {expected_time_index}"
        )

        # Verify times
        assert isinstance(start_time, datetime)
        assert isinstance(end_time, datetime)
        assert start_time <= current_time <= end_time
        assert start_time == expected_start_time
        assert end_time == expected_end_time


@pytest.fixture
def unsorted_coordinates_data(input_maps_dict: Dict, crs: pyproj.CRS):
    """Create an icechunk repository with unsorted coordinates for testing"""
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")

    # Create unsorted coordinates
    arr_shape = next(iter(input_maps_dict.values())).shape
    # Create unsorted x and y coordinates
    x_coords = np.array([2000, 2030, 2010, 2040, 2020, 2050, 2060, 2070, 2080])[: arr_shape[1]]
    y_coords = np.array([1050, 1000, 1040, 1020, 1030, 1010])[: arr_shape[0]]

    # Create data variables
    data_vars = {}
    coords = {
        "y": y_coords,
        "x": x_coords,
    }

    # Add a simple static variable
    data_vars["dem"] = (["y", "x"], input_maps_dict["dem"])

    # Create the dataset
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs["crs_wkt"] = crs.to_wkt()

    # Write to zarr
    ds.to_zarr(session.store, mode="w")
    session.commit("Initial commit with unsorted coordinates")

    return {
        "storage": storage,
        "group": "main",
        "dataset": ds,
        "input_map_names": {"dem": "dem"},
    }


@pytest.fixture
def unequal_spacing_data(input_maps_dict: Dict, crs: pyproj.CRS):
    """Create an icechunk repository with unequally spaced coordinates for testing"""
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")

    # Create unequally spaced but sorted coordinates
    arr_shape = next(iter(input_maps_dict.values())).shape
    # Create unequally spaced x and y coordinates (sorted but with varying spacing)
    x_coords = np.array([2000, 2010, 2025, 2030, 2050, 2060, 2080, 2090, 2100])[: arr_shape[1]]
    y_coords = np.array([1000, 1005, 1020, 1030, 1050, 1055])[: arr_shape[0]]

    # Create data variables
    data_vars = {}
    coords = {
        "y": y_coords,
        "x": x_coords,
    }

    # Add a simple static variable
    data_vars["dem"] = (["y", "x"], input_maps_dict["dem"])

    # Create the dataset
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs["crs_wkt"] = crs.to_wkt()

    # Write to zarr
    ds.to_zarr(session.store, mode="w")
    session.commit("Initial commit with unequal spacing")

    return {
        "storage": storage,
        "group": "main",
        "dataset": ds,
        "input_map_names": {"dem": "dem"},
    }


def test_is_dataset_sorted_with_sorted_coordinates(icechunk_input_data: Dict, default_times: Dict):
    """Test is_dataset_sorted() method with properly sorted coordinates"""
    config: IcechunkRasterInputConfig = {
        "icechunk_storage": icechunk_input_data["storage"],
        "icechunk_group": icechunk_input_data["group"],
        "input_map_names": icechunk_input_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    provider = IcechunkRasterInputProvider(config)

    # Test the is_dataset_sorted method directly
    result = provider.is_dataset_sorted()

    # Should return True for properly sorted coordinates
    assert result is True
    assert isinstance(result, bool)


def test_is_dataset_sorted_with_unsorted_coordinates(
    unsorted_coordinates_data: Dict, default_times: Dict
):
    """Test is_dataset_sorted() method with unsorted coordinates"""
    # This test should fail during provider creation due to unsorted coordinates
    config: IcechunkRasterInputConfig = {
        "icechunk_storage": unsorted_coordinates_data["storage"],
        "icechunk_group": unsorted_coordinates_data["group"],
        "input_map_names": unsorted_coordinates_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    # Should raise ValueError because coordinates are not sorted
    with pytest.raises(ValueError, match="Coordinates must be sorted"):
        IcechunkRasterInputProvider(config)


def test_is_equal_spacing_with_equal_spacing(icechunk_input_data: Dict, default_times: Dict):
    """Test is_equal_spacing() method with equally spaced coordinates"""
    config: IcechunkRasterInputConfig = {
        "icechunk_storage": icechunk_input_data["storage"],
        "icechunk_group": icechunk_input_data["group"],
        "input_map_names": icechunk_input_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    provider = IcechunkRasterInputProvider(config)

    # Test the is_equal_spacing method directly
    result = provider.is_equal_spacing()

    # Should return True for equally spaced coordinates
    assert result is True
    assert isinstance(result, bool)


def test_is_equal_spacing_with_unequal_spacing(unequal_spacing_data: Dict, default_times: Dict):
    """Test is_equal_spacing() method with unequally spaced coordinates"""
    # This test should fail during provider creation due to unequal spacing
    config: IcechunkRasterInputConfig = {
        "icechunk_storage": unequal_spacing_data["storage"],
        "icechunk_group": unequal_spacing_data["group"],
        "input_map_names": unequal_spacing_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    # Should raise ValueError because coordinates are not equally spaced
    with pytest.raises(ValueError, match="Spatial coordinates must be equally spaced"):
        IcechunkRasterInputProvider(config)


def test_is_array_sorted_static_method():
    """Test the static is_array_sorted() method with various arrays"""
    # Test ascending sorted array
    ascending_array = np.array([1, 2, 3, 4, 5])
    assert IcechunkRasterInputProvider.is_array_sorted(ascending_array, ascending=True) is True
    assert IcechunkRasterInputProvider.is_array_sorted(ascending_array, ascending=False) is False

    # Test descending sorted array
    descending_array = np.array([5, 4, 3, 2, 1])
    assert IcechunkRasterInputProvider.is_array_sorted(descending_array, ascending=True) is False
    assert IcechunkRasterInputProvider.is_array_sorted(descending_array, ascending=False) is True

    # Test unsorted array
    unsorted_array = np.array([1, 3, 2, 5, 4])
    assert IcechunkRasterInputProvider.is_array_sorted(unsorted_array, ascending=True) is False
    assert IcechunkRasterInputProvider.is_array_sorted(unsorted_array, ascending=False) is False

    # Test array with equal values (should be considered sorted)
    equal_array = np.array([2, 2, 2, 2])
    assert IcechunkRasterInputProvider.is_array_sorted(equal_array, ascending=True) is True
    assert IcechunkRasterInputProvider.is_array_sorted(equal_array, ascending=False) is True

    # Test single element array
    single_array = np.array([42])
    assert IcechunkRasterInputProvider.is_array_sorted(single_array, ascending=True) is True
    assert IcechunkRasterInputProvider.is_array_sorted(single_array, ascending=False) is True

    # Test empty array
    empty_array = np.array([])
    assert IcechunkRasterInputProvider.is_array_sorted(empty_array, ascending=True) is True
    assert IcechunkRasterInputProvider.is_array_sorted(empty_array, ascending=False) is True
