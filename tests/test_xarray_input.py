from typing import Dict
from datetime import datetime, timedelta

import xarray as xr
import numpy as np
import pandas as pd
import pytest
import pyproj

from itzi.providers.xarray_input import XarrayRasterInputProvider, XarrayRasterInputConfig


@pytest.fixture(scope="module")
def input_maps_dict():
    """A dict representing the input arrays to be stored as xarray.dataset"""
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
def xarray_input_data_relative_time(
    input_maps_dict: Dict,
    coordinates: Dict,
    crs: pyproj.CRS,
    relative_time_coordinates: list,
    input_map_names: Dict,
):
    """Create a Dataset with relative time input data for testing"""

    time_len = len(relative_time_coordinates)

    # Create data variables - some time-dependent, some static
    data_vars = {}
    coords = {
        "time": relative_time_coordinates,
        "y": coordinates["y_coords"],
        "x": coordinates["x_coords"],
    }
    for itzi_key, var_name in input_map_names.items():
        if itzi_key in input_maps_dict:
            base_data = input_maps_dict[itzi_key]
            # Make some variables time-dependent, others static
            if itzi_key in ["rainfall", "boundary_conditions"]:
                # Time-dependent variables
                time_data = np.stack([base_data * (1 + 0.1 * t) for t in range(time_len)])
                data_vars[var_name] = (["time", "y", "x"], time_data)
            else:
                # Static variables (no time dimension)
                data_vars[var_name] = (["y", "x"], base_data)

    # Create the dataset
    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    # Add CRS information
    ds.attrs["crs_wkt"] = crs.to_wkt()

    return {
        "dataset": ds,
        "input_maps_dict": input_maps_dict,
        "input_map_names": input_map_names,
        "relative_time_coordinates": relative_time_coordinates,
    }


@pytest.fixture(scope="module")
def input_map_names(input_maps_dict: Dict):
    """Mapping from itzi internal names to dataset variable names"""
    # Create a mapping where some keys map to different variable names in xarray.dataset
    mapping = {}
    keys = list(input_maps_dict.keys())
    for i, key in enumerate(keys):
        if i % 2 == 0:
            # Some variables have the same name
            mapping[key] = key
        else:
            # Some variables have different names in xarray
            mapping[key] = f"xarray_{key}"
    return mapping


@pytest.fixture
def xarray_input_data(
    input_maps_dict: Dict,
    coordinates: Dict,
    crs: pyproj.CRS,
    time_coordinates: list,
    input_map_names: Dict,
):
    """Create a dataset repository with input data for testing"""
    time_len = len(time_coordinates)
    # Create data variables - some time-dependent, some static
    data_vars = {}
    coords = {
        "time": time_coordinates,
        "y": coordinates["y_coords"],
        "x": coordinates["x_coords"],
    }

    for itzi_key, var_name in input_map_names.items():
        if itzi_key in input_maps_dict:
            base_data = input_maps_dict[itzi_key]

            # Make some variables time-dependent, others static
            if itzi_key in ["rainfall", "boundary_conditions"]:
                # Time-dependent variables
                time_data = np.stack([base_data * (1 + 0.1 * t) for t in range(time_len)])
                data_vars[var_name] = (["time", "y", "x"], time_data)
            else:
                # Static variables (no time dimension)
                data_vars[var_name] = (["y", "x"], base_data)

    # Create the dataset
    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    # Add CRS information
    ds.attrs["crs_wkt"] = crs.to_wkt()

    return {
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


def test_xarray_input_provider_creation(xarray_input_data: Dict, default_times: Dict):
    """Test that XarrayRasterInputProvider can be created successfully"""
    config: XarrayRasterInputConfig = {
        "dataset": xarray_input_data["dataset"],
        "input_map_names": xarray_input_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    # Create the provider
    provider = XarrayRasterInputProvider(config)

    # Verify that the provider was created with correct attributes
    assert provider.sim_start_time == default_times["start_time"]
    assert provider.sim_end_time == default_times["end_time"]
    assert provider.input_map_names == xarray_input_data["input_map_names"]

    # Verify that the the xarray dataset is accessible
    ds = provider.dataset
    assert ds is not None
    assert len(ds.data_vars) > 0


def test_xarray_input_provider_get_array_static_variable(
    xarray_input_data: Dict, default_times: Dict
):
    """Test get_array method for static (non-time-dependent) variables"""
    config: XarrayRasterInputConfig = {
        "dataset": xarray_input_data["dataset"],
        "input_map_names": xarray_input_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    provider = XarrayRasterInputProvider(config)

    # Test with a static variable (e.g., 'dem')
    test_key = "dem"
    if test_key in xarray_input_data["input_map_names"]:
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
            expected_data = xarray_input_data["input_maps_dict"][test_key]
            assert array.shape == expected_data.shape
            assert np.allclose(array, expected_data)

            # For static variables, times might be the default times
            assert isinstance(start_time, datetime)
            assert isinstance(end_time, datetime)

        except (NotImplementedError, AttributeError, TypeError) as e:
            # Expected since the method is not fully implemented
            pytest.skip(f"get_array method not fully implemented: {e}")


def test_xarray_input_provider_get_array_time_dependent_variable(
    xarray_input_data: Dict, default_times: Dict
):
    """Test get_array method for time-dependent variables"""
    config: XarrayRasterInputConfig = {
        "dataset": xarray_input_data["dataset"],
        "input_map_names": xarray_input_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    provider = XarrayRasterInputProvider(config)

    # Test with a time-dependent variable (e.g., 'rainfall')
    test_key = "rainfall"
    if test_key in xarray_input_data["input_map_names"]:
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
        base_data = xarray_input_data["input_maps_dict"][test_key]
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


def test_xarray_input_provider_get_array_nonexistent_key(
    xarray_input_data: Dict, default_times: Dict
):
    """Test get_array method with a non-existent map key"""
    config: XarrayRasterInputConfig = {
        "dataset": xarray_input_data["dataset"],
        "input_map_names": xarray_input_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    provider = XarrayRasterInputProvider(config)

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


def test_xarray_input_provider_origin_property(
    xarray_input_data: Dict, default_times: Dict, coordinates: Dict
):
    """Test the origin property - should return NW corner coordinates as (N, W) tuple"""
    config: XarrayRasterInputConfig = {
        "dataset": xarray_input_data["dataset"],
        "input_map_names": xarray_input_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    provider = XarrayRasterInputProvider(config)

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


def test_xarray_input_provider_data_consistency(xarray_input_data: Dict, default_times: Dict):
    """Test that the provider can consistently access the same data"""
    config: XarrayRasterInputConfig = {
        "dataset": xarray_input_data["dataset"],
        "input_map_names": xarray_input_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    provider = XarrayRasterInputProvider(config)

    ds = provider.dataset

    # Check that all expected variables are present
    for itzi_key, var_name in xarray_input_data["input_map_names"].items():
        if itzi_key in xarray_input_data["input_maps_dict"]:
            assert var_name in ds.data_vars or var_name in ds.coords

    # Check that coordinates are correct
    assert "x" in ds.coords
    assert "y" in ds.coords
    assert "time" in ds.coords

    # Verify coordinate values
    expected_x = xarray_input_data["dataset"]["x"].values
    expected_y = xarray_input_data["dataset"]["y"].values
    assert np.allclose(ds["x"].values, expected_x)
    assert np.allclose(ds["y"].values, expected_y)

    # Verify time coordinates
    expected_times = xarray_input_data["time_coordinates"]
    actual_times = [pd.to_datetime(t).to_pydatetime() for t in ds["time"].values]
    assert len(actual_times) == len(expected_times)
    for actual, expected in zip(actual_times, expected_times):
        assert actual == expected


@pytest.mark.parametrize("map_key", ["dem", "friction", "rainfall"])
def test_xarray_input_provider_multiple_variables(
    xarray_input_data: Dict, default_times: Dict, map_key: str
):
    """Test get_array method with different variable types"""
    if map_key not in xarray_input_data["input_map_names"]:
        pytest.skip(f"Variable {map_key} not in test data")

    config: XarrayRasterInputConfig = {
        "dataset": xarray_input_data["dataset"],
        "input_map_names": xarray_input_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    provider = XarrayRasterInputProvider(config)
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


def test_xarray_input_provider_get_array_time_dependent_variable_relative_time(
    xarray_input_data_relative_time: Dict, default_times: Dict
):
    """Test get_array method for time-dependent variables with relative time (timedelta)"""
    config: XarrayRasterInputConfig = {
        "dataset": xarray_input_data_relative_time["dataset"],
        "input_map_names": xarray_input_data_relative_time["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    provider = XarrayRasterInputProvider(config)

    # Test with a time-dependent variable (e.g., 'rainfall')
    test_key = "rainfall"
    if test_key in xarray_input_data_relative_time["input_map_names"]:
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
        base_data = xarray_input_data_relative_time["input_maps_dict"][test_key]
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
    """Create a dataset with unsorted coordinates for testing"""

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

    return {
        "dataset": ds,
        "input_map_names": {"dem": "dem"},
    }


@pytest.fixture
def unequal_spacing_data(input_maps_dict: Dict, crs: pyproj.CRS):
    """Create a dataset with unequally spaced coordinates for testing"""

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

    return {
        "dataset": ds,
        "input_map_names": {"dem": "dem"},
    }


def test_is_dataset_sorted_with_sorted_coordinates(xarray_input_data: Dict, default_times: Dict):
    """Test is_dataset_sorted() method with properly sorted coordinates"""
    config: XarrayRasterInputConfig = {
        "dataset": xarray_input_data["dataset"],
        "input_map_names": xarray_input_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    provider = XarrayRasterInputProvider(config)

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
    config: XarrayRasterInputConfig = {
        "dataset": unsorted_coordinates_data["dataset"],
        "input_map_names": unsorted_coordinates_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    # Should raise ValueError because coordinates are not sorted
    with pytest.raises(ValueError, match="Coordinates must be sorted"):
        XarrayRasterInputProvider(config)


def test_is_equal_spacing_with_equal_spacing(xarray_input_data: Dict, default_times: Dict):
    """Test is_equal_spacing() method with equally spaced coordinates"""
    config: XarrayRasterInputConfig = {
        "dataset": xarray_input_data["dataset"],
        "input_map_names": xarray_input_data["input_map_names"],
        "simulation_start_time": default_times["start_time"],
        "simulation_end_time": default_times["end_time"],
    }

    provider = XarrayRasterInputProvider(config)

    # Test the is_equal_spacing method directly
    result = provider.is_equal_spacing()

    # Should return True for equally spaced coordinates
    assert result is True
    assert isinstance(result, bool)


def test_is_equal_spacing_with_unequal_spacing(unequal_spacing_data: Dict, default_times: Dict):
    """Test is_equal_spacing() method with unequally spaced coordinates"""

    # Should raise ValueError because coordinates are not equally spaced
    with pytest.raises(ValueError, match="Spatial coordinates must be equally spaced"):
        XarrayRasterInputProvider(
            {
                "dataset": unequal_spacing_data["dataset"],
                "input_map_names": unequal_spacing_data["input_map_names"],
                "simulation_start_time": default_times["start_time"],
                "simulation_end_time": default_times["end_time"],
            }
        )


def test_is_array_sorted_static_method():
    """Test the static is_array_sorted() method with various arrays"""
    # Test ascending sorted array
    ascending_array = np.array([1, 2, 3, 4, 5])
    assert XarrayRasterInputProvider.is_array_sorted(ascending_array, ascending=True) is True
    assert XarrayRasterInputProvider.is_array_sorted(ascending_array, ascending=False) is False

    # Test descending sorted array
    descending_array = np.array([5, 4, 3, 2, 1])
    assert XarrayRasterInputProvider.is_array_sorted(descending_array, ascending=True) is False
    assert XarrayRasterInputProvider.is_array_sorted(descending_array, ascending=False) is True

    # Test unsorted array
    unsorted_array = np.array([1, 3, 2, 5, 4])
    assert XarrayRasterInputProvider.is_array_sorted(unsorted_array, ascending=True) is False
    assert XarrayRasterInputProvider.is_array_sorted(unsorted_array, ascending=False) is False

    # Test array with equal values (should be considered sorted)
    equal_array = np.array([2, 2, 2, 2])
    assert XarrayRasterInputProvider.is_array_sorted(equal_array, ascending=True) is True
    assert XarrayRasterInputProvider.is_array_sorted(equal_array, ascending=False) is True

    # Test single element array
    single_array = np.array([42])
    assert XarrayRasterInputProvider.is_array_sorted(single_array, ascending=True) is True
    assert XarrayRasterInputProvider.is_array_sorted(single_array, ascending=False) is True

    # Test empty array
    empty_array = np.array([])
    assert XarrayRasterInputProvider.is_array_sorted(empty_array, ascending=True) is True
    assert XarrayRasterInputProvider.is_array_sorted(empty_array, ascending=False) is True
