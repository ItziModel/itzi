"""
Copyright (C) 2025 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

import numpy as np

from itzi import rastermetrics


def test_calculate_total_volume():
    """Test calculate_total_volume with known inputs."""
    # Create a test depth array (3x3 grid)
    depth_array = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    cell_surface_area = 10.0  # m²

    # Calculate expected result manually
    total_depth = np.sum(depth_array)
    expected_volume = total_depth * cell_surface_area

    # Call the function and assert result
    result = rastermetrics.calculate_total_volume(depth_array, cell_surface_area)
    assert np.isclose(result, expected_volume)


def test_calculate_wse():
    """Test calculate_wse with known inputs."""
    # Create test arrays (3x3 grid)
    h_array = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    dem_array = np.array(
        [[10.0, 10.1, 10.2], [10.3, 10.4, 10.5], [10.6, 10.7, 10.8]], dtype=np.float32
    )

    # Calculate expected result manually
    expected_wse = h_array + dem_array

    # Call the function and assert result
    result = rastermetrics.calculate_wse(h_array, dem_array)
    assert np.allclose(result, expected_wse)


def test_calculate_flux():
    """Test calculate_flux with known inputs."""
    # Create test array (3x3 grid)
    flow_array = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    cell_size = 10.0  # m

    # Calculate expected result manually
    expected_flux = flow_array * cell_size

    # Call the function and assert result
    result = rastermetrics.calculate_flux(flow_array, cell_size)
    assert np.allclose(result, expected_flux)


def test_calculate_average_rate_from_total():
    """Test calculate_average_rate_from_total with various inputs."""
    # Create test array (3x3 grid)
    total_volume_array = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32
    )
    interval_seconds = 60.0  # 1 minute

    # Test case 1: No conversion (conversion_factor = 1.0)
    expected_rate1 = total_volume_array / interval_seconds
    result1 = rastermetrics.calculate_average_rate_from_total(
        total_volume_array, interval_seconds, 1.0
    )
    assert np.allclose(result1, expected_rate1)

    # Test case 2: Conversion from m/s to mm/h (factor = 1000 * 3600)
    conversion_factor = 1000 * 3600  # Convert to mm/h
    expected_rate2 = (total_volume_array / interval_seconds) * conversion_factor
    result2 = rastermetrics.calculate_average_rate_from_total(
        total_volume_array, interval_seconds, conversion_factor
    )
    assert np.allclose(result2, expected_rate2)


def test_accumulate_rate_to_total():
    """Test accumulate_rate_to_total with various inputs."""
    # Create test arrays (3x3 grid)
    accum_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    rate_array = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    time_delta_seconds = 60.0  # 1 minute

    # Store original accum_array for comparison (shallow copy is sufficient for numeric arrays)
    original_accum_array = accum_array.copy()

    # Calculate expected result manually
    expected_accumulation = rate_array * time_delta_seconds
    expected_result = original_accum_array + expected_accumulation

    # Call the function (should modify accum_array in-place)
    rastermetrics.accumulate_rate_to_total(
        accum_array, rate_array, time_delta_seconds, padded=False
    )

    # Assert that accum_array was modified in-place to the expected result
    assert np.allclose(accum_array, expected_result)

    # Make sure the original array has not changed
    assert not np.allclose(accum_array, original_accum_array)

    # Test case 2: Zero time delta
    accum_array2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rate_array2 = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    original_accum_array2 = accum_array2.copy()

    # With zero time delta, accum_array should remain unchanged
    rastermetrics.accumulate_rate_to_total(accum_array2, rate_array2, 0.0)
    assert np.allclose(accum_array2, original_accum_array2)
