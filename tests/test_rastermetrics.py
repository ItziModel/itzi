import numpy as np

from src.itzi import rastermetrics

def test_calculate_total_volume():
    """Test calculate_total_volume with known inputs."""
    # Create a test depth array (3x3 grid)
    depth_array = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])
    cell_surface_area = 10.0  # m²
    
    # Calculate expected result manually
    total_depth = np.sum(depth_array)
    expected_volume = total_depth * cell_surface_area
    
    # Call the function and assert result
    result = rastermetrics.calculate_total_volume(depth_array, cell_surface_area)
    assert np.isclose(result, expected_volume)


def test_calculate_continuity_error():
    """Test calculate_continuity_error with various inputs."""
    # Test case 1: Positive volume error and change
    volume_error = 0.5
    volume_change = 10.0
    expected_error = (volume_error / volume_change) * 100
    result = rastermetrics.calculate_continuity_error(volume_error, volume_change)
    assert np.isclose(result, expected_error)

    # Test case 2: Zero volume change (should return 0)
    result = rastermetrics.calculate_continuity_error(volume_error, 0.0)
    assert np.isclose(result, 0.0)

    # Test case 3: Negative volume error
    result = rastermetrics.calculate_continuity_error(-0.5, volume_change)
    assert np.isclose(result, -expected_error)


def test_calculate_wse():
    """Test calculate_wse with known inputs."""
    # Create test arrays (3x3 grid)
    h_array = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])
    dem_array = np.array([
        [10.0, 10.1, 10.2],
        [10.3, 10.4, 10.5],
        [10.6, 10.7, 10.8]
    ])

    # Calculate expected result manually
    expected_wse = h_array + dem_array

    # Call the function and assert result
    result = rastermetrics.calculate_wse(h_array, dem_array)
    assert np.allclose(result, expected_wse)


def test_calculate_flux():
    """Test calculate_flux with known inputs."""
    # Create test array (3x3 grid)
    flow_array = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])
    cell_size = 10.0  # m

    # Calculate expected result manually
    expected_flux = flow_array * cell_size

    # Call the function and assert result
    result = rastermetrics.calculate_flux(flow_array, cell_size)
    assert np.allclose(result, expected_flux)


def test_calculate_average_rate_from_total():
    """Test calculate_average_rate_from_total with various inputs."""
    # Create test array (3x3 grid)
    total_volume_array = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])
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
