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
