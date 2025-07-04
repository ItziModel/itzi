# coding=utf8
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


def calculate_total_volume(depth_array: np.ndarray, cell_surface_area: float) -> float:
    """Calculates the total volume from a depth array.
    Args:
        depth_array: 2D array of water depths (m)
        cell_surface_area: Area of each grid cell (m²)
    Returns:
        Total water volume (m³)
    """
    return np.sum(depth_array) * cell_surface_area


def calculate_continuity_error(volume_error: float, volume_change: float) -> float:
    """Calculates the continuity error percentage.
    Args:
        volume_error: Difference between expected and actual volume (m³)
        volume_change: Total volume change during time step (m³)
    Returns:
        Continuity error as a percentage
    """
    if volume_change == 0:
        return 0.0
    return (volume_error / volume_change) * 100


# --- Output Array Calculations (Previously in Report) ---


def calculate_wse(h_array: np.ndarray, dem_array: np.ndarray) -> np.ndarray:
    """Calculates Water Surface Elevation.
    Args:
        h_array: 2D array of water depths (m)
        dem_array: 2D array of ground elevations (m)
    Returns:
        2D array of water surface elevations (m)
    """
    return h_array + dem_array


def calculate_flux(flow_array: np.ndarray, cell_size: float) -> np.ndarray:
    """Calculates the flux (Qx or Qy) by multiplying flow array by cell size.
    Args:
        flow_array: 2D array of flow values (m²/s)
        cell_size: Grid cell size in perpendicular direction (m)
    Returns:
        2D array of flux values (m³/s)
    """
    return flow_array * cell_size


# --- Statistical Calculations (Previously in Report) ---


def calculate_average_rate_from_total(
    total_volume_array: np.ndarray,
    interval_seconds: float,
    conversion_factor: float = 1.0,
) -> np.ndarray:
    """Calculates an average rate from a cumulated volume array.
    Args:
        total_volume_array: 2D array of total volumes (m)
        interval_seconds: Time interval over which volumes were accumulated (s)
        conversion_factor: Factor to convert rate units (default: 1.0 for m/s)
    Returns:
        2D array of average rates in converted units
    """
    return (total_volume_array / interval_seconds) * conversion_factor


def accumulate_rate_to_total(
    stat_array: np.ndarray, rate_array: np.ndarray, time_delta_seconds: float
) -> None:
    """Accumulates a rate array into a total array over a time delta.
       The operation is performed in-place on stat_array.
    Args:
        stat_array: 2D array to accumulate into (modified in-place)
        rate_array: 2D array of rates to accumulate
        time_delta_seconds: Time interval over which to accumulate (s)
    Returns:
        None (modifies stat_array in-place)
    """
    stat_array += rate_array * time_delta_seconds
