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

cimport cython
from cython.parallel cimport prange
from libc.math cimport NAN

ctypedef cython.floating DTYPE_t

import numpy as np

DTYPE = np.float32

# Threshold for serial computation
cdef int small_array_size = 1_000_000


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef DTYPE_t arr_sum(DTYPE_t[:, ::1] arr, bint padded=False):
    """Return the sum of an array using parallel reduction.
    """
    cdef int row_start, row_end, col_start, col_end, row_idx, col_idx
    cdef DTYPE_t asum = 0.
    cdef int total_elements

    if padded:
        row_start = 1
        row_end = arr.shape[0] - 1
        col_start = 1
        col_end = arr.shape[1] - 1
    else:
        row_start = 0
        row_end = arr.shape[0]
        col_start = 0
        col_end = arr.shape[1]

    total_elements = (row_end - row_start) * (col_end - col_start)

    # For small arrays, overhead of parallel processing is not worth it
    if total_elements <= small_array_size:
        for row_idx in range(row_start, row_end):
            for col_idx in range(col_start, col_end):
                asum += arr[row_idx, col_idx]
    else:
        for row_idx in prange(row_start, row_end, nogil=True):
            for col_idx in range(col_start, col_end):
                asum += arr[row_idx, col_idx]
    return asum


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def set_ext_array(
    DTYPE_t[:, :] arr_qext,
    DTYPE_t[:, :] arr_drain,
    DTYPE_t[:, :] arr_eff_precip,
    DTYPE_t[:, :] arr_ext,
):
    """Calculate the new ext_array to be used in depth update
    """
    cdef int rows, cols, row_idx, col_idx
    cdef DTYPE_t qext, rain, inf, qdrain
    rows = arr_qext.shape[0]
    cols = arr_qext.shape[1]
    cdef int total_elements = rows* cols

    # For small arrays, overhead of parallel processing is not worth it
    if total_elements <= small_array_size:
        for row_idx in range(rows):
            for col_idx in range(cols):
                arr_ext[row_idx, col_idx] = arr_qext[row_idx, col_idx] + arr_drain[row_idx, col_idx] + arr_eff_precip[row_idx, col_idx]
    else:
        for row_idx in prange(rows, nogil=True):
            for col_idx in range(cols):
                arr_ext[row_idx, col_idx] = arr_qext[row_idx, col_idx] + arr_drain[row_idx, col_idx] + arr_eff_precip[row_idx, col_idx]


def calculate_total_volume(
    DTYPE_t[:, ::1] depth_array, DTYPE_t cell_surface_area, bint padded=False
) -> DTYPE_t:
    """Calculates the total volume from a depth array.
    Args:
        depth_array: 2D array of water depths (m)
        cell_surface_area: Area of each grid cell (m²)
        padded: boolean, if True, will process array as padded
    Returns:
        Total water volume (m³)
    """
    total_sum = arr_sum(depth_array, padded)
    return total_sum * cell_surface_area


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def calculate_h_from_wse(DTYPE_t[:, :] arr_wse, DTYPE_t[:, :] arr_dem) -> np.ndarray:
    """Calculate the water depth array from arrays of terrain elevation and water surface elevation.
    If negative, default to zero.
    """
    cdef int rmax, cmax, r, c
    rmax = arr_dem.shape[0]
    cmax = arr_dem.shape[1]
    cdef DTYPE_t[:, :] arr_h = np.empty((rmax, cmax), dtype=DTYPE)

    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            arr_h[r, c] = max(arr_wse[r, c] - arr_dem[r, c], 0)
    return np.asarray(arr_h)


# --- Output Array Calculations ---

@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def calculate_wse(DTYPE_t[:, :] h_array, DTYPE_t[:, :] dem_array) -> np.ndarray:
    """Calculates Water Surface Elevation.
    Args:
        h_array: 2D array of water depths (m)
        dem_array: 2D array of ground elevations (m)
    Returns:
        2D array of water surface elevations (m)
    """
    cdef int rmax, cmax, r, c
    rmax = dem_array.shape[0]
    cmax = dem_array.shape[1]
    cdef DTYPE_t[:, :] arr_wse = np.empty((rmax, cmax), dtype=DTYPE)

    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            arr_wse[r, c] = h_array[r, c] + dem_array[r, c]
    return np.asarray(arr_wse)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def calculate_flux(DTYPE_t[:, :] flow_array, DTYPE_t cell_size) -> np.ndarray:
    """Calculates the flux (Qx or Qy) by multiplying flow array by cell size.
    Args:
        flow_array: 2D array of flow values (m²/s)
        cell_size: Grid cell size in perpendicular direction (m)
    Returns:
        2D array of flux values (m³/s)
    """
    cdef int rmax, cmax, r, c
    rmax = flow_array.shape[0]
    cmax = flow_array.shape[1]
    cdef DTYPE_t[:, :] arr_flux = np.empty((rmax, cmax), dtype=DTYPE)

    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            arr_flux[r, c] = flow_array[r, c] * cell_size
    return np.asarray(arr_flux)


# --- Statistical Calculations ---

@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def calculate_average_rate_from_total(
    DTYPE_t[:, :] total_volume_array,
    DTYPE_t interval_seconds,
    DTYPE_t conversion_factor,
) -> np.ndarray:
    """Calculates an average rate from a cumulated volume array.
    Args:
        total_volume_array: 2D array of total volumes (m)
        interval_seconds: Time interval over which volumes were accumulated (s)
        conversion_factor: Factor to convert rate units (default: 1.0 for m/s)
    Returns:
        2D array of average rates in converted units
    """
    cdef int rmax, cmax, r, c
    rmax = total_volume_array.shape[0]
    cmax = total_volume_array.shape[1]
    cdef DTYPE_t[:, :] arr_mean = np.empty((rmax, cmax), dtype=DTYPE)

    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            arr_mean[r, c] = (total_volume_array[r, c] / interval_seconds) * conversion_factor
    return np.asarray(arr_mean)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def accumulate_rate_to_total(
    DTYPE_t[:, ::1] accum_array,
    DTYPE_t[:, ::1] rate_array,
    DTYPE_t time_delta_seconds,
    padded=False,
):
    """Accumulates a rate array into a total array over a time delta.
       The operation is performed in-place on accum_array.
       With padded array, works on the inner array only
    Args:
        accum_array: 2D array to accumulate into (modified in-place)
        rate_array: 2D array of rates to accumulate
        time_delta_seconds: Time interval over which to accumulate (s)
    Returns:
        None (modifies accum_array in-place)
    """
    cdef int row_start, row_end
    cdef int col_start, col_end
    cdef int row_idx, col_idx

    if padded:
        row_start = 1
        row_end = rate_array.shape[0] - 1
        col_start = 1
        col_end = rate_array.shape[1] - 1
    else:
        row_start = 0
        row_end = rate_array.shape[0]
        col_start = 0
        col_end = rate_array.shape[1]

    cdef int total_elements = (row_end - row_start) * (col_end - col_start)
    # For small arrays the parall overhead is greater than the benefits
    if total_elements <= small_array_size:
        for row_idx in range(row_start, row_end):
            for col_idx in range(col_start, col_end):
                accum_array[row_idx, col_idx] += rate_array[row_idx, col_idx] * time_delta_seconds
    else:
        for row_idx in prange(row_start, row_end, nogil=True):
            for col_idx in range(col_start, col_end):
                accum_array[row_idx, col_idx] += rate_array[row_idx, col_idx] * time_delta_seconds
