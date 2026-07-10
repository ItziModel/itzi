"""
Copyright (C) 2026 Laurent Courty

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

ctypedef cython.floating DTYPE_t
cdef float PI = 3.1415926535898


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def apply_hydrology(
    DTYPE_t[:, :] arr_rain,
    DTYPE_t[:, :] arr_inf,
    DTYPE_t[:, :] arr_capped_losses,
    DTYPE_t[:, :] arr_h,
    DTYPE_t[:, :] arr_eff_precip,
    DTYPE_t dt,
):
    """Update arr_eff_precip in m/s
    rain and infiltration in m/s, deph in m, dt in seconds"""
    cdef int rmax, cmax, r, c
    cdef DTYPE_t hydro_raw, hydro_capped, losses_limit
    rmax = arr_rain.shape[0]
    cmax = arr_rain.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            hydro_raw = arr_rain[r, c] - arr_inf[r, c] - arr_capped_losses[r, c]
            losses_limit = - arr_h[r, c] / dt
            hydro_capped = max(losses_limit, hydro_raw)
            arr_eff_precip[r, c] = hydro_capped


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def infiltration_user(
    DTYPE_t[:, :] arr_h,
    DTYPE_t[:, :] arr_inf_in,
    DTYPE_t[:, :] arr_inf_out,
    DTYPE_t dt
):
    """Calculate infiltration rate using a user-defined fixed rate
    """
    cdef int rmax, cmax, r, c

    rmax = arr_h.shape[0]
    cmax = arr_h.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            # cap the rate
            arr_inf_out[r, c] = cap_infiltration_rate(dt, arr_h[r, c], arr_inf_in[r, c])


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def infiltration_ga(
    DTYPE_t[:, :] arr_h,
    DTYPE_t[:, :] arr_eff_por,
    DTYPE_t[:, :] arr_pressure,
    DTYPE_t[:, :] arr_conduct,
    DTYPE_t[:, :] arr_inf_amount,
    DTYPE_t[:, :] arr_water_soil_content,
    DTYPE_t[:, :] arr_inf_out,
    DTYPE_t dt
):
    """Calculate infiltration rate using the Green-Ampt formula
    """
    cdef int rmax, cmax, r, c
    cdef DTYPE_t infrate, avail_porosity, poros_cappress, conduct
    rmax = arr_h.shape[0]
    cmax = arr_h.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            conduct = arr_conduct[r, c]
            avail_porosity = max(arr_eff_por[r, c] - arr_water_soil_content[r, c], 0)
            poros_cappress = avail_porosity * (arr_pressure[r, c] + arr_h[r, c])
            infrate = conduct * (1 + (poros_cappress / arr_inf_amount[r, c]))
            # cap the rate
            infrate = cap_infiltration_rate(dt, arr_h[r, c], infrate)
            # update total infiltration amount
            arr_inf_amount[r, c] += infrate * dt
            # populate output infiltration array
            arr_inf_out[r, c] = infrate


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef DTYPE_t cap_infiltration_rate(DTYPE_t dt, DTYPE_t h, DTYPE_t infrate) noexcept nogil:
    """Cap the infiltration rate to not generate negative depths
    """
    return min(h / dt, infrate)
