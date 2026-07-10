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
from libc.math cimport sqrt as c_sqrt
from libc.math cimport atan2 as c_atan
from libc.math cimport fmax

ctypedef cython.floating DTYPE_t
cdef float PI = 3.1415926535898
cdef int solve_h_tile_rows = 64
cdef int solve_h_tile_cols = 128


def get_solve_h_tile_size():
    """Return the tile size used by `solve_h`."""
    return solve_h_tile_rows, solve_h_tile_cols


def set_solve_h_tile_size(int tile_rows, int tile_cols):
    """Set the tile size used by `solve_h`."""
    if tile_rows <= 0 or tile_cols <= 0:
        raise ValueError("solve_h tile sizes must be positive")

    global solve_h_tile_rows, solve_h_tile_cols
    solve_h_tile_rows = tile_rows
    solve_h_tile_cols = tile_cols


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
cdef inline void solve_h_tile(
    DTYPE_t[:, ::1] arr_ext,
    DTYPE_t[:, ::1] arr_qe,
    DTYPE_t[:, ::1] arr_qs,
    DTYPE_t[:, ::1] arr_bct,
    DTYPE_t[:, ::1] arr_bcv,
    DTYPE_t[:, ::1] arr_h,
    DTYPE_t[:, ::1] arr_hmax,
    DTYPE_t[:, ::1] arr_hfix,
    DTYPE_t[:, ::1] arr_herr,
    DTYPE_t[:, ::1] arr_hfe,
    DTYPE_t[:, ::1] arr_hfs,
    DTYPE_t[:, ::1] arr_v,
    DTYPE_t[:, ::1] arr_vdir,
    DTYPE_t[:, ::1] arr_vmax,
    DTYPE_t[:, ::1] arr_fr,
    DTYPE_t dx,
    DTYPE_t dy,
    DTYPE_t dt,
    DTYPE_t g,
    int r_start,
    int r_end,
    int c_start,
    int c_end,
) noexcept nogil:
    """Update depth, velocity, and Froude values for one tile."""
    cdef int r, c
    cdef DTYPE_t qext, qe, qw, qn, qs, h, q_sum, h_new, hmax, bct, bcv
    cdef DTYPE_t hfe, hfs, hfw, hfn, ve, vw, vn, vs, vx, vy, v, vdir
    cdef DTYPE_t eps = 1e-12  # Small epsilon to avoid division by zero

    for r in range(r_start, r_end):
        for c in range(c_start, c_end):
            qext = arr_ext[r, c]
            qe = arr_qe[r, c]
            qw = arr_qe[r, c-1]
            qn = arr_qs[r-1, c]
            qs = arr_qs[r, c]
            bct = arr_bct[r, c]
            bcv = arr_bcv[r, c]
            h = arr_h[r, c]
            hmax = arr_hmax[r, c]
            # Sum of flows in m/s
            q_sum = (qw - qe) / dx + (qn - qs) / dy
            # calculate new flow depth
            h_new = h + (qext + q_sum) * dt
            if h_new < 0.:
                # Write error. Always positive (mass creation)
                arr_herr[r, c] += - h_new
                h_new = 0.
            # Apply fixed water level
            if bct == 4:
                # Positive if water enters the domain
                arr_hfix[r, c] += bcv - h_new
                h_new = bcv
            # Update max depth array
            arr_hmax[r, c] = max(h_new, hmax)
            # Update depth array
            arr_h[r, c] = h_new

            ## Velocity and Froude ##
            hfe = arr_hfe[r, c]
            hfw = arr_hfe[r, c-1]
            hfn = arr_hfs[r-1, c]
            hfs = arr_hfs[r, c]
            # Branchless velocity calculations for vectorization
            # Use fmax to avoid division by zero,
            # then multiply by zero or one by using boolean operation
            ve = qe / fmax(hfe, eps) * (hfe > 0.)
            vw = qw / fmax(hfw, eps) * (hfw > 0.)
            vs = qs / fmax(hfs, eps) * (hfs > 0.)
            vn = qn / fmax(hfn, eps) * (hfn > 0.)
            # Velocities at the center of the cell
            vx = .5 * (ve + vw)
            vy = .5 * (vs + vn)

            # velocity magnitude and direction
            v = c_sqrt(vx*vx + vy*vy)  # sqrt faster than hypot
            arr_v[r, c] = v
            arr_vmax[r, c] = max(v, arr_vmax[r, c])
            vdir = c_atan(-vy, vx) * 180. / PI
            # Branchless. Add 360 only to negative numbers
            vdir = vdir + 360. * (vdir < 0)
            arr_vdir[r, c] = vdir

            # Froude number - use epsilon to avoid division by zero
            arr_fr[r, c] = v / c_sqrt(g * fmax(h_new, eps)) * (h_new > 0.)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
def solve_h(
    DTYPE_t[:, ::1] arr_ext,
    DTYPE_t[:, ::1] arr_qe,
    DTYPE_t[:, ::1] arr_qs,
    DTYPE_t[:, ::1] arr_bct,
    DTYPE_t[:, ::1] arr_bcv,
    DTYPE_t[:, ::1] arr_h,
    DTYPE_t[:, ::1] arr_hmax,
    DTYPE_t[:, ::1] arr_hfix,
    DTYPE_t[:, ::1] arr_herr,
    DTYPE_t[:, ::1] arr_hfe,
    DTYPE_t[:, ::1] arr_hfs,
    DTYPE_t[:, ::1] arr_v,
    DTYPE_t[:, ::1] arr_vdir,
    DTYPE_t[:, ::1] arr_vmax,
    DTYPE_t[:, ::1] arr_fr,
    DTYPE_t dx,
    DTYPE_t dy,
    DTYPE_t dt,
    DTYPE_t g
):
    """Update the water depth and max depth
    Adjust water depth according to in-domain 'boundary' condition
    Calculate vel. magnitude in m/s, direction in degree and Froude number.
    """
    cdef int rmax, cmax
    cdef int inner_rows, inner_cols
    cdef int num_tiles_r, num_tiles_c, tile_count
    cdef int tile_idx, tile_r, tile_c
    cdef int r_start, r_end, c_start, c_end

    rmax = arr_ext.shape[0] - 1
    cmax = arr_ext.shape[1] - 1
    if rmax <= 1 or cmax <= 1:
        return

    inner_rows = rmax - 1
    inner_cols = cmax - 1
    num_tiles_r = (inner_rows + solve_h_tile_rows - 1) // solve_h_tile_rows
    num_tiles_c = (inner_cols + solve_h_tile_cols - 1) // solve_h_tile_cols
    tile_count = num_tiles_r * num_tiles_c

    for tile_idx in prange(tile_count, nogil=True, schedule='static'):
        tile_r = tile_idx // num_tiles_c
        tile_c = tile_idx % num_tiles_c

        r_start = 1 + tile_r * solve_h_tile_rows
        r_end = r_start + solve_h_tile_rows
        if r_end > rmax:
            r_end = rmax

        c_start = 1 + tile_c * solve_h_tile_cols
        c_end = c_start + solve_h_tile_cols
        if c_end > cmax:
            c_end = cmax

        solve_h_tile(
            arr_ext=arr_ext,
            arr_qe=arr_qe,
            arr_qs=arr_qs,
            arr_bct=arr_bct,
            arr_bcv=arr_bcv,
            arr_h=arr_h,
            arr_hmax=arr_hmax,
            arr_hfix=arr_hfix,
            arr_herr=arr_herr,
            arr_hfe=arr_hfe,
            arr_hfs=arr_hfs,
            arr_v=arr_v,
            arr_vdir=arr_vdir,
            arr_vmax=arr_vmax,
            arr_fr=arr_fr,
            dx=dx,
            dy=dy,
            dt=dt,
            g=g,
            r_start=r_start,
            r_end=r_end,
            c_start=c_start,
            c_end=c_end,
        )
