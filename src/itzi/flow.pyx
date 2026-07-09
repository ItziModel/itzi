"""
Copyright (C) 2015-2026 Laurent Courty

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
from libc.math cimport pow as c_pow
from libc.math cimport sqrt as c_sqrt
from libc.math cimport atan2 as c_atan
from libc.math cimport fmax, fmin, copysign

ctypedef cython.floating DTYPE_t
cdef float PI = 3.1415926535898
cdef int solve_h_tile_rows = 64
cdef int solve_h_tile_cols = 128
cdef int solve_q_tile_rows = 64
cdef int solve_q_tile_cols = 128


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


def get_solve_q_tile_size():
    """Return the tile size used by `solve_q`."""
    return solve_q_tile_rows, solve_q_tile_cols


def set_solve_q_tile_size(int tile_rows, int tile_cols):
    """Set the tile size used by `solve_q`."""
    if tile_rows <= 0 or tile_cols <= 0:
        raise ValueError("solve_q tile sizes must be positive")

    global solve_q_tile_rows, solve_q_tile_cols
    solve_q_tile_rows = tile_rows
    solve_q_tile_cols = tile_cols


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
cdef inline void solve_qe_interior_at(
    DTYPE_t[:, ::1] arr_z,
    DTYPE_t[:, ::1] arr_n,
    DTYPE_t[:, ::1] arr_h,
    DTYPE_t[:, ::1] arr_qe,
    DTYPE_t[:, ::1] arr_qs,
    DTYPE_t[:, ::1] arr_hfe,
    DTYPE_t[:, ::1] arr_qe_new,
    DTYPE_t dt,
    DTYPE_t dx,
    DTYPE_t g,
    DTYPE_t theta,
    DTYPE_t hf_min,
    DTYPE_t slope_threshold,
    DTYPE_t max_slope,
    DTYPE_t z0,
    DTYPE_t wse0,
    DTYPE_t n0,
    DTYPE_t qe,
    DTYPE_t qs,
    int r,
    int c,
) noexcept nogil:
    """Solve eastward flow for one interior face."""
    cdef DTYPE_t wse_e
    cdef DTYPE_t z_e
    cdef DTYPE_t ne
    cdef DTYPE_t qe_st, qe_vect
    cdef DTYPE_t qe_new
    cdef DTYPE_t hf_e
    cdef DTYPE_t h_e
    cdef DTYPE_t slope_e

    z_e = arr_z[r, c+1]
    h_e = arr_h[r, c+1]
    wse_e = z_e + h_e
    hf_e = hflow(z0=z0, z1=z_e, wse0=wse0, wse1=wse_e)
    arr_hfe[r, c] = hf_e

    ne = 0.5 * (n0 + arr_n[r, c+1])
    slope_e = (wse0 - wse_e) / dx

    if hf_e <= 0:
        qe_new = 0
    elif hf_e > hf_min and abs(slope_e) < slope_threshold:
        qe_st = .25 * (qs + arr_qs[r-1, c] + arr_qs[r-1, c+1] + arr_qs[r, c+1])
        qe_vect = c_sqrt(qe*qe + qe_st*qe_st)  # Faster than hypot
        qe_new = flow_almeida2013(
            hf=hf_e,
            n=ne,
            qm1=arr_qe[r, c-1],
            q0=qe,
            qp1=arr_qe[r, c+1],
            q_norm=qe_vect,
            theta=theta,
            g=g,
            dt=dt,
            slope=slope_e,
        )
    else:
        qe_new = flow_GMS(
            flow_depth=hf_e,
            n=ne,
            slope=fmin(abs(slope_e), max_slope),
        )
        qe_new = copysign(qe_new, slope_e)

    arr_qe_new[r, c] = qe_new


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
cdef inline void solve_qe_west_boundary_at(
    DTYPE_t[:, ::1] arr_z,
    DTYPE_t[:, ::1] arr_h,
    DTYPE_t[:, ::1] arr_qe,
    DTYPE_t[:, ::1] arr_hfe,
    DTYPE_t[:, ::1] arr_bctype,
    DTYPE_t[:, ::1] arr_qe_new,
    int r,
) noexcept nogil:
    """Solve eastward boundary flow on the west edge."""
    cdef DTYPE_t z0, h0, wse0
    cdef DTYPE_t z_e, h_e, wse_e
    cdef DTYPE_t z_ee, h_ee, wse_ee
    cdef DTYPE_t hf_e, hf_ee

    z0 = arr_z[r, 0]
    h0 = arr_h[r, 0]
    wse0 = z0 + h0
    z_e = arr_z[r, 1]
    h_e = arr_h[r, 1]
    wse_e = z_e + h_e
    hf_e = hflow(z0=z0, z1=z_e, wse0=wse0, wse1=wse_e)
    arr_hfe[r, 0] = hf_e

    z_ee = arr_z[r, 2]
    h_ee = arr_h[r, 2]
    wse_ee = z_ee + h_ee
    hf_ee = hflow(z0=z_e, z1=z_ee, wse0=wse_e, wse1=wse_ee)

    arr_qe_new[r, 0] = boundary_flow(
        bctype=arr_bctype[r, 1],
        q_domain=arr_qe[r, 1],
        flow_depth_domain=hf_ee,
        flow_depth_boundary=hf_e,
    )


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
cdef inline void solve_qe_east_boundary_at(
    DTYPE_t[:, ::1] arr_z,
    DTYPE_t[:, ::1] arr_h,
    DTYPE_t[:, ::1] arr_qe,
    DTYPE_t[:, ::1] arr_hfe,
    DTYPE_t[:, ::1] arr_bctype,
    DTYPE_t[:, ::1] arr_qe_new,
    int col_east_boundary,
    int r,
) noexcept nogil:
    """Solve eastward boundary flow on the east edge."""
    cdef DTYPE_t z0, h0, wse0
    cdef DTYPE_t z_e, h_e, wse_e
    cdef DTYPE_t z_w, h_w, wse_w
    cdef DTYPE_t hf_e, hf_w

    z0 = arr_z[r, col_east_boundary]
    h0 = arr_h[r, col_east_boundary]
    wse0 = z0 + h0
    z_e = arr_z[r, col_east_boundary + 1]
    h_e = arr_h[r, col_east_boundary + 1]
    wse_e = z_e + h_e
    hf_e = hflow(z0=z0, z1=z_e, wse0=wse0, wse1=wse_e)
    arr_hfe[r, col_east_boundary] = hf_e

    z_w = arr_z[r, col_east_boundary - 1]
    h_w = arr_h[r, col_east_boundary - 1]
    wse_w = z_w + h_w
    hf_w = hflow(z0=z0, z1=z_w, wse0=wse0, wse1=wse_w)

    arr_qe_new[r, col_east_boundary] = boundary_flow(
        bctype=arr_bctype[r, col_east_boundary],
        q_domain=arr_qe[r, col_east_boundary - 1],
        flow_depth_domain=hf_w,
        flow_depth_boundary=hf_e,
    )


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
cdef inline void solve_qe_top_zero_at(
    DTYPE_t[:, ::1] arr_z,
    DTYPE_t[:, ::1] arr_h,
    DTYPE_t[:, ::1] arr_hfe,
    DTYPE_t[:, ::1] arr_qe_new,
    int c,
) noexcept nogil:
    """Zero eastward flow along the top edge."""
    cdef DTYPE_t z0, h0, wse0
    cdef DTYPE_t z_e, h_e, wse_e

    z0 = arr_z[0, c]
    h0 = arr_h[0, c]
    wse0 = z0 + h0
    z_e = arr_z[0, c+1]
    h_e = arr_h[0, c+1]
    wse_e = z_e + h_e
    arr_hfe[0, c] = hflow(z0=z0, z1=z_e, wse0=wse0, wse1=wse_e)
    arr_qe_new[0, c] = 0


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
cdef inline void solve_qs_interior_at(
    DTYPE_t[:, ::1] arr_z,
    DTYPE_t[:, ::1] arr_n,
    DTYPE_t[:, ::1] arr_h,
    DTYPE_t[:, ::1] arr_qe,
    DTYPE_t[:, ::1] arr_qs,
    DTYPE_t[:, ::1] arr_hfs,
    DTYPE_t[:, ::1] arr_qs_new,
    DTYPE_t dt,
    DTYPE_t dy,
    DTYPE_t g,
    DTYPE_t theta,
    DTYPE_t hf_min,
    DTYPE_t slope_threshold,
    DTYPE_t max_slope,
    DTYPE_t z0,
    DTYPE_t wse0,
    DTYPE_t n0,
    DTYPE_t qe,
    DTYPE_t qs,
    int r,
    int c,
) noexcept nogil:
    """Solve southward flow for one interior face."""
    cdef DTYPE_t wse_s
    cdef DTYPE_t z_s
    cdef DTYPE_t ns
    cdef DTYPE_t qs_st, qs_vect
    cdef DTYPE_t qs_new
    cdef DTYPE_t hf_s
    cdef DTYPE_t h_s
    cdef DTYPE_t slope_s

    z_s = arr_z[r+1, c]
    h_s = arr_h[r+1, c]
    wse_s = z_s + h_s
    hf_s = hflow(z0=z0, z1=z_s, wse0=wse0, wse1=wse_s)
    arr_hfs[r, c] = hf_s

    ns = 0.5 * (n0 + arr_n[r+1, c])
    slope_s = (wse0 - wse_s) / dy

    if hf_s <= 0:
        qs_new = 0
    elif hf_s > hf_min and abs(slope_s) < slope_threshold:
        qs_st = .25 * (qe + arr_qe[r+1, c] + arr_qe[r+1, c-1] + arr_qe[r, c-1])
        qs_vect = c_sqrt(qs*qs + qs_st*qs_st)
        qs_new = flow_almeida2013(
            hf=hf_s,
            n=ns,
            qm1=arr_qs[r-1, c],
            q0=qs,
            qp1=arr_qs[r+1, c],
            q_norm=qs_vect,
            theta=theta,
            g=g,
            dt=dt,
            slope=slope_s,
        )
    else:
        qs_new = flow_GMS(
            flow_depth=hf_s,
            n=ns,
            slope=fmin(abs(slope_s), max_slope),
        )
        qs_new = copysign(qs_new, slope_s)

    arr_qs_new[r, c] = qs_new


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
cdef inline void solve_qs_north_boundary_at(
    DTYPE_t[:, ::1] arr_z,
    DTYPE_t[:, ::1] arr_h,
    DTYPE_t[:, ::1] arr_qs,
    DTYPE_t[:, ::1] arr_hfs,
    DTYPE_t[:, ::1] arr_bctype,
    DTYPE_t[:, ::1] arr_qs_new,
    int c,
) noexcept nogil:
    """Solve southward boundary flow on the north edge."""
    cdef DTYPE_t z0, h0, wse0
    cdef DTYPE_t z_s, h_s, wse_s
    cdef DTYPE_t z_ss, h_ss, wse_ss
    cdef DTYPE_t hf_s, hf_ss

    z0 = arr_z[0, c]
    h0 = arr_h[0, c]
    wse0 = z0 + h0
    z_s = arr_z[1, c]
    h_s = arr_h[1, c]
    wse_s = z_s + h_s
    hf_s = hflow(z0=z0, z1=z_s, wse0=wse0, wse1=wse_s)
    arr_hfs[0, c] = hf_s

    z_ss = arr_z[2, c]
    h_ss = arr_h[2, c]
    wse_ss = z_ss + h_ss
    hf_ss = hflow(z0=z_s, z1=z_ss, wse0=wse_s, wse1=wse_ss)

    arr_qs_new[0, c] = boundary_flow(
        bctype=arr_bctype[1, c],
        q_domain=arr_qs[1, c],
        flow_depth_domain=hf_ss,
        flow_depth_boundary=hf_s,
    )


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
cdef inline void solve_qs_south_boundary_at(
    DTYPE_t[:, ::1] arr_z,
    DTYPE_t[:, ::1] arr_h,
    DTYPE_t[:, ::1] arr_qs,
    DTYPE_t[:, ::1] arr_hfs,
    DTYPE_t[:, ::1] arr_bctype,
    DTYPE_t[:, ::1] arr_qs_new,
    int row_south_boundary,
    int c,
) noexcept nogil:
    """Solve southward boundary flow on the south edge."""
    cdef DTYPE_t z0, h0, wse0
    cdef DTYPE_t z_s, h_s, wse_s
    cdef DTYPE_t z_n, h_n, wse_n
    cdef DTYPE_t hf_s, hf_n

    z0 = arr_z[row_south_boundary, c]
    h0 = arr_h[row_south_boundary, c]
    wse0 = z0 + h0
    z_s = arr_z[row_south_boundary + 1, c]
    h_s = arr_h[row_south_boundary + 1, c]
    wse_s = z_s + h_s
    hf_s = hflow(z0=z0, z1=z_s, wse0=wse0, wse1=wse_s)
    arr_hfs[row_south_boundary, c] = hf_s

    z_n = arr_z[row_south_boundary - 1, c]
    h_n = arr_h[row_south_boundary - 1, c]
    wse_n = z_n + h_n
    hf_n = hflow(z0=z0, z1=z_n, wse0=wse0, wse1=wse_n)

    arr_qs_new[row_south_boundary, c] = boundary_flow(
        bctype=arr_bctype[row_south_boundary, c],
        q_domain=arr_qs[row_south_boundary - 1, c],
        flow_depth_domain=hf_n,
        flow_depth_boundary=hf_s,
    )


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
cdef inline void solve_qs_left_zero_at(
    DTYPE_t[:, ::1] arr_z,
    DTYPE_t[:, ::1] arr_h,
    DTYPE_t[:, ::1] arr_hfs,
    DTYPE_t[:, ::1] arr_qs_new,
    int r,
) noexcept nogil:
    """Zero southward flow along the left edge."""
    cdef DTYPE_t z0, h0, wse0
    cdef DTYPE_t z_s, h_s, wse_s

    z0 = arr_z[r, 0]
    h0 = arr_h[r, 0]
    wse0 = z0 + h0
    z_s = arr_z[r+1, 0]
    h_s = arr_h[r+1, 0]
    wse_s = z_s + h_s
    arr_hfs[r, 0] = hflow(z0=z0, z1=z_s, wse0=wse0, wse1=wse_s)
    arr_qs_new[r, 0] = 0


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
cdef inline void solve_q_interior_core_tile(
    DTYPE_t[:, ::1] arr_z,
    DTYPE_t[:, ::1] arr_n,
    DTYPE_t[:, ::1] arr_h,
    DTYPE_t[:, ::1] arr_qe,
    DTYPE_t[:, ::1] arr_qs,
    DTYPE_t[:, ::1] arr_hfe,
    DTYPE_t[:, ::1] arr_hfs,
    DTYPE_t[:, ::1] arr_qe_new,
    DTYPE_t[:, ::1] arr_qs_new,
    DTYPE_t dt,
    DTYPE_t dx,
    DTYPE_t dy,
    DTYPE_t g,
    DTYPE_t theta,
    DTYPE_t hf_min,
    DTYPE_t slope_threshold,
    DTYPE_t max_slope,
    int r_start,
    int r_end,
    int c_start,
    int c_end,
) noexcept nogil:
    """Solve interior eastward and southward flows for one tile."""
    cdef int r, c
    cdef DTYPE_t z0, h0, wse0, n0, qe, qs

    for r in range(r_start, r_end):
        for c in range(c_start, c_end):
            z0 = arr_z[r, c]
            h0 = arr_h[r, c]
            wse0 = z0 + h0
            n0 = arr_n[r, c]
            qe = arr_qe[r, c]
            qs = arr_qs[r, c]

            solve_qe_interior_at(
                arr_z,
                arr_n,
                arr_h,
                arr_qe,
                arr_qs,
                arr_hfe,
                arr_qe_new,
                dt,
                dx,
                g,
                theta,
                hf_min,
                slope_threshold,
                max_slope,
                z0,
                wse0,
                n0,
                qe,
                qs,
                r,
                c,
            )
            solve_qs_interior_at(
                arr_z,
                arr_n,
                arr_h,
                arr_qe,
                arr_qs,
                arr_hfs,
                arr_qs_new,
                dt,
                dy,
                g,
                theta,
                hf_min,
                slope_threshold,
                max_slope,
                z0,
                wse0,
                n0,
                qe,
                qs,
                r,
                c,
            )


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
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def flow_dir(
    DTYPE_t[:, :] arr_max_dz,
    DTYPE_t[:, :] arr_dz0,
    DTYPE_t[:, :] arr_dz1,
    DTYPE_t[:, :] arr_dir
):
    """Update arr_dir with a rain-routing direction:
    0: the flow is going dowstream, index-wise
    1: the flow is going upstream, index-wise
    -1: no routing happening on that face
    """
    cdef int rmax, cmax, r, c
    cdef DTYPE_t max_dz, dz0, dz1, qdir
    rmax = arr_max_dz.shape[0]
    cmax = arr_max_dz.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            max_dz = arr_max_dz[r, c]
            dz0 = arr_dz0[r, c]
            dz1 = arr_dz1[r, c]
            qdir = arr_dir[r, c]
            if max_dz > 0:
                if max_dz == dz0:
                    qdir = 0
                elif max_dz == dz1:
                    qdir = 1
                else:
                    qdir = -1
            else:
                qdir = -1
            # update results array
            arr_dir[r, c] = qdir


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
def solve_q(
    DTYPE_t[:, ::1] arr_dire,
    DTYPE_t[:, ::1] arr_dirs,
    DTYPE_t[:, ::1] arr_z,
    DTYPE_t[:, ::1] arr_n,
    DTYPE_t[:, ::1] arr_h,
    DTYPE_t[:, ::1] arr_qe,
    DTYPE_t[:, ::1] arr_qs,
    DTYPE_t[:, ::1] arr_hfe,
    DTYPE_t[:, ::1] arr_hfs,
    DTYPE_t[:, ::1] arr_bctype,
    DTYPE_t[:, ::1] arr_bcvalue,
    DTYPE_t[:, ::1] arr_qe_new,
    DTYPE_t[:, ::1] arr_qs_new,
    DTYPE_t[:, ::1] arr_bcaccum,
    DTYPE_t dt,
    DTYPE_t dx,
    DTYPE_t dy,
    DTYPE_t g,
    DTYPE_t theta,
    DTYPE_t hf_min,
    DTYPE_t v_rout,
    DTYPE_t slope_threshold,
    DTYPE_t max_slope,
):
    """Calculate flow depth at the edges in m and flow in m2/s.
    Flow is positive when going east and south,
    and is computed at the S and E edges of each cell.
    Expect arrays padded by 1 cell all around.
    """

    cdef int rows, cols
    cdef int row_south_boundary
    cdef int col_east_boundary
    cdef int core_rows, core_cols
    cdef int num_tiles_r, num_tiles_c, tile_count
    cdef int tile_idx, tile_r, tile_c
    cdef int r_start, r_end, c_start, c_end
    cdef int r, c
    cdef DTYPE_t z0, h0, wse0, n0, qe, qs

    rows = arr_z.shape[0]
    cols = arr_z.shape[1]
    if rows < 3 or cols < 3:
        return

    row_south_boundary = rows - 2
    col_east_boundary = cols - 2

    core_rows = row_south_boundary - 1
    core_cols = col_east_boundary - 1
    if core_rows > 0 and core_cols > 0:
        num_tiles_r = (core_rows + solve_q_tile_rows - 1) // solve_q_tile_rows
        num_tiles_c = (core_cols + solve_q_tile_cols - 1) // solve_q_tile_cols
        tile_count = num_tiles_r * num_tiles_c

        for tile_idx in prange(tile_count, nogil=True, schedule='static'):
            tile_r = tile_idx // num_tiles_c
            tile_c = tile_idx % num_tiles_c

            r_start = 1 + tile_r * solve_q_tile_rows
            r_end = r_start + solve_q_tile_rows
            if r_end > row_south_boundary:
                r_end = row_south_boundary

            c_start = 1 + tile_c * solve_q_tile_cols
            c_end = c_start + solve_q_tile_cols
            if c_end > col_east_boundary:
                c_end = col_east_boundary

            solve_q_interior_core_tile(
                arr_z=arr_z,
                arr_n=arr_n,
                arr_h=arr_h,
                arr_qe=arr_qe,
                arr_qs=arr_qs,
                arr_hfe=arr_hfe,
                arr_hfs=arr_hfs,
                arr_qe_new=arr_qe_new,
                arr_qs_new=arr_qs_new,
                dt=dt,
                dx=dx,
                dy=dy,
                g=g,
                theta=theta,
                hf_min=hf_min,
                slope_threshold=slope_threshold,
                max_slope=max_slope,
                r_start=r_start,
                r_end=r_end,
                c_start=c_start,
                c_end=c_end,
            )

    if row_south_boundary >= 1 and col_east_boundary > 1:
        r = row_south_boundary
        for c in range(1, col_east_boundary):
            z0 = arr_z[r, c]
            h0 = arr_h[r, c]
            wse0 = z0 + h0
            n0 = arr_n[r, c]
            qe = arr_qe[r, c]
            qs = arr_qs[r, c]
            solve_qe_interior_at(
                arr_z=arr_z,
                arr_n=arr_n,
                arr_h=arr_h,
                arr_qe=arr_qe,
                arr_qs=arr_qs,
                arr_hfe=arr_hfe,
                arr_qe_new=arr_qe_new,
                dt=dt,
                dx=dx,
                g=g,
                theta=theta,
                hf_min=hf_min,
                slope_threshold=slope_threshold,
                max_slope=max_slope,
                z0=z0,
                wse0=wse0,
                n0=n0,
                qe=qe,
                qs=qs,
                r=r,
                c=c,
            )

    if row_south_boundary > 1:
        c = col_east_boundary
        for r in range(1, row_south_boundary):
            z0 = arr_z[r, c]
            h0 = arr_h[r, c]
            wse0 = z0 + h0
            n0 = arr_n[r, c]
            qe = arr_qe[r, c]
            qs = arr_qs[r, c]
            solve_qs_interior_at(
                arr_z=arr_z,
                arr_n=arr_n,
                arr_h=arr_h,
                arr_qe=arr_qe,
                arr_qs=arr_qs,
                arr_hfs=arr_hfs,
                arr_qs_new=arr_qs_new,
                dt=dt,
                dy=dy,
                g=g,
                theta=theta,
                hf_min=hf_min,
                slope_threshold=slope_threshold,
                max_slope=max_slope,
                z0=z0,
                wse0=wse0,
                n0=n0,
                qe=qe,
                qs=qs,
                r=r,
                c=c,
            )

    if col_east_boundary > 1:
        for c in range(1, col_east_boundary):
            solve_qe_top_zero_at(
                arr_z=arr_z,
                arr_h=arr_h,
                arr_hfe=arr_hfe,
                arr_qe_new=arr_qe_new,
                c=c,
            )

    if row_south_boundary > 1:
        for r in range(1, row_south_boundary):
            solve_qs_left_zero_at(
                arr_z=arr_z,
                arr_h=arr_h,
                arr_hfs=arr_hfs,
                arr_qs_new=arr_qs_new,
                r=r,
            )

    for r in range(row_south_boundary + 1):
        solve_qe_west_boundary_at(
            arr_z=arr_z,
            arr_h=arr_h,
            arr_qe=arr_qe,
            arr_hfe=arr_hfe,
            arr_bctype=arr_bctype,
            arr_qe_new=arr_qe_new,
            r=r,
        )
        solve_qe_east_boundary_at(
            arr_z=arr_z,
            arr_h=arr_h,
            arr_qe=arr_qe,
            arr_hfe=arr_hfe,
            arr_bctype=arr_bctype,
            arr_qe_new=arr_qe_new,
            col_east_boundary=col_east_boundary,
            r=r,
        )

    for c in range(col_east_boundary + 1):
        solve_qs_north_boundary_at(
            arr_z=arr_z,
            arr_h=arr_h,
            arr_qs=arr_qs,
            arr_hfs=arr_hfs,
            arr_bctype=arr_bctype,
            arr_qs_new=arr_qs_new,
            c=c,
        )
        solve_qs_south_boundary_at(
            arr_z=arr_z,
            arr_h=arr_h,
            arr_qs=arr_qs,
            arr_hfs=arr_hfs,
            arr_bctype=arr_bctype,
            arr_qs_new=arr_qs_new,
            row_south_boundary=row_south_boundary,
            c=c,
        )


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def accumulate_boundary_fluxes(
    DTYPE_t[:, ::1] arr_qe_new,
    DTYPE_t[:, ::1] arr_qs_new,
    DTYPE_t[:, ::1] arr_bcaccum,
    DTYPE_t dt,
    DTYPE_t dx,
    DTYPE_t dy,
):
    """Accumulate boundary fluxes into the boundary storage array."""
    cdef int r, c
    cdef int row_south_boundary = arr_bcaccum.shape[0] - 2
    cdef int col_east_boundary = arr_bcaccum.shape[1] - 2

    for r in range(1, row_south_boundary + 1):
        # West boundary: positive eastward flow enters the domain.
        arr_bcaccum[r, 1] += arr_qe_new[r, 0] * dt / dx
        # East boundary: positive eastward flow leaves the domain.
        arr_bcaccum[r, col_east_boundary] -= arr_qe_new[r, col_east_boundary] * dt / dx

    for c in range(1, col_east_boundary + 1):
        # North boundary: positive southward flow enters the domain.
        arr_bcaccum[1, c] += arr_qs_new[0, c] * dt / dy
        # South boundary: positive southward flow leaves the domain.
        arr_bcaccum[row_south_boundary, c] -= arr_qs_new[row_south_boundary, c] * dt / dy


cdef DTYPE_t hflow(DTYPE_t z0, DTYPE_t z1, DTYPE_t wse0, DTYPE_t wse1) noexcept nogil:
    """calculate flow depth
    """
    return max(wse1, wse0) - max(z1, z0)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef DTYPE_t flow_almeida2013(
    DTYPE_t hf,
    DTYPE_t n,
    DTYPE_t qm1,
    DTYPE_t q0,
    DTYPE_t qp1,
    DTYPE_t q_norm,
    DTYPE_t theta,
    DTYPE_t g,
    DTYPE_t dt,
    DTYPE_t slope,
) noexcept nogil:
    """Solve flow using q-centered scheme from Almeida et Al. (2013)
    """
    cdef DTYPE_t term_1, term_2, term_3

    term_1 = theta * q0 + (1 - theta) * (qm1 + qp1) * 0.5
    term_2 = g * hf * dt * slope
    term_3 = 1 + g * dt * (n*n) * q_norm / c_pow(hf, 7./3.)
    # If flow direction is not coherent with surface slope,
    # use only previous flow, i.e. ~switch to Bates 2010
    if term_1 * term_2 < 0:
        term_1 = q0
    return (term_1 + term_2) / term_3


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef DTYPE_t flow_GMS(
    DTYPE_t flow_depth,
    DTYPE_t n,
    DTYPE_t slope,
) noexcept nogil:
    """Solve flow in m2/s with the Gauckler-Manning-Strickler formula.
    """
    cdef DTYPE_t v
    # Hydraulics radius is flow_depth because the wetted perimeter is only the flow width, so it cancels out.
    v = (1.0 / n) * c_pow(flow_depth, 2.0 / 3.0) * c_sqrt(slope)
    return v * flow_depth


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef DTYPE_t boundary_flow(
    DTYPE_t bctype,
    DTYPE_t q_domain,
    DTYPE_t flow_depth_domain,
    DTYPE_t flow_depth_boundary,
) noexcept nogil:
    """Solve flow in m2/s with the Gauckler-Manning-Strickler formula.
    """
    cdef DTYPE_t domain_velocity, boundary_flow

    # Open boundary: velocity inside the domain is equal to velocity at the boundary
    if bctype == 2 and flow_depth_domain > 0:
        boundary_flow = (q_domain / flow_depth_domain) * flow_depth_boundary
    # user-defined WSE - flow solved with GMS formula
    elif bctype == 3:
        # Not implemented yet
        boundary_flow = 0.
    # Everything else is closed
    else:
        boundary_flow = 0.
    return boundary_flow


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef DTYPE_t rain_routing(
    DTYPE_t h0,
    DTYPE_t wse0,
    DTYPE_t wse1,
    DTYPE_t dt,
    DTYPE_t cell_len,
    DTYPE_t v_routing,
) noexcept nogil:
    """Calculate flow routing at a face in m2/s
    Cf. Sampson et al. (2013)
    """
    cdef DTYPE_t maxflow, q_routing, dh
    # fraction of the depth to be routed
    dh = wse0 - wse1
    # make sure it's positive (should not happend, checked before)
    dh = max(dh, 0)
    # if WSE of destination cell is below the dem of the drained cell, set to h0
    dh = min(dh, h0)
    maxflow = cell_len * dh / dt
    q_routing = min(dh * v_routing, maxflow)
    return q_routing


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
