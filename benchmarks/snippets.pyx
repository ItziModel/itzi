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

from libc.math cimport atan2 as c_atan2
from libc.math cimport atan2f as c_atan2f
from libc.math cimport copysign as c_copysign
from libc.math cimport copysignf as c_copysignf
from libc.math cimport fabs as c_fabs
from libc.math cimport fabsf as c_fabsf
from libc.math cimport fmax, fmaxf, fmin, fminf
from libc.math cimport hypot
from libc.math cimport pow as c_pow
from libc.math cimport powf as c_powf
from libc.math cimport sqrt as c_sqrt
from libc.math cimport sqrtf as c_sqrtf
from libc.math cimport cbrt as c_cbrt

ctypedef cython.floating DTYPE_t

cdef double PI_D = 3.14159265358979323846
cdef double RAD_TO_DEG_D = 180.0 / PI_D
cdef float PI_F = 3.1415927410125732
cdef float RAD_TO_DEG_F = <float>(180.0 / PI_F)
cdef float ZERO_F = 0.0
cdef float HALF_F = 0.5
cdef float ONE_F = 1.0
cdef float TWO_THIRDS_F = <float>(2.0 / 3.0)
cdef float SEVEN_THIRDS_F = <float>(7.0 / 3.0)
cdef float DEG_360_F = 360.0
cdef float EPS_F = 1e-12


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
def branchless_velocity(
    DTYPE_t[:, ::1] arr_qe,
    DTYPE_t[:, ::1] arr_qs,
    DTYPE_t[:, ::1] arr_hfe,
    DTYPE_t[:, ::1] arr_hfs,
):
    """function for benchmarking purpose
    """
    cdef int rmax, cmax, r, c
    cdef DTYPE_t qe, qw, qn, qs
    cdef DTYPE_t hfe, hfs, hfw, hfn, ve, vw, vn, vs
    cdef DTYPE_t eps = 1e-12  # Small epsilon to avoid division by zero

    rmax = arr_qe.shape[0] - 1
    cmax = arr_qe.shape[1] - 1
    for r in prange(1, rmax, nogil=True):
        for c in range(1, cmax):
            qe = arr_qe[r, c]
            qw = arr_qe[r, c-1]
            qn = arr_qs[r-1, c]
            qs = arr_qs[r, c]

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


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
def branching_velocity(
    DTYPE_t[:, ::1] arr_qe,
    DTYPE_t[:, ::1] arr_qs,
    DTYPE_t[:, ::1] arr_hfe,
    DTYPE_t[:, ::1] arr_hfs,
):
    """function for benchmarking purpose
    """
    cdef int rmax, cmax, r, c
    cdef DTYPE_t qe, qw, qn, qs
    cdef DTYPE_t hfe, hfs, hfw, hfn, ve, vw, vn, vs

    rmax = arr_qe.shape[0] - 1
    cmax = arr_qe.shape[1] - 1
    for r in prange(1, rmax, nogil=True):
        for c in range(1, cmax):
            qe = arr_qe[r, c]
            qw = arr_qe[r, c-1]
            qn = arr_qs[r-1, c]
            qs = arr_qs[r, c]

            hfe = arr_hfe[r, c]
            hfw = arr_hfe[r, c-1]
            hfn = arr_hfs[r-1, c]
            hfs = arr_hfs[r, c]
            # branching velocity calculations
            if hfe <= 0.:
                ve = 0.
            else:
                ve = qe / hfe
            if hfw <= 0.:
                vw = 0.
            else:
                vw = qw / hfw
            if hfs <= 0.:
                vs = 0.
            else:
                vs = qs / hfs
            if hfn <= 0.:
                vn = 0.
            else:
                vn = qn / hfn


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
def arr_hypot(DTYPE_t[:, ::1] arr_qe, DTYPE_t[:, ::1] arr_qs):
    """function for benchmarking purpose
    """
    cdef int rmax, cmax, r, c
    cdef DTYPE_t qe, qs, q

    rmax = arr_qe.shape[0] - 1
    cmax = arr_qe.shape[1] - 1
    for r in prange(1, rmax, nogil=True):
        for c in range(1, cmax):
            qe = arr_qe[r, c]
            qs = arr_qs[r, c]

            q = hypot(qe, qs)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
def arr_sqrt(DTYPE_t[:, ::1] arr_qe, DTYPE_t[:, ::1] arr_qs):
    """function for benchmarking purpose
    """
    cdef int rmax, cmax, r, c
    cdef DTYPE_t qe, qs, q

    rmax = arr_qe.shape[0] - 1
    cmax = arr_qe.shape[1] - 1
    for r in prange(1, rmax, nogil=True):
        for c in range(1, cmax):
            qe = arr_qe[r, c]
            qs = arr_qs[r, c]

            q = c_sqrt(qe*qe + qs*qs)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
def arr_pow_two_thirds(DTYPE_t[:, ::1] arr_h, DTYPE_t[:, ::1] arr_out):
    """function for benchmarking purpose
    """
    cdef int rmax, cmax, r, c
    cdef DTYPE_t h

    rmax = arr_h.shape[0]
    cmax = arr_h.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            h = arr_h[r, c]
            arr_out[r, c] = c_pow(h, 2.0 / 3.0)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
def arr_pow_two_thirds_float32(float[:, ::1] arr_h, float[:, ::1] arr_out):
    """function for benchmarking purpose
    """
    cdef int rmax, cmax, r, c
    cdef float h

    rmax = arr_h.shape[0]
    cmax = arr_h.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            h = arr_h[r, c]
            arr_out[r, c] = c_powf(h, TWO_THIRDS_F)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
def arr_cbrt_two_thirds(DTYPE_t[:, ::1] arr_h, DTYPE_t[:, ::1] arr_out):
    """function for benchmarking purpose
    """
    cdef int rmax, cmax, r, c
    cdef DTYPE_t h

    rmax = arr_h.shape[0]
    cmax = arr_h.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            h = arr_h[r, c]
            arr_out[r, c] = c_cbrt(h * h)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
def arr_pow_seven_thirds(DTYPE_t[:, ::1] arr_h, DTYPE_t[:, ::1] arr_out):
    """function for benchmarking purpose
    """
    cdef int rmax, cmax, r, c
    cdef DTYPE_t h

    rmax = arr_h.shape[0]
    cmax = arr_h.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            h = arr_h[r, c]
            arr_out[r, c] = c_pow(h, 7.0 / 3.0)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
def arr_pow_seven_thirds_float32(float[:, ::1] arr_h, float[:, ::1] arr_out):
    """function for benchmarking purpose
    """
    cdef int rmax, cmax, r, c
    cdef float h

    rmax = arr_h.shape[0]
    cmax = arr_h.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            h = arr_h[r, c]
            arr_out[r, c] = c_powf(h, SEVEN_THIRDS_F)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
def arr_cbrt_seven_thirds(DTYPE_t[:, ::1] arr_h, DTYPE_t[:, ::1] arr_out):
    """function for benchmarking purpose
    """
    cdef int rmax, cmax, r, c
    cdef DTYPE_t h

    rmax = arr_h.shape[0]
    cmax = arr_h.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            h = arr_h[r, c]
            arr_out[r, c] = h * h * c_cbrt(h)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
def arr_almeida2013_generic(
    DTYPE_t[:, ::1] arr_hf,
    DTYPE_t[:, ::1] arr_n,
    DTYPE_t[:, ::1] arr_qm1,
    DTYPE_t[:, ::1] arr_q0,
    DTYPE_t[:, ::1] arr_qp1,
    DTYPE_t[:, ::1] arr_q_norm,
    DTYPE_t[:, ::1] arr_slope,
    DTYPE_t[:, ::1] arr_out,
    DTYPE_t theta,
    DTYPE_t g,
    DTYPE_t dt,
):
    """Function for benchmarking the main solve_q update formula."""
    cdef int rmax, cmax, r, c
    cdef DTYPE_t hf, n, qm1, q0, qp1, q_norm, slope
    cdef DTYPE_t term_1, term_2, term_3

    rmax = arr_hf.shape[0]
    cmax = arr_hf.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            hf = arr_hf[r, c]
            n = arr_n[r, c]
            qm1 = arr_qm1[r, c]
            q0 = arr_q0[r, c]
            qp1 = arr_qp1[r, c]
            q_norm = arr_q_norm[r, c]
            slope = arr_slope[r, c]

            term_1 = theta * q0 + (1.0 - theta) * (qm1 + qp1) * 0.5
            term_2 = g * hf * dt * slope
            term_3 = 1.0 + g * dt * (n * n) * q_norm / c_pow(hf, 7.0 / 3.0)
            if term_1 * term_2 < 0:
                term_1 = q0
            arr_out[r, c] = (term_1 + term_2) / term_3


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
def arr_almeida2013_float32(
    float[:, ::1] arr_hf,
    float[:, ::1] arr_n,
    float[:, ::1] arr_qm1,
    float[:, ::1] arr_q0,
    float[:, ::1] arr_qp1,
    float[:, ::1] arr_q_norm,
    float[:, ::1] arr_slope,
    float[:, ::1] arr_out,
    float theta,
    float g,
    float dt,
):
    """Function for benchmarking the float32-only solve_q update formula."""
    cdef int rmax, cmax, r, c
    cdef float hf, n, qm1, q0, qp1, q_norm, slope
    cdef float term_1, term_2, term_3

    rmax = arr_hf.shape[0]
    cmax = arr_hf.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            hf = arr_hf[r, c]
            n = arr_n[r, c]
            qm1 = arr_qm1[r, c]
            q0 = arr_q0[r, c]
            qp1 = arr_qp1[r, c]
            q_norm = arr_q_norm[r, c]
            slope = arr_slope[r, c]

            term_1 = theta * q0 + (ONE_F - theta) * (qm1 + qp1) * HALF_F
            term_2 = g * hf * dt * slope
            term_3 = ONE_F + g * dt * (n * n) * q_norm / c_powf(hf, SEVEN_THIRDS_F)
            if term_1 * term_2 < ZERO_F:
                term_1 = q0
            arr_out[r, c] = (term_1 + term_2) / term_3


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
def arr_signed_flow_gms_generic(
    DTYPE_t[:, ::1] arr_flow_depth,
    DTYPE_t[:, ::1] arr_n,
    DTYPE_t[:, ::1] arr_slope,
    DTYPE_t[:, ::1] arr_out,
    DTYPE_t max_slope,
):
    """Function for benchmarking the signed GMS fallback path in solve_q."""
    cdef int rmax, cmax, r, c
    cdef DTYPE_t flow_depth, n, slope, clipped_slope, v

    rmax = arr_flow_depth.shape[0]
    cmax = arr_flow_depth.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            flow_depth = arr_flow_depth[r, c]
            n = arr_n[r, c]
            slope = arr_slope[r, c]
            clipped_slope = fmin(c_fabs(slope), max_slope)
            v = (1.0 / n) * c_pow(flow_depth, 2.0 / 3.0) * c_sqrt(clipped_slope)
            arr_out[r, c] = c_copysign(v * flow_depth, slope)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
def arr_signed_flow_gms_float32(
    float[:, ::1] arr_flow_depth,
    float[:, ::1] arr_n,
    float[:, ::1] arr_slope,
    float[:, ::1] arr_out,
    float max_slope,
):
    """Function for benchmarking the float32-only signed GMS path."""
    cdef int rmax, cmax, r, c
    cdef float flow_depth, n, slope, clipped_slope, v

    rmax = arr_flow_depth.shape[0]
    cmax = arr_flow_depth.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            flow_depth = arr_flow_depth[r, c]
            n = arr_n[r, c]
            slope = arr_slope[r, c]
            clipped_slope = fminf(c_fabsf(slope), max_slope)
            v = (ONE_F / n) * c_powf(flow_depth, TWO_THIRDS_F) * c_sqrtf(clipped_slope)
            arr_out[r, c] = c_copysignf(v * flow_depth, slope)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
def arr_velocity_diagnostics_generic(
    DTYPE_t[:, ::1] arr_qx,
    DTYPE_t[:, ::1] arr_qy,
    DTYPE_t[:, ::1] arr_h,
    DTYPE_t[:, ::1] arr_v,
    DTYPE_t[:, ::1] arr_vdir,
    DTYPE_t[:, ::1] arr_fr,
    DTYPE_t g,
):
    """Function for benchmarking the solve_h diagnostic math path."""
    cdef int rmax, cmax, r, c
    cdef DTYPE_t qx, qy, h, vx, vy, v, vdir
    cdef DTYPE_t eps = 1e-12

    rmax = arr_qx.shape[0]
    cmax = arr_qx.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            qx = arr_qx[r, c]
            qy = arr_qy[r, c]
            h = arr_h[r, c]

            vx = qx / fmax(h, eps) * (h > 0.0)
            vy = qy / fmax(h, eps) * (h > 0.0)
            v = c_sqrt(vx * vx + vy * vy)
            arr_v[r, c] = v
            vdir = c_atan2(-vy, vx) * RAD_TO_DEG_D
            vdir = vdir + 360.0 * (vdir < 0.0)
            arr_vdir[r, c] = vdir
            arr_fr[r, c] = v / c_sqrt(g * fmax(h, eps)) * (h > 0.0)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.initializedcheck(False)  # Skip initialization checks for performance
@cython.nonecheck(False)  # Skip None checks for performance
def arr_velocity_diagnostics_float32(
    float[:, ::1] arr_qx,
    float[:, ::1] arr_qy,
    float[:, ::1] arr_h,
    float[:, ::1] arr_v,
    float[:, ::1] arr_vdir,
    float[:, ::1] arr_fr,
    float g,
):
    """Function for benchmarking the float32-only solve_h diagnostic math path."""
    cdef int rmax, cmax, r, c
    cdef float qx, qy, h, vx, vy, v, vdir

    rmax = arr_qx.shape[0]
    cmax = arr_qx.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            qx = arr_qx[r, c]
            qy = arr_qy[r, c]
            h = arr_h[r, c]

            vx = qx / fmaxf(h, EPS_F) * (h > ZERO_F)
            vy = qy / fmaxf(h, EPS_F) * (h > ZERO_F)
            v = c_sqrtf(vx * vx + vy * vy)
            arr_v[r, c] = v
            vdir = c_atan2f(-vy, vx) * RAD_TO_DEG_F
            vdir = vdir + DEG_360_F * (vdir < ZERO_F)
            arr_vdir[r, c] = vdir
            arr_fr[r, c] = v / c_sqrtf(g * fmaxf(h, EPS_F)) * (h > ZERO_F)
