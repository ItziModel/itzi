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

from libc.math cimport pow as c_pow
from libc.math cimport sqrt as c_sqrt
from libc.math cimport cbrt as c_cbrt
from libc.math cimport hypot, fmax

ctypedef cython.floating DTYPE_t


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
