# coding=utf8
"""
Copyright (C) 2015  Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""
from __future__ import division
cimport numpy as np
cimport cython
from cython.parallel cimport prange
import numpy as np
from libc.math cimport pow as c_pow
from libc.math cimport sqrt as c_sqrt
from libc.math cimport fabs as c_abs

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

@cython.wraparound(False)  # Disable negative index check
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def flow_dir(np.ndarray[DTYPE_t, ndim=2] arr_max_dz,
            np.ndarray[DTYPE_t, ndim=2] arr_dz0,
            np.ndarray[DTYPE_t, ndim=2] arr_dz1,
            np.ndarray[np.int8_t, ndim=2] arr_dir):
    '''Populate arr_dir with a rain-routing direction:
    0: the flow is going dowstream, index-wise
    1: the flow is going upstream, index-wise
    -1: no routing happening on that face
    '''
    cdef int rmax, cmax, r, c, qdir
    cdef float max_dz, dz0, dz1
    rmax = arr_max_dz.shape[0]
    cmax = arr_max_dz.shape[1]
    with nogil:
        for r in prange(rmax):
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
def solve_qnorm(np.ndarray[DTYPE_t, ndim=2] arr_q0,
        np.ndarray[DTYPE_t, ndim=2] arr_q1, np.ndarray[DTYPE_t, ndim=2] arr_q2,
        np.ndarray[DTYPE_t, ndim=2] arr_q3, np.ndarray[DTYPE_t, ndim=2] arr_q4,
        np.ndarray[DTYPE_t, ndim=2] arr_qnorm):
    '''Calculate the qnorm
    '''
    cdef int rmax, cmax, r, c
    cdef float q0, q1, q2, q3, q4, q_st

    rmax = arr_q0.shape[0]
    cmax = arr_q0.shape[1]
    with nogil:
        for r in prange(rmax):
            for c in range(cmax):
                q0 = arr_q0[r, c]
                q1 = arr_q1[r, c]
                q2 = arr_q2[r, c]
                q3 = arr_q3[r, c]
                q4 = arr_q4[r, c]
                # calculate average flow from stencil
                q_st = (q1 + q2 + q3 + q4) * .25
                # calculate qnorm
                arr_qnorm[r, c] = c_sqrt(q0*q0 + q_st*q_st)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def solve_q(np.ndarray[np.int8_t, ndim=2] arr_dir,
        np.ndarray[DTYPE_t, ndim=2] arr_z0, np.ndarray[DTYPE_t, ndim=2] arr_z1,
        np.ndarray[DTYPE_t, ndim=2] arr_n0, np.ndarray[DTYPE_t, ndim=2] arr_n1,
        np.ndarray[DTYPE_t, ndim=2] arr_h0, np.ndarray[DTYPE_t, ndim=2] arr_h1,
        np.ndarray[DTYPE_t, ndim=2] arr_q0, np.ndarray[DTYPE_t, ndim=2] arr_q1,
        np.ndarray[DTYPE_t, ndim=2] arr_qm1, np.ndarray[DTYPE_t, ndim=2] arr_qnorm,
        np.ndarray[DTYPE_t, ndim=2] arr_q0_new, np.ndarray[DTYPE_t, ndim=2] arr_hf,
        float dt, float cell_len, float g, float theta, float hf_min, float v_rout, float sl_thres):
    '''Calculate flow through the domain, including hflow
    '''

    cdef int rmax, cmax, r, c, qdir
    cdef float z1, z0, h1, h0, wse1, wse0, hf
    cdef float q0, qup, qdown, q_vect, n
    cdef float term_1, term_2, term_3, q0_new, slope, num, den

    rmax = arr_z0.shape[0]
    cmax = arr_z0.shape[1]
    with nogil:
        for r in prange(rmax):
            for c in xrange(cmax):
                # calculate wse
                z1 = arr_z1[r, c]
                z0 = arr_z0[r, c]
                h1 = arr_h1[r, c]
                h0 = arr_h0[r, c]
                wse1 = z1 + h1
                wse0 = z0 + h0
                # flow depth (hf)
                hf = max(wse1, wse0) - max(z1, z0)
                # update hflow array
                arr_hf[r, c] = hf

                # q
                q0 = arr_q0[r, c]
                qup = arr_qm1[r, c]
                qdown = arr_q1[r, c]
                # q_vect
                q_vect = arr_qnorm[r, c]

                # flow dir
                qdir = arr_dir[r, c]

                n = 0.5 * (arr_n0[r, c] + arr_n1[r, c])
                slope = (wse0 - wse1) / cell_len
                if hf <= 0:
                    q0_new = 0
                # Almeida 2013
                elif hf > hf_min:
                    q0_new = almeida2013(theta, q0, qup, qdown, n,
                                        g, hf, dt, slope, q_vect)
                # flow going W or N, i.e negative
                elif hf <= hf_min and qdir == 0 and wse1 > wse0:
                    q0_new = - rain_routing(h1, wse1, wse0,
                                            dt, cell_len, v_rout)
                # flow going E or S, i.e positive
                elif hf <= hf_min and  qdir == 1 and wse0 > wse1:
                    q0_new = rain_routing(h0, wse0, wse1,
                                            dt, cell_len, v_rout)
                else:
                    q0_new = 0
                # populate the array
                arr_q0_new[r,c] = q0_new

@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef float rain_routing(float h0, float wse0, float wse1, float dt,
                        float cell_len, float v_routing) nogil:
    """Calculate flow routing at a face in m2/s
    Cf. Sampson et al. (2013)
    """
    cdef float maxflow, q_routing, dh
    # fraction of the depth to be routed
    dh = wse0 - wse1
    # make sure it's positive (should not happend, checked before)
    dh = max(dh, 0)
    # if WSE of destination cell is below the dem of the drained cell, set to h0
    dh = min(dh, h0)
    maxflow = cell_len * dh / dt
    if dh <= 0.0001:
        q_routing = maxflow
    else:
        q_routing = min(dh * v_routing, maxflow)
    return q_routing

@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef float almeida2013(float theta, float q0, float qup, float qdown, float n,
        float g, float hf, float dt, float slope, float q_vect) nogil:
    '''Solve flow using q-centered scheme from Almeida and Bates (2010)
    '''
    cdef float term_1, term_2, term_3
    term_1 = theta * q0 + (1 - theta) * (qup + qdown) * 0.5
    term_2 = g * hf * dt * slope
    term_3 = 1 + g * dt * (n*n) * q_vect / c_pow(hf, 7./3.)
    # If flow direction is not coherent with surface slope,
    # use only previous flow, i.e. ~switch to Bates 2010
    if term_1 * term_2 < 0:
        term_1 = q0
    return (term_1 + term_2) / term_3


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def solve_h(np.ndarray[DTYPE_t, ndim=2] arr_ext,
        np.ndarray[DTYPE_t, ndim=2] arr_qe, np.ndarray[DTYPE_t, ndim=2] arr_qw,
        np.ndarray[DTYPE_t, ndim=2] arr_qn, np.ndarray[DTYPE_t, ndim=2] arr_qs,
        np.ndarray[DTYPE_t, ndim=2] arr_h, float dx, float dy, float dt):
    '''Update the water depth
    '''
    cdef int rmax, cmax, r, c
    cdef float qext, qe, qw, qn, qs, h, q_sum, h_new

    rmax = arr_qe.shape[0]
    cmax = arr_qe.shape[1]
    with nogil:
        for r in prange(rmax):
            for c in range(cmax):
                qext = arr_ext[r, c]
                qe = arr_qe[r, c]
                qw = arr_qw[r, c]
                qn = arr_qn[r, c]
                qs = arr_qs[r, c]
                h = arr_h[r, c]
                # Sum of flows in m/s
                q_sum = (qw - qe) / dx + (qn - qs) / dy
                # calculatre new flow depth, min depth zero
                h_new = max(h + (qext + q_sum) * dt, 0)
                # Update depth array
                arr_h[r, c] = h_new
