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
import numpy as np
from libc.math cimport pow, sqrt, abs

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

def solve_q_loop(
        np.ndarray[DTYPE_t, ndim=2] arr_z,
        np.ndarray[DTYPE_t, ndim=2] arr_n,
        np.ndarray[DTYPE_t, ndim=2] arr_h_old,
        np.ndarray[DTYPE_t, ndim=2] arrp_qw,
        np.ndarray[DTYPE_t, ndim=2] arrp_qn,
        np.ndarray[DTYPE_t, ndim=2] arr_qw_new,
        np.ndarray[DTYPE_t, ndim=2] arr_qn_new,
        float dt, float dx, float dy, float g, float theta, float hf_min):
    '''Solve flow equation, including hflow and q_vect, using
    loop through the domain
    '''
#~     assert arr_z.dtype == arr_n.dtype == arr_h_old.dtype == \
#~             arrp_qw.dtype == arrp_qn.dtype == DTYPE
    cdef unsigned int rmax, cmax, r, c, rq, cq
    cdef float zup, z0, hup, h0, wseup, wse0, hf
    cdef float q0, qy1, qy2, qy3, qy4, qy_avg, q_vect, num, den
    cdef float n, qup, qdown, term_1, term_2, term_3, qw_new, qn_new, slope
    cdef float qx1, qx2, qx3, qx4, qx_avg

    rmax = arr_z.shape[0]
    cmax = arr_z.shape[1]

    for r in xrange(rmax):
        for c in xrange(cmax):
            # add 1 to indices applied on padded q array
            cq = c + 1
            rq = r + 1
            #####################
            # flow in x dimension
            if c == 0:
                continue
            # calculate flow depth (hf)
            zup = arr_z[r, c-1]
            z0 = arr_z[r,c]
            hup = arr_h_old[r,c-1]
            h0 = arr_h_old[r,c]
            wseup = zup + hup
            wse0 = z0 + h0
            hf = max(wseup, wse0) - max(zup, z0)
            # q vector norm
            q0 = arrp_qw[rq, cq]
            qy1 = arrp_qn[rq-1, cq-1]
            qy2 = arrp_qn[rq-1, cq]
            qy3 = arrp_qn[rq, cq-1]
            qy4 = arrp_qn[rq, cq]
            qy_avg = (qy1 + qy2 + qy3 + qy4) * 0.25
            q_vect = sqrt(qy_avg*qy_avg + q0*q0)
            # calculate flow
            n = 0.5 * (arr_n[r, c] + arr_n[r, c-1])
            qup = arrp_qw[rq, cq-1]
            qdown = arrp_qw[rq, cq+1]
            if hf <= hf_min:
                qw_new = 0
            else:
                term_1 = (theta * q0 + ((1 - theta) * 0.5) * (qup + qdown))
                term_2 = (g * hf * (dt / dx) * (wse0 - wseup))
                term_3 = (1 + g * dt * (n*n) * q_vect / pow(hf, 7./3.))
                qw_new = (term_1 - term_2) / term_3
            # If flow is going upstream, recalculate with Bates 2010
            if (wseup - wse0) * qw_new < 0:
                slope = (wse0 - wseup) / dx
                num = q0 - g * hf * dt * slope
                den = 1 + g * hf * dt * n*n * abs(q0) / pow(hf, 10./3.)
                qw_new = num / den
            # populate the array
            arr_qw_new[r,c] = qw_new

            #####################
            # flow in y dimension
            if r == 0:
                continue
            # calculate flow depth (hf)
            zup = arr_z[r-1, c]
            z0 = arr_z[r,c]
            hup = arr_h_old[r-1,c]
            h0 = arr_h_old[r,c]
            wseup = zup + hup
            wse0 = z0 + h0
            hf = max(wseup, wse0) - max(zup, z0)
            # q vector norm
            q0 = arrp_qn[rq, cq]
            qx1 = arrp_qw[rq-1, cq-1]
            qx2 = arrp_qw[rq-1, cq]
            qx3 = arrp_qw[rq, cq-1]
            qx4 = arrp_qw[rq, cq]
            qx_avg = (qx1 + qx2 + qx3 + qx4) * 0.25
            q_vect = sqrt(qx_avg*qx_avg + q0*q0)
            # calculate flow
            n = 0.5 * (arr_n[r, c] + arr_n[r-1, c])
            qup = arrp_qn[rq-1, cq]
            qdown = arrp_qn[rq+1, cq]
            if hf <= hf_min:
                qn_new = 0
            else:
                term_1 = (theta * q0 + ((1 - theta) * 0.5) * (qup + qdown))
                term_2 = (g * hf * (dt / dy) * (wse0 - wseup))
                term_3 = (1 + g * dt * (n*n) * q_vect / pow(hf, 7./3.))
                qn_new = (term_1 - term_2) / term_3
            # If flow is going upstream, recalculate with Bates 2010
            if (wseup - wse0) * qn_new < 0:
                slope = (wse0 - wseup) / dy
                num = q0 - g * hf * dt * slope
                den = 1 + g * hf * dt * n*n * abs(q0) / pow(hf, 10./3.)
                qn_new = num / den
            arr_qn_new[r,c] = qn_new
    return 0

def solve_q_loop2(
        np.ndarray[DTYPE_t, ndim=2] arr_z0, np.ndarray[DTYPE_t, ndim=2] arr_z1,
        np.ndarray[DTYPE_t, ndim=2] arr_n0, np.ndarray[DTYPE_t, ndim=2] arr_n1,
        np.ndarray[DTYPE_t, ndim=2] arr_h0, np.ndarray[DTYPE_t, ndim=2] arr_h1,
        np.ndarray[DTYPE_t, ndim=2] arr_q0, np.ndarray[DTYPE_t, ndim=2] arr_q1,
        np.ndarray[DTYPE_t, ndim=2] arr_qm1, np.ndarray[DTYPE_t, ndim=2] arr_qnorm,
        np.ndarray[DTYPE_t, ndim=2] arr_q0_new,
        float dt, float cell_len, float g, float theta, float hf_min):
    '''Solve flow equation, including hflow, using
    loop through the domain
    '''
    assert arr_z0.dtype == arr_z1.dtype == arr_n0.dtype == \
        arr_n1.dtype == arr_h0.dtype == arr_h1.dtype == \
        arr_q0.dtype == arr_q1.dtype == arr_qm1.dtype == \
        arr_qnorm.dtype == arr_q0_new.dtype == DTYPE
    cdef unsigned int rmax, cmax, r, c
    cdef float z1, z0, h1, h0, wse1, wse0, hf
    cdef float q0, qup, qdown, q_vect, n
    cdef float term_1, term_2, term_3, q0_new, slope, num, den

    rmax = arr_z0.shape[0]
    cmax = arr_z0.shape[1]
    for r in xrange(rmax):
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

            # q
            q0 = arr_q0[r, c]
            qup = arr_qm1[r, c]
            qdown = arr_q1[r, c]
            # q_vect
            q_vect = arr_qnorm[r, c]

            # calculate flow
            n = 0.5 * (arr_n0[r, c] + arr_n1[r, c])
            if hf <= hf_min:
                q0_new = 0
            else:
                term_1 = (theta * q0 + ((1 - theta) * 0.5) * (qup + qdown))
                term_2 = (g * hf * (dt / cell_len) * (wse1 - wse0))
                term_3 = (1 + g * dt * (n*n) * q_vect / pow(hf, 7./3.))
                q0_new = (term_1 - term_2) / term_3
            # If flow is going upstream, recalculate with Bates 2010
            if (wse0 - wse1) * q0_new < 0:
                slope = (wse1 - wse0) / cell_len
                num = q0 - g * hf * dt * slope
                den = 1 + g * hf * dt * n*n * abs(q0) / pow(hf, 10./3.)
                q0_new = num / den
            # populate the array
            arr_q0_new[r,c] = q0_new
    return 0
