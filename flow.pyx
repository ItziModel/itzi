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
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def solve_q(np.ndarray[np.int8_t, ndim=2] arr_dir,
        np.ndarray[DTYPE_t, ndim=2] arr_z0, np.ndarray[DTYPE_t, ndim=2] arr_z1,
        np.ndarray[DTYPE_t, ndim=2] arr_n0, np.ndarray[DTYPE_t, ndim=2] arr_n1,
        np.ndarray[DTYPE_t, ndim=2] arr_h0, np.ndarray[DTYPE_t, ndim=2] arr_h1,
        np.ndarray[DTYPE_t, ndim=2] arr_q0, np.ndarray[DTYPE_t, ndim=2] arr_q1,
        np.ndarray[DTYPE_t, ndim=2] arr_qm1, np.ndarray[DTYPE_t, ndim=2] arr_qnorm,
        np.ndarray[DTYPE_t, ndim=2] arr_q0_new, np.ndarray[DTYPE_t, ndim=2] arr_hf,
        float dt, float cell_len, float g, float theta, float hf_min, float v_rout):
    '''Solve flow equation, including hflow, using
    loop through the domain
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

                # calculate flow
                n = 0.5 * (arr_n0[r, c] + arr_n1[r, c])
                slope = (wse1 - wse0) / cell_len
                # rain routing flow going W or N
                if qdir == 0 and wse1 > wse0 and hf < hf_min:
                    q0_new = - rain_routing(h1, wse1, wse0, dt,
                                cell_len, v_rout)
                # rain routing flow going E or S
                elif qdir == 1 and wse0 > wse1 and hf < hf_min:
                    q0_new = rain_routing(h0, wse0, wse1, dt,
                                cell_len, v_rout)
                # Almeida 2013
                elif hf > hf_min:
                    term_1 = theta * q0 + (1 - theta) * (qup + qdown) * 0.5
                    term_2 = g * hf * dt * slope
                    # If flow direction is not coherent with surface slope,
                    # use only previous flow, i.e. ~switch to Bates 2010
                    if term_1 * term_2 > 0:
                        term_1 = q0
                        # q_vect is calculated, why not using it ?
                        # q_vect = c_abs(q0)
                    term_3 = 1 + g * dt * (n*n) * q_vect / c_pow(hf, 7./3.)
                    q0_new = (term_1 - term_2) / term_3
                else:
                    q0_new = 0
                # populate the array
                arr_q0_new[r,c] = q0_new

#~ @cython.wraparound(False)  # Disable negative index check
#~ @cython.cdivision(True)  # Don't check division by zero
#~ @cython.boundscheck(False)  # turn of bounds-checking for entire function
#~ cdef float routing_flow(float h0, float h1, float z0, float z1,
#~     float cell_length, float v_routing, float dt) nogil:
#~     '''Return a routing flow in m2/s
#~     '''
#~     cdef float dh
#~     # fraction of the depth to be routed
#~     dh = (z0 + h0) - (z1 + h1)
#~     # if WSE of neighbour is below the dem of the current cell, set to h0
#~     dh = min(dh, h0)
#~     # don't allow reverse flow
#~     dh = max(dh, 0.)
#~     # prevent over-drainage of the cell in case of long time-step
#~     if v_routing * dt > cell_length:
#~         v_routing = cell_length / dt
#~     return dh * cell_length * v_routing

@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef float rain_routing(float h0, float wse0, float wse1, float dt,
                 float cell_len, float v_routing) nogil:
    """Calculate flow routing at a face in m2/s
    Cf. Sampson et al. (2013)
    """
    cdef float maxflow, q_routing
    cdef float dh = wse0 - wse1
    # fraction of the depth to be routed
    dh = wse0 - wse1
    # make sure it's positive (should not happend)
    dh = max(dh, 0)
    # if WSE of direction cell is below the dem of the drained cell, set to h0
    dh = min(dh, h0)
    maxflow = cell_len * dh / dt
    q_routing = dh * v_routing
    q_routing = min(q_routing, maxflow)
    return q_routing

@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def route_rain(np.ndarray[np.int8_t, ndim=2] arr_dir,
        np.ndarray[DTYPE_t, ndim=2] arr_h0, np.ndarray[DTYPE_t, ndim=2] arr_h1,
        np.ndarray[DTYPE_t, ndim=2] arr_z0, np.ndarray[DTYPE_t, ndim=2] arr_z1,
        np.ndarray[DTYPE_t, ndim=2] arr_hf, np.ndarray[DTYPE_t, ndim=2] arr_q_new,
        float cell_len, float v_rout, float dt, float hf_min):
    '''assign routing flow to flow array
    '''
    cdef float h0, h1, z0, z1, wse0, wse1, hf, rout_q
    cdef int rmax, cmax, r, c, qdir
    rmax = arr_q_new.shape[0]
    cmax = arr_q_new.shape[1]
    with nogil:
        for r in prange(rmax):
            for c in range(cmax):
                h0 = arr_h0[r, c]
                h1 = arr_h1[r, c]
                z0 = arr_z0[r, c]
                z1 = arr_z1[r, c]
                wse0 = h0 + z1
                wse1 = h1 + z1
                hf = arr_hf[r, c]
                qdir = arr_dir[r, c]

                # only where flow depth under threshold
                if hf <= hf_min:
                    # flow going upstream, i.e. 'left', i.e. negative value
                    if h1 > h0 and wse1 > wse0:
                        rout_q = - rain_routing(h1, wse1, wse0, dt,
                                    cell_len, v_rout)
                        arr_q_new[r, c] = rout_q
                    # flow going downstream, i.e. 'right', i.e. positive value
                    elif h0 > h1 and wse0 > wse1:
                        rout_q = rain_routing(h0, wse0, wse1, dt,
                                    cell_len, v_rout)
                        arr_q_new[r, c] = rout_q
