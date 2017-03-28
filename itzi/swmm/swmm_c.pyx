# coding=utf8

"""
Copyright (C) 2017 Laurent Courty

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
import os
cimport numpy as np
cimport cython
from cython.parallel cimport prange
import numpy as np
from libc.math cimport pow as c_pow
from libc.math cimport sqrt as c_sqrt
from libc.math cimport fabs as c_abs

# set paths
ROOT = '.'
DIR = os.path.join(ROOT, 'source')
HEADERS = os.path.join(DIR, 'headers.h')
H_SWMM5 = os.path.join(DIR, 'swmm5.h')

ctypedef np.float32_t F32_t
ctypedef np.int32_t I32_t
ctypedef char* C30

cdef float PI = 3.1415926535898
cdef float FOOT = 0.3048

cdef extern from "source/headers.h" nogil:
    ctypedef struct linkData:
        float flow
        float depth
        float velocity
        float volume
        int type
        float offset1
        float offset2
        float yFull
        float froude
    ctypedef struct nodeData:
        float inflow
        float outflow
        float head
        float crestElev
        int   type
        int   subIndex
        float invertElev
        float initDepth
        float fullDepth
        float surDepth
        float pondedArea
        int   degree
        char  updated
        float crownElev
        float losses
        float newVolume
        float fullVolume
        float overflow
        float newDepth
        float newLatFlow

cdef extern from "source/swmm5.h" nogil:
    int swmm_getNodeData(int index, nodeData* data)
    int swmm_getLinkData(int index, linkData* data)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def update_links(I32_t[:] link_id, F32_t[:] flow, F32_t[:] depth,
                 F32_t[:] velocity, F32_t[:] volume, I32_t[:] link_type,
                 F32_t[:] start_node_offset, F32_t[:] end_node_offset,
                 F32_t[:] full_depth, F32_t[:] froude):
    """get current values from the C objects and put them in the
    corresponding array
    """
    cdef int r, rmax
    cdef linkData link_data
    rmax = link_id.shape[0]
    for r in prange(rmax, nogil=True):
        # get values
        swmm_getLinkData(link_id[r], &link_data)
        # data type
        link_type[r] = link_data.type

        # affect values
        flow[r] = link_data.flow * FOOT ** 3
        depth[r] = link_data.depth * FOOT
        velocity[r] = link_data.velocity
        volume[r] = link_data.volume * FOOT ** 3
        start_node_offset[r] = link_data.offset1 * FOOT
        end_node_offset[r] = link_data.offset2 * FOOT
        full_depth[r] = link_data.yFull * FOOT
        froude[r] = link_data.froude


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def update_nodes(I32_t[:] node_id, F32_t[:] inflow, F32_t[:] outflow, F32_t[:] head,
                 F32_t[:] crest_elev, I32_t[:] node_type, I32_t[:] sub_index,
                 F32_t[:] invert_elev, F32_t[:] init_depth,
                 F32_t[:] full_depth, F32_t[:] sur_depth,
                 F32_t[:] ponded_area, I32_t[:] degree,
                 F32_t[:] crown_elev, F32_t[:] losses, F32_t[:] volume,
                 F32_t[:] full_volume, F32_t[:] overflow,
                 F32_t[:] depth, F32_t[:] lat_flow):

    """get current values from the C objects and put them in the
    corresponding array
    """
    cdef int r, rmax
    cdef nodeData node_data
    rmax = node_id.shape[0]
    for r in prange(rmax, nogil=True):
        # get values
        swmm_getNodeData(node_id[r], &node_data)
        # data type
        node_type[r] = node_data.type

        # affect values
        sub_index[r] = node_data.subIndex
        degree[r] = node_data.degree

        # translate to SI units
        invert_elev[r] = node_data.invertElev * FOOT
        init_depth[r]  = node_data.initDepth * FOOT
        full_depth[r]  = node_data.fullDepth * FOOT
        depth[r]       = node_data.newDepth * FOOT
        sur_depth[r]   = node_data.surDepth * FOOT
        crown_elev[r]  = node_data.crownElev * FOOT
        head[r]        = node_data.head * FOOT
        crest_elev[r]  = node_data.crestElev * FOOT

        ponded_area[r] = node_data.pondedArea * FOOT ** 2
        volume[r]      = node_data.newVolume * FOOT ** 3
        full_volume[r] = node_data.fullVolume * FOOT ** 3

        inflow[r]      = node_data.inflow * FOOT ** 3
        outflow[r]     = node_data.outflow * FOOT ** 3
        losses[r]      = node_data.losses * FOOT ** 3
        overflow[r]    = node_data.overflow * FOOT ** 3
        lat_flow[r]    = node_data.newLatFlow * FOOT ** 3


