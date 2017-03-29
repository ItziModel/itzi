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
from libc.math cimport copysign as c_copysign

ctypedef np.float32_t F32_t
ctypedef np.int32_t I32_t
ctypedef char* C30

cdef float PI = 3.1415926535898
cdef float FOOT = 0.3048
cdef float WEIR_LENGTH = 0.1
cdef float ORIFICE_COEFF = 0.4

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
    float node_getSurfArea(int j, double d)

cdef extern from "source/swmm5.h" nogil:
    int swmm_getNodeData(int index, nodeData* data)
    int swmm_getLinkData(int index, linkData* data)
    int swmm_addNodeInflow(int index, double inflow)

cdef enum linkage_types:
    NOT_LINKED
    NO_LINKAGE
    FREE_WEIR
    SUBMERGED_WEIR
    ORIFICE

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


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def apply_linkage_flow(I32_t[:] arr_node_id, F32_t[:] arr_crest_elev,
                       F32_t[:] arr_depth, F32_t[:] arr_head,
                       I32_t[:] arr_row, I32_t[:] arr_col,
                       F32_t[:,:] arr_wse,
                       I32_t[:] arr_linkage_type, F32_t[:,:] arr_qdrain,
                       float cell_surf, float dt2d, float dt1d, float g):
    '''select the linkage type then calculate the flow between
    surface and drainage models (Chen et al. 2007)
    flow sign is :
     - negative when entering the drainage (leaving the 2D model)
     - positive when leaving the drainage (entering the 2D model)
    cell_surf: area in m² of the cell above the node
    dt2d: time-step of the 2d model in seconds
    dt1d: time-step of the drainage model in seconds
    '''
    cdef int i, imax, node_id, linkage_type, row, col
    cdef float crest_elev, node_head, node_depth, weir_width, overflow_area
    cdef float water_surf_up, water_surf_down, water_surf_diff, upstream_depth
    cdef float weir_coeff, orif_coeff, dh, new_linkage_flow, maxflow
    cdef float wse, unsigned_q, qdrain
    imax = arr_node_id.shape[0]
    for i in prange(imax, nogil=True):
        # read values of node
        node_id = arr_node_id[i]
        crest_elev = arr_crest_elev[i]
        node_head = arr_head[i]
        node_depth = arr_depth[i]
        row = arr_row[i]
        col = arr_col[i]
        # water level on the surface
        wse = arr_wse[row, col]

        ## linkage type ##
        overflow_area = node_getSurfArea(node_id, node_depth)
        # weir width is the circumference (node considered circular)
        weir_width = PI * 2. * c_sqrt(overflow_area / PI)

        if wse <= crest_elev and node_head <= crest_elev:
            linkage_type = linkage_types.NO_LINKAGE
        elif (wse > crest_elev > node_head or
              wse <= crest_elev < node_head):
            linkage_type = linkage_types.FREE_WEIR
        elif (node_head >= crest_elev and
              wse > crest_elev and
              ((wse - crest_elev) <
               (overflow_area / weir_width))):
            linkage_type = linkage_types.SUBMERGED_WEIR
        elif ((wse - crest_elev) >=
              (overflow_area / weir_width)):
            linkage_type = linkage_types.ORIFICE
        # populate array with linkage type
        arr_linkage_type[i] = linkage_type

        ## linkage flow ##
        water_surf_up = max(wse, node_head)
        water_surf_down = min(wse, node_head)
        water_surf_diff = water_surf_up - water_surf_down
        upstream_depth = water_surf_up - crest_elev
        weir_coeff = get_weir_coeff(upstream_depth)

        # calculate the flow
        if linkage_type == linkage_types.NO_LINKAGE:
            unsigned_q = 0.

        elif linkage_type == linkage_types.FREE_WEIR:
            unsigned_q = (weir_coeff * weir_width *
                          c_pow(upstream_depth, 3/2.) *
                          c_sqrt(2. * g))

        elif linkage_type == linkage_types.SUBMERGED_WEIR:
            unsigned_q = (weir_coeff * weir_width * upstream_depth *
                          c_sqrt(2. * g * water_surf_diff))

        elif linkage_type == linkage_types.ORIFICE:
            unsigned_q = (ORIFICE_COEFF * overflow_area *
                          c_sqrt(2. * g * water_surf_diff))
        else:
            unsigned_q = 0.

        # assign flow sign
        new_linkage_flow = c_copysign(unsigned_q, node_head - wse)

        # flow leaving the 2D domain can't drain the corresponding cell
        if new_linkage_flow < 0:
            dh = wse - max(crest_elev, node_head)
            maxflow = dh * cell_surf * dt2d
            new_linkage_flow = max(new_linkage_flow, -maxflow)
        # flow leaving the drainage can't be higher than the water column above wse
        elif new_linkage_flow > 0:
            dh = node_head - wse
            maxflow = dh * overflow_area * dt1d
            new_linkage_flow = min(new_linkage_flow, maxflow)

        # apply flow to 2D model and drainage model
        arr_qdrain[row, col] = new_linkage_flow
        swmm_addNodeInflow(i, - new_linkage_flow / FOOT ** 3)


cdef float get_weir_coeff(float upstream_depth) nogil:
    """Calculate the weir coefficient for linkage
    according to equation found in Bos (1985)
    upstream_depth: depth of water in the inflow element above crest_elev
                    (depends on the direction of the flow)
    """
    return 0.93 + 0.1 * upstream_depth / WEIR_LENGTH
