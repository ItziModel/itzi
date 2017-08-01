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

cdef float PI = 3.1415926535898
cdef float FOOT = 0.3048

# coefficients taken from Rubinato et al. (2017)
# http://doi.org/10.1016/j.jhydrol.2017.06.024
cdef float ORIFICE_COEFF = 0.167
cdef float FREE_WEIR_COEFF = 0.54
cdef float SUBMERGED_WEIR_COEFF = 0.056

ctypedef np.float32_t F32_t
ctypedef np.int32_t I32_t

ctypedef struct link_struct:
    I32_t idx
    F32_t flow
    F32_t depth
    F32_t velocity
    F32_t volume
    I32_t type
    F32_t offset1
    F32_t offset2
    F32_t full_depth
    F32_t froude

ctypedef struct node_struct:
    I32_t idx
    I32_t linkage_type
    F32_t inflow
    F32_t outflow
    F32_t linkage_flow
    F32_t head
    F32_t crest_elev
    I32_t type
    I32_t sub_index
    F32_t invert_elev
    F32_t init_depth
    F32_t full_depth
    F32_t sur_depth
    F32_t ponded_area
    I32_t degree
    F32_t crown_elev
    F32_t losses
    F32_t volume
    F32_t full_volume
    F32_t overflow
    F32_t depth
    F32_t lat_flow
    I32_t row
    I32_t col

cdef extern from "source/headers.h" nogil:
    ctypedef struct linkData:
        double flow
        double depth
        double velocity
        double volume
        int    type
        double offset1
        double offset2
        double yFull
        double froude
    ctypedef struct nodeData:
        double inflow
        double outflow
        double head
        double crestElev
        int    type
        int    subIndex
        double invertElev
        double initDepth
        double fullDepth
        double surDepth
        double pondedArea
        int    degree
        char   updated
        double crownElev
        double losses
        double newVolume
        double fullVolume
        double overflow
        double newDepth
        double newLatFlow
    # functions
    double node_getSurfArea(int j, double d)
    int    project_findObject(int type, char* id)
    # exported values
    double MinSurfArea

cdef extern from "source/swmm5.h" nogil:
    int swmm_getNodeID(int index, char* id)
    int swmm_getLinkID(int index, char* id)
    int swmm_getNodeData(int index, nodeData* data)
    int swmm_getLinkData(int index, linkData* data)
    int swmm_addNodeInflow(int index, double inflow)
    int swmm_setNodeFullDepth(int index, double depth)
    int swmm_setNodePondedArea(int index, double area)

cdef enum linkage_types:
    NOT_LINKED
    NO_LINKAGE
    FREE_WEIR
    SUBMERGED_WEIR
    ORIFICE


def get_object_index(int obj_type_code, bytes object_id):
    """return the index of an object for a given ID and type
    """
    cdef char* c_obj_id
    c_obj_id = object_id
    return project_findObject(obj_type_code, c_obj_id)


def set_ponding_area(int node_idx):
    """Set the ponded area equal to node area.
    SWMM internal ponding don't have meaning any more with the 2D coupling
    The ponding depth is used to keep the node head consistent with
    the WSE of the 2D model.
    """
    cdef float ponding_area
    swmm_setNodePondedArea(node_idx, MinSurfArea)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def update_links(link_struct[:] arr_links):
    """get current values from the C objects and put them in the
    corresponding array
    """
    cdef int r, rmax, link_idx
    cdef linkData link_data
    cdef char* link_id
    cdef link_struct link
    rmax = arr_links.shape[0]
    # those operations are not thread safe
    for r in range(rmax):
        # get values
        link = arr_links[r]
        link_idx = link.idx
        swmm_getLinkData(link_idx, &link_data)
        # data type
        link.type = link_data.type

        # assign values
        link.flow = link_data.flow * FOOT ** 3
        link.depth = link_data.depth * FOOT
        link.velocity = link_data.velocity
        link.volume = link_data.volume * FOOT ** 3
        link.offset1 = link_data.offset1 * FOOT
        link.offset2 = link_data.offset2 * FOOT
        link.full_depth = link_data.yFull * FOOT
        link.froude = link_data.froude

        # update array
        arr_links[r] = link


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def update_nodes(node_struct[:] arr_node):

    """get current values from the C objects and put them in the
    corresponding array
    """
    cdef int r, rmax
    cdef nodeData node_data
    cdef char* node_id
    cdef node_struct node
    rmax = arr_node.shape[0]
    # those operations are not thread safe
    for r in range(rmax):
        # get values
        node = arr_node[r]
        swmm_getNodeData(node.idx, &node_data)
        # data type
        node.type = node_data.type

        # assign values
        node.sub_index = node_data.subIndex
        node.degree = node_data.degree

        # translate to SI units
        node.invert_elev = node_data.invertElev * FOOT
        node.init_depth  = node_data.initDepth * FOOT
        node.full_depth  = node_data.fullDepth * FOOT
        node.depth       = node_data.newDepth * FOOT
        node.sur_depth   = node_data.surDepth * FOOT
        node.crown_elev  = node_data.crownElev * FOOT
        node.head        = node_data.head * FOOT
        node.crest_elev  = node_data.crestElev * FOOT

        node.ponded_area = node_data.pondedArea * FOOT ** 2
        node.volume      = node_data.newVolume * FOOT ** 3
        node.full_volume = node_data.fullVolume * FOOT ** 3

        node.inflow      = node_data.inflow * FOOT ** 3
        node.outflow     = node_data.outflow * FOOT ** 3
        node.losses      = node_data.losses * FOOT ** 3
        node.overflow    = node_data.overflow * FOOT ** 3
        node.lat_flow    = node_data.newLatFlow * FOOT ** 3
        # update array
        arr_node[r] = node


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def apply_linkage_flow(node_struct[:] arr_node,
                       F32_t[:,:] arr_h, F32_t[:,:] arr_z, F32_t[:,:] arr_qdrain,
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
    cdef int i, imax, linkage_type, row, col
    cdef float crest_elev, weir_width, overflow_area
    cdef float dh, new_linkage_flow, maxflow
    cdef float wse, z, qdrain
    cdef node_struct node

    imax = arr_node.shape[0]
#~     for i in prange(imax, nogil=True):
    for i in range(imax,):
        node = arr_node[i]
        # don't do anything if the node is not linked
        if node.linkage_type == linkage_types.NOT_LINKED:
            continue

        # corresponding grid coordinates at drainage node
        row = node.row
        col = node.col
        # values on the surface
        z = arr_z[row, col]
        wse = z + arr_h[row, col]

        # the actual crest elevation should be equal to DEM
        if node.crest_elev != z:
            full_depth = z - node.invert_elev
            swmm_setNodeFullDepth(node.idx, full_depth)
            crest_elev = z
        else:
            crest_elev = node.crest_elev

        ## linkage type ##
        overflow_area = get_overflow_area(node.idx, node.depth)
        # weir width is the circumference (node considered circular)
        weir_width = PI * 2. * c_sqrt(overflow_area / PI)
        # determine linkage type
        linkage_type = get_linkage_type(wse, crest_elev, node.head,
                                        weir_width, overflow_area)

        ## linkage flow ##
        new_linkage_flow = get_linkage_flow(wse, node.head, weir_width,
                                            crest_elev, linkage_type,
                                            overflow_area, g)

        ## flow limiter ###
        # flow leaving the 2D domain can't drain the corresponding cell
        if new_linkage_flow < 0:
            dh = wse - max(crest_elev, node.head)
            maxflow = dh * cell_surf * dt2d
            new_linkage_flow = max(new_linkage_flow, -maxflow)
        # flow leaving the drainage can't be higher than the water column above wse
        elif new_linkage_flow > 0:
            dh = node.head - wse
            maxflow = dh * overflow_area * dt1d
            new_linkage_flow = min(new_linkage_flow, maxflow)

        # apply flow to 2D model and drainage model
        arr_qdrain[row, col] = new_linkage_flow
        swmm_addNodeInflow(node.idx, - new_linkage_flow / FOOT ** 3)
        # update node array
        node.linkage_type = linkage_type
        node.linkage_flow = new_linkage_flow
        arr_node[i] = node


cdef int get_linkage_type(float wse, float crest_elev,
                          float node_head, float weir_width,
                          float overflow_area) nogil:
    """
    """
    cdef int linkage_type

    if wse <= crest_elev and node_head <= crest_elev:
        linkage_type = linkage_types.NO_LINKAGE
    elif ((node_head > wse) or (wse - crest_elev) >= (overflow_area / weir_width)):
        linkage_type = linkage_types.ORIFICE
    elif wse > crest_elev > node_head:
        linkage_type = linkage_types.FREE_WEIR
    elif (node_head >= crest_elev and wse > crest_elev and
          ((wse - crest_elev) < (overflow_area / weir_width))):
        linkage_type = linkage_types.SUBMERGED_WEIR
    return linkage_type


cdef float get_linkage_flow(float wse, float node_head,
                            float weir_width, float crest_elev,
                            int linkage_type, float overflow_area,
                            float g):
    """flow sign is :
            - negative when entering the drainage (leaving the 2D model)
            - positive when leaving the drainage (entering the 2D model)
    """
    cdef float head_up, head_down, head_diff
    cdef float weir_coeff, orifice_coeff, upstream_depth, unsigned_q

    head_up = max(wse, node_head)
    head_down = min(wse, node_head)
    head_diff = head_up - head_down
    upstream_depth = head_up - crest_elev

    # calculate the flow
    if linkage_type == linkage_types.NO_LINKAGE:
        unsigned_q = 0.

    elif linkage_type == linkage_types.FREE_WEIR:
        unsigned_q = ((2./3.) * FREE_WEIR_COEFF * weir_width *
                      c_pow(upstream_depth, 3/2.) *
                      c_sqrt(2. * g))

    elif linkage_type == linkage_types.SUBMERGED_WEIR:
        unsigned_q = (SUBMERGED_WEIR_COEFF * weir_width * upstream_depth *
                      c_sqrt(2. * g * head_diff))

    elif linkage_type == linkage_types.ORIFICE:
        unsigned_q = ORIFICE_COEFF * overflow_area * c_sqrt(2. * g * head_diff)

    # assign flow sign
    return c_copysign(unsigned_q, node_head - wse)


cdef float get_overflow_area(int node_idx, float node_depth):
    """return overflow area defauting to MinSurfArea
    """
    cdef float overflow_area, surf_area
    surf_area = node_getSurfArea(node_idx, node_depth)
    if surf_area <= 0.:
        overflow_area = MinSurfArea
    return overflow_area * FOOT ** 2
