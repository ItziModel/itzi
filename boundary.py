#! /usr/bin/python
# coding=utf8

"""
COPYRIGHT:    (C) 2015 by Laurent Courty

              This program is free software under the GNU General Public
              License. Read the LICENSE file for details.
"""
from __future__ import division
import numpy as np

class Boundary(object):
    """
    A boundary of the computation domain
    """
    def __init__(self, cell_width, cell_length, boundary_pos):
        self.pos = boundary_pos
        self.cw = cell_width
        self.cl = cell_length
        if self.pos in ('W', 'N'):
            self.postype = 'upstream'
        elif self.pos in ('E', 'S'):
            self.postype = 'downstream'
        else:
            assert False, "Unknown boundary position: {}".format(self.pos)

    def get_boundary_flow(self, qin, qboundary, hflow, n, z, depth,
                            bctype, bcvalue):
        """Take 1D numpy arrays as input
        Return an updated 1D array of flow through the boundary
        """
        # check if all input arrays have the same size
        assert (qin.shape == qboundary.shape == hflow.shape == n.shape ==
                z.shape == depth.shape == bctype.shape == bcvalue.shape)
        # select slices according to boundary types
        slice_closed = np.where(bctype == 1)
        slice_open = np.where(bctype == 2)
        slice_wse = np.where(bctype == 3)
        # Boundary type 1 (closed)
        qboundary[slice_closed] = 0
        # Boundary type 2 (open)
        qboundary[slice_open] = self.get_flow_open_boundary(qin[slice_open],
                                hflow[slice_open], depth[slice_open])
        # Boundary type 3 (user-defined wse)
        slope = self.get_slope(depth[slice_wse],
                            z[slice_wse], bcvalue[slice_wse])
        qboundary[slice_wse] = self.get_flow_wse_boundary(n[slice_wse],
                                hflow[slice_wse], slope)
        return self

    def get_flow_open_boundary(self, qin, hf, hf_boundary):
        """velocity at the boundary equal to velocity inside domain
        """
        return qin / hf * hf_boundary

    def get_slope(self, h, z, user_wse):
        """Return the slope between two water surface elevation,
        invert the results (and hence the flow) if a downstream boundary
        """
        slope = (user_wse - h + z) / self.cl
        if self.postype == 'upstream':
            return slope
        elif self.postype == 'downstream':
            return - slope
        else:
            assert False, "Unknown postype {}".format(self.postype)

    def get_flow_wse_boundary(self, n, hf, slope):
        """
        Gauckler-Manning-Strickler flow equation
        """
        v = (1/n) * np.power(hf, 2/3) * np.power(slope, 1/2)
        return v * hf * self.cw
