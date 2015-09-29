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

    def get_boundary_flow(self, qold, qnew, hflow, n, z, depth,
                            bctype, bcvalue):
        """Take 1D numpy array as input
        """

        slice_closed = np.where(bctype == 1)
        slice_open = np.where(bctype == 2)
        slice_wse = np.where(bctype == 3)
        # Boundary type 1 (closed)
        qnew[slice_closed] = 0
        # Boundary type 2 (open)
        qnew[slice_open] = self.open_boundary(qold[slice_open],
                                hflow[slice_open], depth[slice_open])
        # Boundary type 3 (user-defined wse)
        

    def open_boundary(self, qold, hf, hf_boundary):
        """Return flow through boundary
        """
        return qold / hf * hf_boundary

    def get_slope(self):
        pass

    def wse_boundary(self, n, hf, slope):
        """
        Gauckler-Manning-Strickler flow equation
        """
        v = (1/n) * pow(hf, 2/3) * pow(slope, 1/2)
        return v * hf * self.cw
