# coding=utf8
"""
Copyright (C) 2016  Laurent Courty

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
from datetime import datetime, timedelta
from collections import namedtuple
import numpy as np

import grass.pygrass.utils as gutils

from swmm import swmm
from itzi_error import DtError

class DrainageSimulation(object):
    """manage simulation of the pipe network
    write results to RasterDomain object
    """
    def __init__(self, domain, swmm_params, igis):
        self.dom = domain
        # create swmm object and open files
        self.swmm5 = swmm.Swmm5()
        self.swmm5.swmm_open(input_file=swmm_params['input'],
                            report_file=swmm_params['report'],
                            output_file=swmm_params['output'])
        self.swmm5.swmm_start()
        # allow ponding
        self.swmm5.set_allow_ponding()
        # Raster computational domain
        self.bbox = igis.reg_bbox
        self.cell_surf = igis.dx * igis.dy
        self.gis = igis

        # create a list of linkable nodes
        swmm_inp = swmm.SwmmInputParser(swmm_params['input'])
        self.create_node_list(swmm_inp.get_juntions_as_dict())

    def __del__(self):
        self.swmm5.swmm_end()
        self.swmm5.swmm_close()

    def solve_dt(self):
        """Get the time-step from swmm object
        """
        old = self.swmm5.get_NewRoutingTime()
        new = self.swmm5.get_OldRoutingTime()
        self._dt = new - old
        if self._dt <= 0:
            self._dt = self.swmm5.routing_getRoutingStep()
        return self

    @property
    def dt(self):
        return timedelta(seconds=self._dt)

    @dt.setter
    def dt(self, value):
        raise DtError("Can't set time-step of a SWMM simulation")

    def is_in_region(self, x, y):
        """For a given coordinate pair(x, y),
        return True is inside raster region, False otherwise.
        """
        bool_x = (self.bbox['w'] < x < self.bbox['e'])
        bool_y = (self.bbox['s'] < y < self.bbox['n'])
        if bool_x and bool_y:
            return True
        else:
            return False

    def create_node_list(self, j_dict):
        """create list of drainage nodes objects with
        corresponding grid coordinates
        """
        self.drain_nodes = []
        for k, n in j_dict.iteritems():
            if self.is_in_region(n.x, n.y):
                node = swmm.SwmmNode(swmm_object=self.swmm5, node_id=k)
                row, col = self.gis.coor2pixel((n.x, n.y))
                self.drain_nodes.append((node, int(row), int(col)))
        return self

    def step(self):
        """Run a swmm time-step
        calculate the exchanges with raster domain
        """
        self.swmm5.swmm_step()
        return self

    def apply_linkage(self, dt2d):
        """For each linked node,
        calculate the flow entering or leaving the drainage network
        Apply the flow to the node and to the relevant raster cell
        """
        arr_h = self.dom.get('h')
        arr_z = self.dom.get('z')
        arr_qd = self.dom.get('q_drain')
        for node, row, col in self.drain_nodes:
            node.update()
            h = arr_h[row, col]
            z = arr_z[row, col]
            wse = h + z
            node.set_crest_elev(z)
            node.set_pondedArea()
            node.set_linkage_flow(wse, self.cell_surf, dt2d, self._dt)
            node.add_inflow(-node.linkage_flow)
            # apply flow in m/s to array
            arr_qd[row, col] = node.linkage_flow / self.cell_surf
        return self
