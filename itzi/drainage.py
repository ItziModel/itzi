# coding=utf8
"""
Copyright (C) 2016-2017 Laurent Courty

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
from datetime import datetime, timedelta
from collections import namedtuple
import numpy as np
import networkx as nx

from swmm import swmm
from itzi_error import DtError


class DrainageSimulation(object):
    """manage simulation of the pipe network
    write results to RasterDomain object
    """
    GridCoords = namedtuple('GridCoords', ['row', 'col'])
    def __init__(self, domain, inp, igis):
        self.dom = domain
        # create swmm object and open files
        self.swmm5 = swmm.Swmm5()
        self.swmm5.swmm_open(input_file=inp,
                             report_file=os.devnull,
                             output_file='')
        self.swmm5.swmm_start()
        # allow ponding
        self.swmm5.set_allow_ponding()

        self.cell_surf = igis.dx * igis.dy
        self.gis = igis

        # create a list of linkable nodes
        swmm_inp = swmm.SwmmInputParser(inp)
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

    def create_node_list(self, j_dict):
        """create list of drainage nodes objects with
        corresponding grid coordinates
        """
        self.drain_nodes = []
        for k, coords in j_dict.iteritems():
            # create Node object
            node = swmm.SwmmNode(swmm_object=self.swmm5, node_id=k,
                                 coordinates=coords)
            if self.gis.is_in_region(coords.x, coords.y):
                # calculate grid coordinates and add them to the object
                row, col = self.gis.coor2pixel((coords.x, coords.y))
                node.grid_coords = DrainageSimulation.GridCoords(int(row),
                                                                 int(col))
            # add to list of nodes
            self.drain_nodes.append(node)
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
        arr_qd = self.dom.get('n_drain')
        for node in self.drain_nodes:
            node.update()
            # only apply if the node is inside the domain
            if node.grid_coords:
                row, col = node.grid_coords
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

    def get_serialized_project_values(self):
        """Return a dict of general drainage project values
        """
        return {}

    def get_serialized_nodes_values(self):
        """Return nodes values in a ID: values dict
        """
        return {n.node_id: n.get_values_as_dict()
                for n in self.drain_nodes}

    def get_serialized_links_values(self):
        """Return links values in a ID:values dict
        """
        return {}


class NetworkResultsWriter(object):
    """Generate a networkx object to represent the drainage network.
    """
    def __init__(self, swmm_object):
        """swmm5 is a swmm5 simulation object
        """
        self.swmm5 = swmm_object

