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

    # define namedtuples
    LayerDescr = namedtuple('LayerDescr', ['table_suffix', 'cols', 'layer_number'])
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
        # geo information
        self.cell_surf = igis.dx * igis.dy
        self.gis = igis

        # definition of linking_elements (used for GRASS vector writing)
        node_col_def = swmm.SwmmNode.get_sql_columns_def()
        link_col_def = swmm.SwmmLink.get_sql_columns_def()
        self.linking_elements = {'node': self.LayerDescr(table_suffix='_node',
                                                         cols=node_col_def,
                                                         layer_number=1),
                                 'link': self.LayerDescr(table_suffix='_link',
                                                         cols=link_col_def,
                                                         layer_number=2)}

        # create a graph made of drainage nodes and links objects
        swmm_inp = swmm.SwmmInputParser(inp)
        node_dict = self.get_node_object_dict(swmm_inp.get_nodes_id_as_dict())
        link_list = self.get_link_object_list(swmm_inp.get_links_id_as_dict())
        self.create_drainage_network_graph(node_dict, link_list)

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

    def get_node_object_dict(self, n_dict):
        """create dict id:SwmmNode object
        """
        drain_nodes = {}
        for k, coords in n_dict.iteritems():
            # create Node object
            node = swmm.SwmmNode(swmm_object=self.swmm5, node_id=k,
                                 coordinates=coords)
            if coords:
                if self.gis.is_in_region(coords.x, coords.y):
                    # calculate grid coordinates and add them to the object
                    row, col = self.gis.coor2pixel((coords.x, coords.y))
                    node.grid_coords = self.GridCoords(int(row), int(col))
            # populate dict
            drain_nodes[k] = node
        return drain_nodes

    def get_link_object_list(self, lnk_dict):
        """Take a dict of objects definition as input.
        Return a list of SwmmLink objects
        """
        drain_links = []
        for k, values in lnk_dict.iteritems():
            drain_link = swmm.SwmmLink(swmm_object=self.swmm5, link_id=k)
            drain_link.vertices = values.vertices
            drain_link.start_node_id = values.in_node
            drain_link.end_node_id = values.out_node
            drain_links.append(drain_link)
        return drain_links

    def create_drainage_network_graph(self, node_dict, link_list):
        """create a networkx object using given links and nodes lists
        """
        self.drainage_network = nx.MultiDiGraph()
        self.drainage_network.add_nodes_from(node_dict.values())
        for link in link_list:
            link.update()
            in_node = link.start_node_id
            out_node = link.end_node_id
            self.drainage_network.add_edge(node_dict[in_node],
                                           node_dict[out_node],
                                           object=link)
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
        for node in self.drainage_network.nodes():
            node.update()
            # only apply if the node is inside the domain
            if node.is_linkable():
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
