# coding=utf8
"""
Copyright (C) 2016-2020 Laurent Courty

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
from __future__ import absolute_import
import os
import math
from datetime import timedelta
from collections import namedtuple
from collections import defaultdict

foot = 0.3048


class LinkageTypes:
    NOT_LINKED = 0
    NO_LINKAGE = 1
    FREE_WEIR = 2
    SUBMERGED_WEIR = 3
    ORIFICE = 4


class DrainageSimulation():
    """manage simulation of the pipe network
    write results to RasterDomain object
    """

    # define namedtuples
    LayerDescr = namedtuple('LayerDescr', ['table_suffix', 'cols', 'layer_number'])
    GridCoords = namedtuple('GridCoords', ['row', 'col'])

    def __init__(self, raster_domain, pyswmm_sim, drainage_params, linked_node_list, g):
        self.dom = raster_domain
        self.g = g
        self.orifice_coeff = drainage_params['orifice_coeff']
        self.free_weir_coeff = drainage_params['free_weir_coeff']
        self.submerged_weir_coeff = drainage_params['submerged_weir_coeff']
        # A list of tuple (pyswmm_node_object, row, col)
        self.linked_nodes = linked_node_list
        # create swmm object, open files and start simulation
        self.swmm_sim = pyswmm_sim
        self.swmm_model = self.swmm_sim._model
        # self.swmm_model.swmm_open()
        self.swmm_model.swmm_start()
        # allow ponding
        # if not self.swmm_model.getSimAnalysisSetting(tka.SimAnalysisSettings.AllowPonding):
        #     raise RuntimeWarning('Ponding not allowed, simulation might be unstable')
        # self.swmm5.allow_ponding()

        self.cell_surf = self.dom.cell_surf
        self.old_time = self.swmm_model.getCurrentSimulationTime()
        self._dt = 0.0
        # A dict {nodeid: linkage_flow}
        self.nodes_flow = defaultdict(lambda: 0)

    def __del__(self):
        """Make sure the swmm simulation is ended and closed properly.
        """
        # self.swmm_sim.report()
        # self.swmm_sim.close()
        self.swmm_model.swmm_report()
        self.swmm_model.swmm_close()

    @property
    def dt(self):
        return timedelta(seconds=self._dt)

    @dt.setter
    def dt(self, value):
        raise NotImplementedError("Can't set time-step of a SWMM simulation")

    def step(self):
        """Run a swmm time-step
        calculate the exchanges with raster domain
        """
        self.swmm_model.swmm_step()
        current_time = self.swmm_model.getCurrentSimulationTime()
        self._dt = (current_time - self.old_time).total_seconds()
        assert self._dt > 0
        self.old_time = self.swmm_model.getCurrentSimulationTime()
        return self

    def apply_linkage(self, dt2d):
        """For each linked node,
        calculate the flow entering or leaving the drainage network
        Apply the flow to the node and to the relevant raster cell
        """
        arr_h = self.dom.get_array('h')
        arr_z = self.dom.get_array('dem')
        arr_qd = self.dom.get_array('n_drain')
        for node, row, col in self.linked_nodes:
            z = arr_z[row, col]
            h = arr_h[row, col]
            wse = z + h
            crest_elev = node.invert_elevation + node.full_depth
            # swmm api does not convert units
            surface_area = self.swmm_model.getSimAnalysisSetting(5) * foot * foot
            # weir width is the circumference (node considered circular)
            weir_width = math.pi * 2. * math.sqrt(surface_area / math.pi)
            linkage_type = self.get_linkage_type(wse, crest_elev, node.head,
                                                 weir_width, surface_area)
            linkage_flow = self.get_linkage_flow(wse, node.head, weir_width,
                                                 crest_elev, linkage_type, surface_area,
                                                 self.orifice_coeff, self.free_weir_coeff,
                                                 self.submerged_weir_coeff)
            ## flow limiter ##
            # flow leaving the 2D domain can't drain the corresponding cell
            if linkage_flow < 0:
                maxflow = (h * self.cell_surf) / self._dt
                linkage_flow = max(linkage_flow, -maxflow)

            ## force flow to zero in case of flow inversion ##
            old_flow = self.nodes_flow[node.nodeid]
            overflow_to_drainage = old_flow > 0 and linkage_flow < 0
            drainage_to_overflow = old_flow < 0 and linkage_flow > 0
            if overflow_to_drainage or drainage_to_overflow:
                linkage_type = LinkageTypes.NO_LINKAGE
                linkage_flow = 0.
            # apply flow to 2D model (m/s) and drainage model (m3/s)
            arr_qd[row, col] = linkage_flow / self.cell_surf
            node.generated_inflow(- linkage_flow)
            # update internal values
            self.nodes_flow[node.nodeid] = linkage_flow
        return self

    def get_linkage_type(self, wse, crest_elev, node_head, weir_width,
                         overflow_area):
        """
        """
        depth_2d = wse - crest_elev
        weir_ratio = overflow_area / weir_width
        overflow = node_head > wse
        drainage = node_head < wse
        ########
        # orifice, free- and submerged-weir
        # M. Rubinato et al. (2017)
        # “Experimental Calibration and Validation of Sewer/surface Flow Exchange Equations
        # in Steady and Unsteady Flow Conditions.”
        # https://doi.org/10.1016/j.jhydrol.2017.06.024.
        overflow_orifice = overflow
        free_weir = drainage and (node_head < crest_elev)
        submerged_weir = drainage and (node_head > crest_elev) and (depth_2d < weir_ratio)
        drainage_orifice = drainage and (node_head > crest_elev) and (depth_2d > weir_ratio)
        ########
        if overflow_orifice or drainage_orifice:
            new_linkage_type = LinkageTypes.ORIFICE
        # drainage free weir
        elif free_weir:
            new_linkage_type = LinkageTypes.FREE_WEIR
        # drainage submerged weir
        elif submerged_weir:
            new_linkage_type = LinkageTypes.SUBMERGED_WEIR
        else:
            new_linkage_type = LinkageTypes.NO_LINKAGE
        return new_linkage_type

    def get_linkage_flow(self, wse, node_head, weir_width,
                         crest_elev, linkage_type, overflow_area,
                         orifice_coeff, free_weir_coeff,
                         submerged_weir_coeff):
        """flow sign is :
                - negative when entering the drainage (leaving the 2D model)
                - positive when leaving the drainage (entering the 2D model)
        """
        head_up = max(wse, node_head)
        head_down = min(wse, node_head)
        head_diff = head_up - head_down
        upstream_depth = head_up - crest_elev

        # calculate the flow
        if linkage_type in [LinkageTypes.NO_LINKAGE, LinkageTypes.NOT_LINKED]:
            unsigned_q = 0.
        elif linkage_type == LinkageTypes.ORIFICE:
            unsigned_q = orifice_coeff * overflow_area * math.sqrt(2. * self.g * head_diff)
        elif linkage_type == LinkageTypes.FREE_WEIR:
            unsigned_q = ((2./3.) * free_weir_coeff * weir_width *
                        math.pow(upstream_depth, 3/2.) *
                        math.sqrt(2. * self.g))
        elif linkage_type == LinkageTypes.SUBMERGED_WEIR:
            unsigned_q = (submerged_weir_coeff * weir_width * upstream_depth *
                        math.sqrt(2. * self.g * upstream_depth))
        return math.copysign(unsigned_q, node_head - wse)

class SwmmInputParser(object):
    """A parser for swmm input text file
    """
    # list of sections keywords
    sections_kwd = ["title",  # project title
                    "option",  # analysis options
                    "junction",  # junction node information
                    "outfall",  # outfall node information
                    "divider",  # flow divider node information
                    "storage",  # storage node information
                    "conduit",  # conduit link information
                    "pump",  # pump link
                    "orifice",  # orifice link
                    "weir",  # weir link
                    "outlet",  # outlet link
                    "xsection",  # conduit, orifice, and weir cross-section geometry
                    'coordinate',  # coordinates of drainage system nodes
                    'vertice',  # coordinates of interior vertex points of links
                   ]
    link_types = ['conduit', 'pump', 'orifice', 'weir', 'outlet']
    # define object containers
    junction_values = ['x', 'y', 'elev', 'ymax', 'y0', 'ysur', 'apond']
    Junction = namedtuple('Junction', junction_values)
    Link = namedtuple('Link', ['in_node', 'out_node', 'vertices'])
    # coordinates container
    Coordinates = namedtuple('Coordinates', ['x', 'y'])

    def __init__(self, input_file):
        # read and parse the input file
        assert os.path.isfile(input_file)
        self.inp = dict.fromkeys(self.sections_kwd)
        self.read_inp(input_file)

    def section_kwd(self, sect_name):
        """verify if the given section name is a valid one.
        Return the corresponding section keyword, None if unknown
        """
        # check done in lowercase, without final 's'
        section_valid = sect_name.lower().rstrip('s')
        result = None
        for kwd in self.sections_kwd:
            if kwd.startswith(section_valid):
                result = kwd
        return result

    def read_inp(self, input_file):
        """Read the inp file and generate a dictionary of lists
        """
        with open(input_file, 'r') as inp:
            for line in inp:
                # got directly to next line if comment or empty
                if line.startswith(';') or not line.strip():
                    continue
                # retrive current standard section name
                elif line.startswith('['):
                    current_section = self.section_kwd(line.strip().strip('[] '))
                elif current_section is None:
                    continue
                else:
                    if self.inp[current_section] is None:
                        self.inp[current_section] = []
                    self.inp[current_section].append(line.strip().split())

    def get_juntions_ids(self):
        """return a list of junctions ids (~name)
        """
        return [j[0] for j in self.inp['junction']]

    def get_juntions_as_dict(self):
        """return a dict of namedtuples
        """
        d = {}
        values = []
        for coor in self.inp['coordinate']:
            for j in self.inp['junction']:
                name = j[0]
                if coor[0] == name:
                    j_val = [float(v) for v in j[1:]]
                    values = [float(coor[1]), float(coor[2])] + j_val
                    d[name] = self.Junction._make(values)
        return d

    def get_nodes_id_as_dict(self):
        """return a dict of id:coordinates
        """
        # sections to search
        node_types = ["junction",
                      "outfall",
                      "divider",
                      "storage"]
        # a list of all nodes id
        nodes = []
        for n_t in node_types:
            # add id only if there are values in the dict entry
            if self.inp[n_t]:
                for line in self.inp[n_t]:
                    nodes.append(line[0])

        # A coordinates dict
        coords_dict = {}
        if self.inp['coordinate']:
            for coords in self.inp['coordinate']:
                coords_dict[coords[0]] = self.Coordinates(float(coords[1]),
                                                          float(coords[2]))
        # fill the dict
        node_dict = {}
        for node_id in nodes:
            if node_id in coords_dict:
                node_dict[node_id] = coords_dict[node_id]
            else:
                node_dict[node_id] = None
        return node_dict

    def get_links_id_as_dict(self):
        """return a list of id:Link
        """
        links_dict = {}
        # loop through all types of links
        for k in self.link_types:
            links = self.inp[k]
            if links is not None:
                for ln in links:
                    ID = ln[0]
                    vertices = self.get_vertices(ID)
                    # names of link, inlet and outlet nodes
                    links_dict[ID] = self.Link(in_node=ln[1],
                                               out_node=ln[2],
                                               vertices=vertices)
        return links_dict

    def get_vertices(self, link_name):
        """For a given link name, return a list of Coordinates objects
        """
        vertices = []
        if isinstance(self.inp['vertice'], list):
            for vertex in self.inp['vertice']:
                if link_name == vertex[0]:
                    vertex_c = self.Coordinates(float(vertex[1]),
                                                float(vertex[2]))
                    vertices.append(vertex_c)
        return vertices
