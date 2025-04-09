# coding=utf8
"""
Copyright (C) 2016-2025 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

import os
import math
from datetime import timedelta
from collections import namedtuple
from collections import defaultdict

import pyswmm
import numpy as np

foot = 0.3048


class LinkageTypes:
    NOT_LINKED = 0
    LINKED_NO_FLOW = 1
    FREE_WEIR = 2
    SUBMERGED_WEIR = 3
    ORIFICE = 4


class DrainageSimulation():
    """manage simulation of the pipe network
    write results to RasterDomain object
    """
    def __init__(self, raster_domain, pyswmm_sim, nodes_list, links_list):
        self.dom = raster_domain
        # A list of tuple (DrainageNode, row, col)
        self.nodes = nodes_list
        # A list of DrainageLink objects
        self.links = links_list
        # create swmm object, open files and start simulation
        self.swmm_sim = pyswmm_sim
        self.swmm_model = self.swmm_sim._model
        # self.swmm_model.swmm_open()
        self.swmm_model.swmm_start()
        # allow ponding
        # TODO: check if allowing ponding is necessary

        self.cell_surf = self.dom.cell_surf
        self._dt = 0.0
        # A dict {nodeid: linkage_flow}
        self.nodes_flow = defaultdict(lambda: 0)
        self.elapsed_time = 0.0

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
        elapsed_days = self.swmm_model.swmm_step()
        elapsed_seconds = elapsed_days * 24 * 3600
        # calculate the time step
        self._dt = elapsed_seconds - self.elapsed_time
        assert self._dt > 0.
        self.elapsed_time = elapsed_seconds
        return self

    def apply_linkage_to_nodes(self, dt2d):
        """For each linked node,
        calculate the flow entering or leaving the drainage network
        Apply the flow to the node and to the relevant raster cell
        """
        arr_h = self.dom.get_array('h')
        arr_z = self.dom.get_array('dem')
        arr_qd = self.dom.get_array('n_drain')
        for node, row, col in self.nodes:
            if node.is_linked():
                z = arr_z[row, col]
                h = arr_h[row, col]
                node.apply_linkage(z, h, self._dt, self.cell_surf)
                # apply flow to 2D model (m/s) and drainage model (m3/s)
                arr_qd[row, col] = node.linkage_flow / self.cell_surf
                # self.nodes_flow[node.nodeid] = linkage_flow
        return self


class DrainageNode(object):
    '''A wrapper around the pyswmm node object.
    Includes the flow linking logic
    '''

    def __init__(self, node_object, coordinates=None,
                 linkage_type=LinkageTypes.NOT_LINKED,
                 orifice_coeff=None,
                 free_weir_coeff=None,
                 submerged_weir_coeff=None,
                 g=9.81):
        self.g = g
        self._model = node_object._model
        self.pyswmm_node = node_object
        # need to add a node validity check
        self.node_id = node_object.nodeid
        self.coordinates = coordinates
        self.orifice_coeff = orifice_coeff
        self.free_weir_coeff = free_weir_coeff
        self.submerged_weir_coeff = submerged_weir_coeff
        self.node_type = self.get_node_type()
        self.surface_area = self._model.getSimAnalysisSetting(
            pyswmm.toolkitapi.SimulationParameters.MinSurfArea)
        # weir width is the circumference (node considered circular)
        self.weir_width = 2 * math.sqrt(self.surface_area * math.pi)
        # Set deafault values
        self.linkage_type = linkage_type
        self.linkage_flow = 0.0
        # TODO: set crest elevation to at least DEM
        # TODO: set surcharge depth
        self.relaxation_factor = 0.9
        self.damping_factor = 0.5

    def get_node_type(self):
        """
        """
        if self.pyswmm_node.is_junction():
            return 'junction'
        elif self.pyswmm_node.is_outfall():
            return 'outfall'
        elif self.pyswmm_node.is_divider():
            return 'divider'
        elif self.pyswmm_node.is_storage():
            return 'storage'
        else:
            raise ValueError(f"Unknown node type for node {self.node_id}")

    def get_full_volume(self):
        return self.surface_area * self.pyswmm_node.full_depth

    def get_overflow(self):
        return self._model.getNodeResult(
            self.node_id, pyswmm.toolkitapi.NodeResults.overflow)

    def get_crest_elev(self):
        """return the crest elevation of the node
        """
        return self.pyswmm_node.invert_elevation + self.pyswmm_node.full_depth

    def get_linkage_type_as_str(self):
        """return the linkage type as a string
        """
        if self.linkage_type == LinkageTypes.LINKED_NO_FLOW:
            return 'linked, no flow'
        elif self.linkage_type == LinkageTypes.FREE_WEIR:
            return 'free weir'
        elif self.linkage_type == LinkageTypes.SUBMERGED_WEIR:
            return 'submerged weir'
        elif self.linkage_type == LinkageTypes.ORIFICE:
            return 'orifice'
        elif self.linkage_type == LinkageTypes.NOT_LINKED:
            return 'not linked'
        else:
            raise ValueError(f"Unknown linkage type for node {self.node_id}")

    def is_linked(self):
        """return True if the node is linked to the 2D domain
        """
        return self.linkage_type != LinkageTypes.NOT_LINKED

    def get_attrs(self):
        """return a list of node data in the right DB order
        """
        attrs = [self.node_id, self.node_type,
                 self.get_linkage_type_as_str(),
                 self.linkage_flow,
                 self.pyswmm_node.total_inflow,
                 self.pyswmm_node.total_outflow,
                 self.pyswmm_node.lateral_inflow,
                 self.pyswmm_node.losses,
                 self.get_overflow(),
                 self.pyswmm_node.depth,
                 self.pyswmm_node.head,
                #  values['crown_elev'],
                 self.get_crest_elev(),
                 self.pyswmm_node.invert_elevation,
                 self.pyswmm_node.initial_depth,
                 self.pyswmm_node.full_depth,
                 self.pyswmm_node.surcharge_depth,
                 self.pyswmm_node.ponding_area,
                #  values['degree'],
                 self.pyswmm_node.volume,
                 self.get_full_volume()]
        return attrs

    def apply_linkage(self, z, h, dt_drainage, cell_surf):
        """Apply the linkage to the node
        """
        wse = z + h
        crest_elev = self.get_crest_elev()
        # Calculate the linkage type and flow
        self.linkage_type = self._get_linkage_type(wse, crest_elev)
        new_linkage_flow = self._get_linkage_flow(wse, crest_elev)
        
        ## flow relaxation ##
        # Apply a relaxation factor (blend new flow with previous flow)
        new_linkage_flow = (self.relaxation_factor * new_linkage_flow +
                            (1 - self.relaxation_factor) * self.linkage_flow)
        
        ## flow limiter ##
        # flow leaving the 2D domain can't drain the corresponding cell
        if new_linkage_flow < 0.:
            maxflow = (h * cell_surf) / dt_drainage
            new_linkage_flow = max(new_linkage_flow, -maxflow)

        ## Dampen flow in case of flow inversion ##
        old_flow = self.linkage_flow
        overflow_to_drainage = old_flow > 0 and new_linkage_flow < 0
        drainage_to_overflow = old_flow < 0 and new_linkage_flow > 0
        if overflow_to_drainage or drainage_to_overflow:
            new_linkage_flow = new_linkage_flow * self.damping_factor

        # pyswmm fails if type not forced to double
        self.pyswmm_node.generated_inflow(np.float64(- new_linkage_flow))
        # update internal values
        self.linkage_flow = new_linkage_flow
        return self

    def _get_linkage_type(self, wse, crest_elev):
        """orifice, free- and submerged-weir
        M. Rubinato et al. (2017)
        “Experimental Calibration and Validation of Sewer/surface Flow Exchange Equations
        in Steady and Unsteady Flow Conditions.”
        https://doi.org/10.1016/j.jhydrol.2017.06.024.
        """
        depth_2d = wse - crest_elev
        weir_ratio = self.surface_area / self.weir_width
        node_head = self.pyswmm_node.head
        overflow = node_head > wse
        drainage = node_head < wse
        free_weir = drainage and (node_head < crest_elev)
        submerged_weir = drainage and (node_head > crest_elev) and (depth_2d < weir_ratio)
        drainage_orifice = drainage and (node_head > crest_elev) and (depth_2d > weir_ratio)
        # orifice
        if overflow or drainage_orifice:
            new_linkage_type = LinkageTypes.ORIFICE
        # drainage free weir
        elif free_weir:
            new_linkage_type = LinkageTypes.FREE_WEIR
        # drainage submerged weir
        elif submerged_weir:
            new_linkage_type = LinkageTypes.SUBMERGED_WEIR
        else:
            new_linkage_type = LinkageTypes.LINKED_NO_FLOW
        return new_linkage_type

    def _get_linkage_flow(self, wse, crest_elev):
        """flow sign is :
                - negative when entering the drainage (leaving the 2D model)
                - positive when leaving the drainage (entering the 2D model)
        """
        node_head = self.pyswmm_node.head
        head_up = max(wse, node_head)
        head_down = min(wse, node_head)
        head_diff = head_up - head_down
        upstream_depth = head_up - crest_elev

        # calculate the flow
        if self.linkage_type in [LinkageTypes.LINKED_NO_FLOW, LinkageTypes.NOT_LINKED]:
            unsigned_q = 0.
        elif self.linkage_type == LinkageTypes.ORIFICE:
            unsigned_q = self.orifice_coeff * self.surface_area * math.sqrt(2. * self.g * head_diff)
        elif self.linkage_type == LinkageTypes.FREE_WEIR:
            unsigned_q = ((2./3.) * self.free_weir_coeff * self.weir_width *
                        math.pow(upstream_depth, 3/2.) *
                        math.sqrt(2. * self.g))
        elif self.linkage_type == LinkageTypes.SUBMERGED_WEIR:
            unsigned_q = (self.submerged_weir_coeff * self.weir_width * upstream_depth *
                        math.sqrt(2. * self.g * upstream_depth))
        return math.copysign(unsigned_q, node_head - wse)


class DrainageLink(object):
    """A wrapper around the pyswmm link object
    """
    def __init__(self, link_object, vertices=[]):
        self.pyswmm_link = link_object
        self.link_id = self.pyswmm_link.linkid
        self.link_type = self._get_link_type()
        self.start_node_id = self.pyswmm_link.inlet_node
        self.end_node_id = self.pyswmm_link.outlet_node
        # vertices include the coordinates of the start and end nodes
        self.vertices = vertices

    def _get_link_type(self):
        """return the type of the link
        """
        if self.pyswmm_link.is_conduit():
            link_type = "conduit"
        elif self.pyswmm_link.is_pump():
            link_type = "pump"
        elif self.pyswmm_link.is_orifice():
            link_type = "orifice"
        elif self.pyswmm_link.is_weir():
            link_type = "weir"
        elif self.pyswmm_link.is_outlet():
            link_type = "outlet"
        else:
            raise ValueError(f"Unknown link type for link {self.link_id}")
        return link_type

    def get_attrs(self):
        """return a list of link data in the right DB order
        """
        attrs = [self.link_id,
                 self.link_type,
                 self.pyswmm_link.flow,
                 self.pyswmm_link.depth,
                #  values['velocity'],
                 self.pyswmm_link.volume,
                 self.pyswmm_link.inlet_offset,
                 self.pyswmm_link.outlet_offset,
                #  values['full_depth'],
                 self.pyswmm_link.froude]
        return attrs


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
