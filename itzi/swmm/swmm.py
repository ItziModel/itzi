# coding=utf8

"""
Copyright (C) 2015-2017  Laurent Courty

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
import ctypes as c
from structs import *
import math
import collections
import swmm_error
import numpy as np
import swmm_c


class Swmm5(object):
    '''A class implementing high-level swmm5 functions.
    '''
    def __init__(self):
        # locate and open SWMM shared library
        so_subdir = 'source/swmm5.so'
        prog_dir = os.path.dirname(__file__)
        swmm_so = os.path.join(prog_dir, so_subdir)
        self.c_swmm5 = c.CDLL(swmm_so)

        self.foot = 0.3048  # foot to metre
        self.is_open = False
        self.is_started = False
        self.routing_model = None
        self.elapsed_time = 0

    def get_version(self):
        '''return swmm version as an integer'''
        return self.c_swmm5.swmm_getVersion()

    def swmm_open(self, input_file, report_file, output_file):
        '''Opens a swmm project
        '''
        err = self.c_swmm5.swmm_open(c.c_char_p(input_file),
                                     c.c_char_p(report_file),
                                     c.c_char_p(output_file))
        if err != 0:
            raise swmm_error.SwmmError(err)
        else:
            self.input_file = input_file
            self.report_file = report_file
            self.output_file = output_file
            self.is_open = True
            self.routing_model = self.get_RouteModel()
        return self

    def swmm_close(self):
        '''Closes a swmm project
        '''
        self.c_swmm5.swmm_close()
        self.is_open = False
        return 0

    def swmm_start(self, save_results = 1):
        '''Starts a swmm simulation
        '''
        if not self.is_open:
            raise swmm_error.NotOpenError
        err = self.c_swmm5.swmm_start(c.c_int(save_results))
        if err != 0:
            raise swmm_error.SwmmError(err)
        self.is_started = True
        return self

    def swmm_end(self):
        '''Ends a swmm simulation
        '''
        if not self.is_started:
            raise swmm_error.NotStartedError
        err = self.c_swmm5.swmm_end()
        if err != 0:
            raise swmm_error.SwmmError(err)
        self.is_started = False
        return self

    def swmm_step(self):
        '''Advances the simulation by one routing time step
        '''
        c_elapsed_time = c.c_double(self.elapsed_time)
        err = self.c_swmm5.swmm_step(c.byref(c_elapsed_time))
        self.elapsed_time = c_elapsed_time.value
        if err != 0:
            raise swmm_error.SwmmError(err)
        return self

    def get_RouteModel(self):
        '''Get the node minimal surface area
        (storage node could be larger)
        '''
        if not self.is_open:
            raise swmm_error.NotOpenError
        route_code = c.c_uint.in_dll(self.c_swmm5, 'RouteModel').value
        # Cf. enum RouteModelType in enums.h
        return ROUTING_MODELS.get(route_code)

    def get_NewRoutingTime(self):
        """retrieve new routing time in msec from shared object
        return the value in seconds
        """
        if not self.is_started:
            raise swmm_error.NotStartedError
        new_routing = c.c_double.in_dll(self.c_swmm5, 'NewRoutingTime').value
        return new_routing / 1000.

    def get_OldRoutingTime(self):
        """retrieve old routing time in msec from shared object
        return the value in seconds
        """
        if not self.is_started:
            raise swmm_error.NotStartedError
        old_routing = c.c_double.in_dll(self.c_swmm5, 'OldRoutingTime').value
        return old_routing / 1000.

    def get_index(self, object_type, object_id):
        """with a given type and id, return the swmm object index
        """
        assert isinstance(object_type, int)
        # Return None if id is not a string
        if not isinstance(object_id, str):
            return None
        else:
            # Call the C function
            link_idx = self.c_swmm5.project_findObject(object_type,
                                                       c.c_char_p(object_id))
            return link_idx

    def get_MinSurfArea(self):
        '''Get the node minimal surface area in sqm
        (storage node could be larger)
        '''
        if not self.is_open:
            raise swmm_error.NotOpenError
        if self.routing_model == 'DW':
            area = c.c_double.in_dll(self.c_swmm5, 'MinSurfArea').value
            return area * self.foot ** 2  # return SI value
        else:
            raise RuntimeError('MinSurfArea only valid for Dynamic Wave routing')

    def set_allow_ponding(self):
        '''Force model to allow ponding
        '''
        if not self.is_open:
            raise swmm_error.NotOpenError
        AllowPonding = c.c_int.in_dll(self.c_swmm5, 'AllowPonding').value
        if AllowPonding != 1:
            self.c_swmm5.swmm_setAllowPonding(c.c_int(1))
        return self

    def set_weighting_factor(self):
        '''Calculate weighting factor necessary to get node results
        between two time step
        '''
        reportTime = c.c_double.in_dll(self.c_swmm5, 'ReportTime')
        OldRoutingTime = c.c_double.in_dll(self.c_swmm5, 'OldRoutingTime')
        NewRoutingTime = c.c_double.in_dll(self.c_swmm5, 'NewRoutingTime')

        return ((reportTime.value - OldRoutingTime.value) /
                (NewRoutingTime.value - OldRoutingTime.value))

    def get_nobjects(self):
        '''Get the number of each object type in the model
        Return a dictionary
        Depends on SWMM enum ObjectType in enums.h
        '''
        # Define the result list length
        nobjects_types = 17  # cf. enums.h
        # retrieve the list as a ctypes array
        c_nobjects = (c.c_int * nobjects_types).in_dll(self.c_swmm5, "Nobjects")
        # Populate the results dictionary
        nobjects = {}
        nobjects['GAGE'] = c_nobjects[0]
        nobjects['SUBCATCH'] = c_nobjects[1]
        nobjects['NODE'] = c_nobjects[2]
        nobjects['LINK'] = c_nobjects[3]
        nobjects['POLLUT'] = c_nobjects[4]
        nobjects['LANDUSE'] = c_nobjects[5]
        nobjects['TIMEPATTERN'] = c_nobjects[6]
        nobjects['CURVE'] = c_nobjects[7]
        nobjects['TSERIES'] = c_nobjects[8]
        nobjects['CONTROL'] = c_nobjects[9]
        nobjects['TRANSECT'] = c_nobjects[10]
        nobjects['AQUIFER'] = c_nobjects[11]
        nobjects['UNITHYD'] = c_nobjects[12]
        nobjects['SNOWMELT'] = c_nobjects[13]
        nobjects['SHAPE'] = c_nobjects[14]
        nobjects['LID'] = c_nobjects[15]
        nobjects['MAX_OBJ_TYPES'] = c_nobjects[16]
        return nobjects

    def get_nnodes(self):
        '''Get the number of each node type
        return a dictionary
        elements defined in SWMM enums.h
        '''
        # Define the result list length
        nnodes_types = 4  # cf. enums.h
        # retrieve the list as a ctypes array
        c_nnodes = (c.c_int * nnodes_types).in_dll(self.c_swmm5, "Nnodes")
        # populate the dictionary
        nnodes = {}
        nnodes['JUNCTION'] = c_nnodes[0]
        nnodes['OUTFALL'] = c_nnodes[1]
        nnodes['STORAGE'] = c_nnodes[2]
        nnodes['DIVIDER'] = c_nnodes[3]
        return nnodes

    def node_getResults(self,
                        node_index = 0,
                        weighting_factor = None):
        '''Computes weighted average of old and new results at a node.
        Input:  node index
                weighting factor
        Returns a dictionary (Cf. SWMM enum NodeResultType in enums.h):
                DEPTH, water depth above invert
                HEAD, hydraulic head
                VOLUME, volume stored & ponded
                LATFLOW, lateral inflow rate
                INFLOW, total inflow rate
                OVERFLOW, overflow rate
                QUAL, concentration of each pollutant
        '''
        if weighting_factor is None:
            weighting_factor = self.set_weighting_factor()
        # generate a list of 7 items (max number of results enums.h)
        arr_var = [i for i in xrange(7)]
        # transform it into a C array
        c_arr_var = (c.c_float * len(arr_var))(*arr_var)

        self.c_swmm5.node_getResults(c.c_int(node_index),
                                     c.c_double(weighting_factor),
                                     c_arr_var)
        # get the results back in python dictionary
        node_var = {}
        node_var['DEPTH'] = c_arr_var[0]
        node_var['HEAD'] = c_arr_var[1]
        node_var['VOLUME'] = c_arr_var[2]
        node_var['LATFLOW'] = c_arr_var[3]
        node_var['INFLOW'] = c_arr_var[4]
        node_var['OVERFLOW'] = c_arr_var[5]
        node_var['QUALITY'] = c_arr_var[6]
        return node_var

    def routing_getRoutingStep(self):
        '''Get swmm routing time step'''
        route_code = c.c_int.in_dll(self.c_swmm5, 'RouteModel').value
        route_step = c.c_double.in_dll(self.c_swmm5, 'RouteStep').value
        c_func = self.c_swmm5.routing_getRoutingStep
        c_func.restype = c.c_double
        routing_step = c_func(c.c_int(route_code),
                              c.c_double(route_step))
        return routing_step


class SwmmLink(object):
    """Define access to a SWMM link object
    It create an interface for the values stored in SwmmNetwork
    """
    sql_columns_def = [(u'cat', 'INTEGER PRIMARY KEY'),
                       (u'link_id', 'TEXT'),
                       (u'type', 'TEXT'),
                       (u'flow', 'REAL'),
                       (u'depth', 'REAL'),
                       (u'velocity', 'REAL'),
                       (u'volume', 'REAL'),
                       (u'offset1', 'REAL'),
                       (u'offset2', 'REAL'),
                       (u'yFull', 'REAL'),
                       (u'froude', 'REAL')]

    def __init__(self, swmm_network, link_id):
        self.swmm_net = swmm_network
        self.link_id = link_id
        self.start_node_id = None
        self.end_node_id = None
        self.vertices = []

    def get_attrs(self):
        """return a list of link data in the right DB order
        """
        values = self.swmm_net.get_link_values(self.link_id)
        link_type = LINK_TYPES[values['link_type']]
        return [self.link_id, link_type, values['flow'], values['depth'],
                values['velocity'], values['volume'],
                values['start_node_offset'], values['end_node_offset'],
                values['full_depth'], values['froude']]


class SwmmNode(object):
    '''Define a SWMM node object
    should be defined by a swmm simulation object and a node ID
    '''
    sql_columns_def = [(u'cat', 'INTEGER PRIMARY KEY'),
                       (u'node_id', 'TEXT'),
                       (u'type', 'TEXT'),
                       (u'linkage_type', 'TEXT')
                       (u'inflow', 'REAL'),
                       (u'outflow', 'REAL'),
                       (u'latFlow', 'REAL')
                       (u'losses', 'REAL'),
                       (u'overflow', 'REAL'),
                       (u'depth', 'REAL'),
                       (u'head', 'REAL'),
                       (u'crownElev', 'REAL'),
                       (u'crestElev', 'REAL'),
                       (u'invertElev', 'REAL'),
                       (u'initDepth', 'REAL'),
                       (u'fullDepth', 'REAL'),
                       (u'surDepth', 'REAL'),
                       (u'pondedArea', 'REAL'),
                       (u'degree', 'INT'),
                       (u'newVolume', 'REAL'),
                       (u'fullVolume', 'REAL')]

    def __init__(self, swmm_network, node_id, coordinates=None, grid_coords=None):
        self.swmm_net = swmm_network
        if not self.swmm_sim.is_started:
            raise swmm_error.NotStartedError
        # need to add a node validity check
        self.node_id = node_id
        self.coordinates = coordinates
        self.grid_coords = grid_coords
        self.node_type = None

    def get_attrs(self):
        """return a list of node data in the right DB order
        """
        values = self.swmm_net.get_node_values(self.node_id)
        self.node_type = NODE_TYPES[values['node_type']]
        linkage_type = LINKAGE_TYPE[values['linkage_type']]
        return [self.node_id, self.node_type, linkage_type,
                values['inflow'], values['outflow'], values['lat_flow'],
                values['losses'], values['overflow'], values['depth'],
                values['head'], values['crown_elev'], values['crest_elev'],
                values['invert_elev'], values['init_depth'], values['full_depth'],
                values['sur_depth'], values['ponded_area'], values['degree'],
                values['volume'], values['full_volume']]

    def is_linkable(self):
        """Return True if the node is used for interactions with surface
        """
        if self.node_type == 'junction' and self.grid_coords:
            return True
        else:
            return False

    def set_crest_elev(self, z):
        '''Set the crest elevation according to the 2D dem
        the crest elevation could not be lower than ground
        update swmm Node.fullDepth if necessary
        '''
        if z > self.crest_elev:
            # set new FullDepth in swmm:
            c_new_FullDepth = c.c_double((z - self.invert_elev) /
                                         self.foot)
            self.swmm_sim.c_swmm5.swmm_setNodeFullDepth(
                c.c_char_p(self.node_id), c_new_FullDepth)
            # update the object's data
            self.update()
        return self

    def set_pondedArea(self):
        '''Set the ponded area equal to overflow area.
        SWMM internal ponding don't have meaning anymore with the 2D coupling
        The ponding depth is used to keep the node head consistant with
        the WSE of the 2D model
        '''
        c_ponded = c.c_double(self.overflow_area / self.foot ** 2)
        self.swmm_sim.c_swmm5.swmm_setNodePondedArea(c.c_char_p(self.node_id),
                                                     c_ponded)
        self.update()
        return self


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
    Junction = collections.namedtuple('Junction', junction_values)
    Link = collections.namedtuple('Link', ['in_node', 'out_node', 'vertices'])
    # coordinates container
    Coordinates = collections.namedtuple('Coordinates', ['x', 'y'])

    def __init__(self, input_file):
        # read and parse the input file
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
        for c in self.inp['coordinate']:
            for j in self.inp['junction']:
                name = j[0]
                if c[0] == name:
                    j_val = [float(v) for v in j[1:]]
                    values = [float(c[1]), float(c[2])] + j_val
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

class SwmmNetwork(object):
    """Represent a SWMM network
    values of Nodes and Links are stored in dicts of np.ndarray
    """
    def __init__(self, swmm_object, nodes_dict, links_dict, g):
        self.g = g
        self.swmm_sim = swmm_object
        # data type of each array
        self.nodes_dtypes = {'node_id': np.int32, 'linkage_type': np.int32,
                             'inflow': np.float32, 'outflow': np.float32,
                             'head': np.float32, 'crest_elev': np.float32,
                             'node_type': np.int32, 'sub_index': np.int32,
                             'invert_elev': np.float32, 'init_depth': np.float32,
                             'full_depth': np.float32, 'sur_depth': np.float32,
                             'ponded_area': np.float32, 'degree': np.int32,
                             'crown_elev': np.float32,
                             'losses': np.float32, 'volume': np.float32,
                             'full_volume': np.float32, 'overflow': np.float32,
                             'depth': np.float32, 'lat_flow': np.float32,
                             'x': np.float32, 'y': np.float32}
        self.nodes = dict.fromkeys(self.nodes_dtypes.keys())
        # data type of each array
        self.links_dtypes = {'link_id': np.int32, 'flow': np.float32,
                             'depth': np.float32, 'velocity': np.float32,
                             'volume': np.float32, 'link_type': np.int32,
                             'start_node_offset': np.float32,
                             'end_node_offset': np.float32,
                             'full_depth': np.float32, 'froude': np.float32}
        self.links = dict.fromkeys(self.links_dtypes.keys())

        # create dicts {idx: id}
        self.links_id = {}
        for link_id in links_dict:
            link_idx = self.swmm_sim.get_index(ObjectType.LINK, link_id)
            self.links_id[link_idx] = link_id
        self.nodes_id = {}
        for node_id in nodes_dict:
            node_idx = self.swmm_sim.get_index(ObjectType.NODE, node_id)
            self.nodes_id[node_idx] = node_id

        # set arrays
        self._create_arrays()

    def get_arr(self, obj_type, k):
        """for a given key, return the corresponding array
        """
        obj_types = {'links': self.links,
                     'nodes': self.nodes}
        return obj_types.get(obj_type).get(k)

    def _create_arrays(self):
        """create arrays according to types and number of objects
        """
        # nodes arrays
        self.nodes['node_id'] = np.array(self.nodes_id.keys(),
                                         dtype=self.nodes_dtypes['node_id'])
        for k in self.nodes:
            if k == 'node_id':
                continue
            self.nodes[k] = np.zeros_like(self.nodes['node_id'],
                                           dtype=self.nodes_dtypes[k])
        # links arrays
        self.links['link_id'] = np.array(self.links_id.keys(),
                                         dtype=self.links_dtypes['link_id'])
        for k in self.links:
            if k == 'link_id':
                continue
            self.links[k] = np.zeros_like(self.links['link_id'],
                                           dtype=self.links_dtypes[k])
        return self

    def update_links(self):
        """Update arrays with values from SWMM using cython function
        """
        swmm_c.update_links(self.links['link_id'], self.links['flow'],
                            self.links['depth'], self.links['velocity'],
                            self.links['volume'], self.links['link_type'],
                            self.links['start_node_offset'],
                            self.links['end_node_offset'],
                            self.links['full_depth'], self.links['froude'])
        return self

    def update_nodes(self):
        """Update arrays with values from SWMM using cython function
        """
        swmm_c.update_nodes(node_id=self.nodes['node_id'],
                            inflow=self.nodes['inflow'], outflow=self.nodes['outflow'],
                            head=self.nodes['head'], crest_elev=self.nodes['crest_elev'],
                            node_type=self.nodes['node_type'], sub_index=self.nodes['sub_index'],
                            invert_elev=self.nodes['invert_elev'], init_depth=self.nodes['init_depth'],
                            full_depth=self.nodes['full_depth'], sur_depth=self.nodes['sur_depth'],
                            ponded_area=self.nodes['ponded_area'], degree=self.nodes['degree'],
                            crown_elev=self.nodes['crown_elev'],
                            losses=self.nodes['losses'], volume=self.nodes['volume'],
                            full_volume=self.nodes['full_volume'], overflow=self.nodes['overflow'],
                            depth=self.nodes['depth'], lat_flow=self.nodes['lat_flow'])
        return self

    def get_link_values(self, link_id):
        """for a given link ID, return a dict of values
        """
        link_values = dict.fromkeys(self.links.keys())
        # get the int link index
        link_idx = self.swmm_sim.get_index(ObjectType.LINK, link_id)
        # find the value of the given node
        for k, arr in self.links:
            link_values[k] = arr[self.nodes['node_id'] == link_idx]
        return link_values

    def get_node_values(self, node_id):
        """for a given node ID, return a dict of values
        """
        node_values = dict.fromkeys(self.links.keys())
        # get the int link index
        node_idx = self.swmm_sim.get_index(ObjectType.NODE, node_id)
        # find the value of the given node
        for k, arr in self.links:
            node_values[k] = arr[self.links['link_id'] == node_idx]
        return link_values

    def apply_linkage_flow(self, arr_wse, arr_qdrain, cell_surf, dt2d, dt1d):
        """
        """
        apply_linkage_flow(arr_node_id, arr_crest_elev,
                           arr_depth, arr_head,
                           arr_row, arr_col,
                           arr_wse,
                           arr_linkage_type, arr_qdrain,
                           cell_surf=cell_surf, dt2d=dt2d, dt1d=dt1d, g=self.g)

    def apply_linkage(self, arr_h, arr_z, arr_qd):
        """arr_h = depth in surface model
           arr_z = terrain elevation
           arr_qd = drainage flow
        """
