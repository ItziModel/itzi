# coding=utf8

"""
Copyright (C) 2015-2020  Laurent Courty

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
import ctypes as c
import collections
import numpy as np

from itzi.swmm.structs import ObjectType, LINK_TYPES, NODE_TYPES, ROUTING_MODELS, LINKAGE_TYPES
import itzi.swmm.swmm_error as swmm_error
import itzi.swmm.swmm_c as swmm_c

SO_SUBDIR = 'swmm_c.so'

class Swmm5(object):
    '''A class implementing high-level swmm5 functions.
    '''
    def __init__(self):
        # locate and open SWMM shared library
        prog_dir = os.path.dirname(__file__)
        swmm_so = os.path.join(prog_dir, SO_SUBDIR)
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
        err = self.c_swmm5.swmm_open(c.c_char_p(input_file.encode('utf-8')),
                                     c.c_char_p(report_file.encode('utf-8')),
                                     c.c_char_p(output_file.encode('utf-8')))
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

    def swmm_start(self, save_results=1):
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
            link_idx = self.c_swmm5.project_findObject(c.c_int(object_type),
                                                       c.c_char_p(object_id))
            return link_idx

    def allow_ponding(self):
        '''Force model to allow ponding
        '''
        if not self.is_open:
            raise swmm_error.NotOpenError
        AllowPonding = c.c_int.in_dll(self.c_swmm5, 'AllowPonding').value
        if AllowPonding != 1:
            self.c_swmm5.swmm_setAllowPonding(c.c_int(1))
        return self

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
        attrs = [self.link_id, link_type, values['flow'], values['depth'],
                 values['velocity'], values['volume'],
                 values['offset1'], values['offset2'],
                 values['full_depth'], values['froude']]
        return attrs


class SwmmNode(object):
    '''Define a SWMM node object
    should be defined by a SwmmNetwork object and a node ID
    '''
    sql_columns_def = [(u'cat', 'INTEGER PRIMARY KEY'),
                       (u'node_id', 'TEXT'),
                       (u'type', 'TEXT'),
                       (u'linkage_type', 'TEXT'),
                       (u'linkage_flow', 'REAL'),
                       (u'inflow', 'REAL'),
                       (u'outflow', 'REAL'),
                       (u'latFlow', 'REAL'),
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

    def __init__(self, swmm_network, node_id, coordinates=None):
        self.swmm_net = swmm_network
        # need to add a node validity check
        self.node_id = node_id
        self.coordinates = coordinates
        self.node_type = None

    def get_attrs(self):
        """return a list of node data in the right DB order
        """
        values = self.swmm_net.get_node_values(self.node_id)
        self.node_type = NODE_TYPES[values['node_type']]
        linkage_type = LINKAGE_TYPES[values['linkage_type']]
        attrs = [self.node_id, self.node_type, linkage_type,
                 values['linkage_flow'],
                 values['inflow'], values['outflow'], values['lat_flow'],
                 values['losses'], values['overflow'], values['depth'],
                 values['head'], values['crown_elev'], values['crest_elev'],
                 values['invert_elev'], values['init_depth'], values['full_depth'],
                 values['sur_depth'], values['ponded_area'], values['degree'],
                 values['volume'], values['full_volume']]
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


class SwmmNetwork(object):
    """Represent a SWMM network
    values of Nodes and Links are stored in dicts of np.ndarray
    """
    NODES_DTYPES = [('idx', np.int32), ('linkage_type', np.int32),
                    ('inflow', np.float32), ('outflow', np.float32),
                    ('linkage_flow', np.float32),
                    ('head', np.float32), ('crest_elev', np.float32),
                    ('node_type', np.int32), ('sub_index', np.int32),
                    ('invert_elev', np.float32), ('init_depth', np.float32),
                    ('full_depth', np.float32), ('sur_depth', np.float32),
                    ('ponded_area', np.float32), ('degree', np.int32),
                    ('crown_elev', np.float32),
                    ('losses', np.float32), ('volume', np.float32),
                    ('full_volume', np.float32), ('overflow', np.float32),
                    ('depth', np.float32), ('lat_flow', np.float32),
                    ('row', np.int32), ('col', np.int32)]

    LINKS_DTYPES = [('idx', np.int32), ('flow', np.float32),
                    ('depth', np.float32), ('velocity', np.float32),
                    ('volume', np.float32), ('link_type', np.int32),
                    ('offset1', np.float32), ('offset2', np.float32),
                    ('full_depth', np.float32), ('froude', np.float32)]

    def __init__(self, nodes_dict, links_dict, igis, g, cell_surf,
                 orifice_coeff, free_weir_coeff, submerged_weir_coeff):
        self.g = g
        self.cell_surf = cell_surf
        self.orifice_coeff = orifice_coeff
        self.free_weir_coeff = free_weir_coeff
        self.submerged_weir_coeff = submerged_weir_coeff
        # GIS interface
        self.gis = igis
        # field names
        self.node_fields = [f[0] for f in self.NODES_DTYPES]
        self.link_fields = [f[0] for f in self.LINKS_DTYPES]

        # create dicts relating index (int) to ID (str) {idx: id}
        self.links_id = {}
        for link_id in links_dict:
            link_idx = swmm_c.get_object_index(ObjectType.LINK, link_id.encode('utf-8'))
            self.links_id[link_idx] = link_id
        self.nodes_id = {}
        for node_id in nodes_dict:
            node_idx = swmm_c.get_object_index(ObjectType.NODE, node_id.encode('utf-8'))
            self.nodes_id[node_idx] = node_id

        # set arrays
        self._create_arrays()
        # set linkability
        self._set_linkable(nodes_dict)

    def get_arr(self, obj_type, k):
        """for a given key, return a view of the corresponding array
        """
        obj_types = {'links': self.links,
                     'nodes': self.nodes}
        return obj_types.get(obj_type)[k]

    def _create_arrays(self):
        """create arrays according to types and number of objects
        """
        # Length of arrays
        nodes_len = len(self.nodes_id.keys())
        links_len = len(self.links_id.keys())

        # Create structured arrays
        self.nodes = np.zeros([nodes_len], dtype=self.NODES_DTYPES)
        self.links = np.zeros([links_len], dtype=self.LINKS_DTYPES)

        # fill indices
        self.nodes['idx'][:] = np.array(list(self.nodes_id.keys()))
        self.links['idx'][:] = np.array(list(self.links_id.keys()))
        return self

    def _set_linkable(self, nodes_dict):
        """Check if the nodes are inside the region and can be linked.
        Set linkage_type accordingly
        0: not linkable
        1: linked, no flow
        """
        for node in self.nodes:
            node_idx = node['idx']
            node_id = self.nodes_id[node_idx]
            coors = nodes_dict[node_id]
            # a node without coordinates cannot be linked
            if coors is None or not self.gis.is_in_region(coors.x, coors.y):
                node['linkage_type'] = 0
            else:
                # get row and column
                row, col = self.gis.coor2pixel(coors)
                node['linkage_type'] = 1
                # in other cases, set as linkable
                node['row'] = int(row)
                node['col'] = int(col)
                # set ponding parameters
                swmm_c.set_ponding_area(node_idx)
        return self

    def update_links(self):
        """Update arrays with values from SWMM using cython function
        """
        swmm_c.update_links(self.links)
        return self

    def update_nodes(self):
        """Update arrays with values from SWMM using cython function
        """
        swmm_c.update_nodes(self.nodes)
        return self

    def get_link_values(self, link_id):
        """for a given link ID, return a dict of values
        """
        link_values = dict.fromkeys(self.link_fields)
        # get the int link index
        link_idx = swmm_c.get_object_index(ObjectType.LINK, link_id)
        # find the value of the given node
        for val_k in link_values:
            link_values[val_k] = np.asscalar(self.links[link_idx][val_k])
        return link_values

    def get_node_values(self, node_id):
        """for a given node ID, return a dict of values
        """
        node_values = dict.fromkeys(self.node_fields)
        # get the int link index
        node_idx = swmm_c.get_object_index(ObjectType.NODE, node_id)
        # find the value of the given node
        for val_k in node_values:
            node_values[val_k] = np.asscalar(self.nodes[node_idx][val_k])
        return node_values

    def apply_linkage(self, arr_h, arr_z, arr_qdrain, dt2d, dt1d):
        """
        """
        # update values from swmm
        self.update_nodes()
        swmm_c.apply_linkage_flow(arr_node=self.nodes, arr_h=arr_h, arr_z=arr_z,
                                  arr_qdrain=arr_qdrain, cell_surf=self.cell_surf,
                                  dt2d=dt2d, dt1d=dt1d, g=self.g,
                                  orifice_coeff=self.orifice_coeff,
                                  free_weir_coeff=self.free_weir_coeff,
                                  submerged_weir_coeff=self.submerged_weir_coeff)
