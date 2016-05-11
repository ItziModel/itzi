# coding=utf8

"""
Copyright (C) 2015-2016  Laurent Courty

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
from structs import NodeData, NodeType, LinkData
import math
import collections
import swmm_error

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

    def swmm_open(self, input_file = None, report_file = None,
                  output_file = None):
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
        if route_code == 0:
            route_model = 'NO_ROUTING'  # no routing
        elif route_code == 1:
            route_model = 'SF'  # steady flow model
        elif route_code == 2:
            route_model = 'KW'  # kinematic wave model
        elif route_code == 3:
            route_model = 'EKW'  # extended kin. wave model
        elif route_code == 4:
            route_model = 'DW'  # dynamic wave model
        else:
            raise ValueError('Unknown routing model')
        return route_model

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

    def swmm_getLinkData(self, link_id=None):
        '''Retrieve link data using GESZ function
        link_id = a string
        '''
        # Return None if no node_id is given or if not a string
        if not isinstance(node_id, str):
            return None
        else:
            # Call the C function
            c_link_data = LinkData()
            err = self.c_swmm5.swmm_getLinkData(c.c_char_p(node_id),
                                                c.byref(c_link_data))
            if err != 0:
                raise swmm_error.SwmmError(err)
            return c_link_data

    def swmm_getNodeData(self, node_id=None):
        '''Retrieve node data using GESZ function
        node_id = a string
        '''
        # Return None if no node_id is given or if not a string
        if not isinstance(node_id, str):
            return None
        else:
            # Call the C function
            c_node_data = NodeData()
            err = self.c_swmm5.swmm_getNodeData(c.c_char_p(node_id),
                                                c.byref(c_node_data))
            if err != 0:
                raise swmm_error.SwmmError(err)
            return c_node_data

    def swmm_addNodeInflow(self, node_id, inflow=0):
        '''Add an inflow to a given node
        node_id: a node ID (string)
        inflow: an inflow in CFS (float)
        '''
        err = self.c_swmm5.swmm_addNodeInflow(c.c_char_p(node_id),
                                              c.c_double(inflow))
        if err != 0:
            raise swmm_error.SwmmError(err)
        return self

    def swmm_getNodeInflows(self):
        '''Get inflows of all nodes
        return a list of inflow values
        '''
        # get number of nodes
        nnodes = self.get_nobjects()['NODE']
        # create the result list
        node_inflows = [i for i in xrange(nnodes)]
        # transform it into a C array
        c_arr_flows = (c.c_double * len(node_inflows))(*node_inflows)
        err = self.c_swmm5.swmm_getNodeInflows(c_arr_flows)
        if err != 0:
            raise swmm_error.SwmmError(err)
        # Put back the result in Python style list
        node_inflows = [i for i in c_arr_flows]
        return node_inflows

    def swmm_getNodeOutflows(self):
        '''Get outflows of all nodes
        return a list of outflows values
        '''
        # get number of nodes
        nnodes = self.get_nobjects()['NODE']
        # create the result list
        node_outflows = [i for i in xrange(nnodes)]
        # transform it into a C array
        c_arr_flows = (c.c_double * len(node_outflows))(*node_outflows)
        err = self.c_swmm5.swmm_getNodeOutflows(c_arr_flows)
        if err != 0:
            raise swmm_error.SwmmError(err)
        # Put back the result in Python style list
        node_outflows = [i for i in c_arr_flows]
        return node_outflows

    def swmm_getNodeHeads(self):
        '''Get hydraulic head of all nodes
        return a list of hydraulic head values
        '''
        # get number of nodes
        nnodes = self.get_nobjects()['NODE']
        # create the result list
        node_heads = [i for i in xrange(nnodes)]
        # transform it into a C array
        c_arr_flows = (c.c_double * len(node_heads))(*node_heads)
        err = self.c_swmm5.swmm_getNodeHeads(c_arr_flows)
        if err != 0:
            raise swmm_error.SwmmError(err)
        # Put back the result in Python style list
        node_heads = [i for i in c_arr_flows]
        return node_heads

    def get_node_data(self, node_id=None):
        '''Retrieve data from a node in SI units
        node_id = a string of node name
        output a dict'''
        # retrieve the node data in US units
        c_node_data = self.swmm_getNodeData(node_id=node_id)
        # Make the conversion in a new dictionary
        node_data_si = {}
        node_data_si['inflow'] = c_node_data.inflow * self.foot ** 3
        node_data_si['outflow'] = c_node_data.outflow * self.foot ** 3
        node_data_si['head'] = c_node_data.head * self.foot
        node_data_si['crestElev'] = c_node_data.crestElev * self.foot
        return node_data_si

    def add_node_inflow(self, node_id, inflow=0):
        '''add an inflow in CMS to a given node'''
        # need to add a name validity check
        inflow_cfs = inflow / self.foot ** 3
        self.swmm_addNodeInflow(node_id=node_id, inflow=inflow_cfs)
        return self

    def routing_getRoutingStep(self):
        '''Get swmm routing time step'''
        route_code = c.c_int.in_dll(self.c_swmm5, 'RouteModel').value
        route_step = c.c_double.in_dll(self.c_swmm5, 'RouteStep').value
        c_func = self.c_swmm5.routing_getRoutingStep
        c_func.restype = c.c_double
        routing_step = c_func(c.c_int(route_code),
                              c.c_double(route_step))
        return routing_step

    #~ def run(self, input_file = None, report_file = None, output_file = None):
        #~ '''Runs a SWMM simulation by calling Python functions
        #~ '''
        #~ # open a SWMM project
        #~ self.swmm_open(input_file = input_file,
                       #~ report_file = report_file,
                       #~ output_file = output_file)
        #~ # initialize all processing systems
        #~ self.swmm_start()
        #~ # Computes the first step
        #~ err, elapsed_time = self.swmm_step()
        #~ # step through the simulation
        #~ while elapsed_time > 0.0 and err == 0:
            #~ # extend the simulation by one routing time step
            #~ err, elapsed_time = self.swmm_step(elapsed_time = elapsed_time)
        #~ # close all processing systems
        #~ self.swmm_end()
        #~ # close the project
        #~ self.swmm_close()
        #~ return err

    def c_run(self, input_file = None, report_file = None, output_file = None):
        '''Runs a swmm simulation by calling the C funtion
        '''
        err = self.c_swmm5.swmm_run(c.c_char_p(input_file),
                                    c.c_char_p(report_file),
                                    c.c_char_p(output_file))
        if err != 0:
            raise swmm_error.SwmmError(err)
        return self


class SwmmNode(object):
    '''Define a SWMM node object
    should be defined by a swmm simulation object and a node ID
    '''
    def __init__(self, swmm_object=None, node_id=None):
        self.swmm_sim = swmm_object
        if not self.swmm_sim.is_started:
            raise swmm_error.NotStartedError
        # need to add a node validity check
        self.node_id = node_id
        self.linkage_flow = 0
        self.g = 9.80665
        # node type correspondance
        node_types = {NodeType.STORAGE: 'storage',
                      NodeType.JUNCTION: 'junction',
                      NodeType.OUTFALL: 'outfall',
                      NodeType.DIVIDER: 'divider'}
        # retrieve the node data in US units
        self.c_node_data = self.swmm_sim.swmm_getNodeData(node_id=self.node_id)
        self.node_type = node_types[self.c_node_data.type]
        self.foot = self.swmm_sim.foot
        self.overflow_area = self.get_overflow_area()
        # calculate circumference from area (node considered circular)
        self.weir_width = math.pi * 2 * math.sqrt(self.overflow_area / math.pi)
        # length of weir in direction of flow
        self.weir_length = 0.1
        # set / update node data from SWMM
        self.update()

    def update(self):
        '''Retrieve node data from SWMM in SI units.
        To be done after each simulation time-step
        '''
        # retrieve the node data in US units
        self.c_node_data = self.swmm_sim.swmm_getNodeData(node_id=self.node_id)

        self.sub_index = self.c_node_data.subIndex
        self.invert_elev = self.c_node_data.invertElev * self.foot
        self.init_depth = self.c_node_data.initDepth * self.foot
        self.full_depth = self.c_node_data.fullDepth * self.foot
        self.sur_depth = self.c_node_data.surDepth * self.foot
        self.ponded_area = self.c_node_data.pondedArea * self.foot ** 2
        self.degree = self.c_node_data.degree
        self.updated = self.c_node_data.updated
        self.crown_elev = self.c_node_data.crownElev * self.foot
        self.inflow = self.c_node_data.inflow * self.foot ** 3
        self.outflow = self.c_node_data.outflow * self.foot ** 3
        self.losses = self.c_node_data.losses * self.foot ** 3
        self.volume = self.c_node_data.newVolume * self.foot ** 3
        self.full_volume = self.c_node_data.fullVolume * self.foot ** 3
        self.overflow = self.c_node_data.overflow * self.foot ** 3
        self.depth = self.c_node_data.newDepth * self.foot
        self.lat_flow = self.c_node_data.newLatFlow * self.foot ** 3
        self.head = self.c_node_data.head * self.foot
        self.crest_elev = self.c_node_data.crestElev * self.foot
        return self

    def add_inflow(self, inflow):
        '''add an inflow in CMS'''
        self.swmm_sim.add_node_inflow(node_id=self.node_id, inflow=inflow)
        return self

    def get_linkage_type(self, wse):
        '''select the linkage type (Chen et al.,2007)
        wse = Water Surface Elevation (from 2D superficial model)
        '''
        if wse <= self.crest_elev and self.head <= self.crest_elev:
            return 'no_linkage'
        elif (wse > self.crest_elev > self.head or
              wse <= self.crest_elev < self.head):
            return 'free_weir'
        elif (self.head >= self.crest_elev and
              wse > self.crest_elev and
              ((wse - self.crest_elev) <
               (self.overflow_area / self.weir_width))):
            return 'submerged_weir'
        elif ((wse - self.crest_elev) >=
              (self.overflow_area / self.weir_width)):
            return 'orifice'
        else:
            raise swmm_error.LinkTypeError(u"node '{id}':, Unknown linkage type "
                                           u"(wse: {wse}, crest: {crest}, head: {head})"
                                           u"".format(id=self.node_id, wse=wse,
                                                      crest=self.crest_elev, head=self.head))

    def set_crest_elev(self, z):
        '''Set the crest elevation according to the 2D dem
        the crest elevation could not be lower than ground
        update swmm Node.fullDepth if necessary
        '''
        new_crest = max(self.crest_elev, z)
        if new_crest != self.crest_elev:
            # set new FullDepth in swmm:
            c_new_FullDepth = c.c_double((new_crest - self.invert_elev) /
                                         self.foot)
            self.swmm_sim.c_swmm5.swmm_setNodeFullDepth(
                c.c_char_p(self.node_id), c_new_FullDepth)
            # update the object's data
            self.update()
        return self

    def get_overflow_area(self):
        '''Retrieve max. surface area of the node
        '''
        if self.node_type == 'storage':
            c_surf_area = self.swmm_sim.c_swmm5.node_getSurfArea(
                    c.c_int(self.node_id), c.c_double(self.full_depth))
            return c_surf_area.value * self.foot ** 2
        elif self.node_type == 'outfall':
            raise RuntimeError('Outfall cannnot overflow')
        else:
            return self.swmm_sim.get_MinSurfArea()

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

    def set_linkage_flow(self, wse, cell_surf, dt2d, dt1d):
        '''Calculate the flow between superficial and drainage models
        Cf. Chen et al.(2007)
        flow sign is :
         - negative when entering the drainage (leaving the 2D model)
         - positive when leaving the drainage (entering the 2D model)
        cell_surf: area in m² of the cell above the node
        dt2d: time-step of the 2d model in seconds
        dt1d: time-step of the drainage model in seconds
        '''
        water_surf_up = max(wse, self.head)
        water_surf_down = min(wse, self.head)
        upstream_depth = water_surf_up - self.crest_elev
        # get the weir coefficient
        weir_coeff = self.get_weir_coeff(upstream_depth=upstream_depth)
        orif_coeff = self.get_orifice_coeff()
        # calculate the flow
        self.linkage_type = self.get_linkage_type(wse)
        if self.linkage_type == 'no_linkage':
            unsigned_q = 0

        elif self.linkage_type == 'free_weir':
            unsigned_q = (weir_coeff * self.weir_width *
                          pow(upstream_depth, 3/2.) *
                          math.sqrt(2 * self.g))

        elif self.linkage_type == 'submerged_weir':
            unsigned_q = (weir_coeff * self.weir_width * upstream_depth *
                          math.sqrt(2 * self.g *
                                    (water_surf_up - water_surf_down)))

        elif self.linkage_type == 'orifice':
            unsigned_q = (orif_coeff * self.overflow_area *
                          math.sqrt(2 * self.g *
                                    (water_surf_up - water_surf_down)))
        else:
            assert False, "unknow linkage type"

        new_linkage_flow = math.copysign(unsigned_q, self.head - wse)

        # flow leaving the 2D domain can't drain the corresponding cell
        if new_linkage_flow < 0:
            dh = wse - max(self.crest_elev, self.head)
            assert dh > 0
            maxflow = dh * cell_surf * dt2d
            new_linkage_flow = max(new_linkage_flow, -maxflow)
        # flow leaving the drainage can't be higher than the water column above wse
        elif new_linkage_flow > 0:
            dh = self.head - wse
            assert dh > 0
            maxflow = dh * self.overflow_area * dt1d
            new_linkage_flow = min(new_linkage_flow, maxflow)

        self.linkage_flow = new_linkage_flow

        return self

    def get_weir_coeff(self, upstream_depth):
        '''Calculate the weir coefficient for linkage
        according to equation found in Bos (1985)
        upstream_depth: depend on the direction of the flow
           = depth of water in the inflow element above self.crest_elev
        '''
        return 0.93 + 0.1 * upstream_depth / self.weir_length

    def get_orifice_coeff(self):
        '''Return the orifice discharge coefficient.
        Using value found in Bos(1985)
        '''
        return 0.62

    def get_values_as_dict(self):
        """return a dict of node values
        """
        return {'type': self.node_type, 'overflow_area': self.overflow_area,
                'init_depth': self.init_depth, 'full_depth': self.full_depth,
                'surcharge_depth': self.sur_depth, 'depth': self.depth,
                'degree': self.degree, 'invert_elev': self.invert_elev,
                'crown_elev': self.crown_elev, 'inflow': self.inflow,
                'outflow': self.outflow, 'lateral_inflow': self.lat_flow,
                'linkage_flow': self.linkage_flow, 'losses': self.losses,
                'volume': self.volume, 'full_volume': self.full_volume,
                'overflow': self.overflow, 'ponded_area': self.ponded_area,
                'head': self.head, 'crest_elev': self.crest_elev}


class SwmmInputParser(object):
    """A parser for swmm input text file
    """
    def __init__(self, input_file):
        # list of sections keywords
        self.sections_kwd = ["title",  # project title
                             "option",  # analysis options
                             "report",   # output reporting instructions
                             "files",  # interface file options
                             "raingage",  # rain gage information
                             "evaporation",  # evaporation data
                             "temperature",  # air temperature and snow melt data
                             "subcatchment",  # basic subcatchment information
                             "subarea",  # subcatchment impervious/pervious sub-area data
                             "infiltration",  # subcatchment infiltration parameters
                             "aquifer",  # groundwater aquifer parameters
                             "groundwater",  # subcatchment groundwater parameters
                             "snowpack",  # subcatchment snow pack parameters
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
                             "losse",  # conduit entrance/exit losses and flap valve
                             "transect",  # transect geometry for conduits with irregular cross-sections
                             "control",  # rules that control pump and regulator operation
                             "pollutant",  # identifies the pollutants being analyzed
                             "landuse",  # land use categories
                             "coverage",  # assignment of land uses to subcatchments
                             "buildup",  # buildup functions for pollutants and land uses
                             "washoff",  # buildup functions for pollutants and land uses
                             "treatment",  # pollutant removal functions at conveyance system nodes
                             "dwf",  # baseline dry weather sanitary inflow at nodes
                             "pattern",  # periodic variation in dry weather inflow
                             "inflow",  # external hydrograph/pollutograph inflow at nodes
                             "loading",  # initial pollutant loads on subcatchments
                             "rdii",  # rainfall-dependent i/i information at nodes
                             "hydrograph",  # unit hydrograph data used to construct rdii inflows
                             "curve",  # x-y tabular data referenced in other sections
                             "timeserie",  # describes how a quantity varies over time
                             "lid_control",  # low impact development control information
                             "lid_usage",  # assignment of lid controls to subcatchments
                             'tag',  # ?
                             'map',  # provides dimensions and distance units for the map
                             'coordinate',  # coordinates of drainage system nodes
                             'vertice',  # coordinates of interior vertex points of curved drainage system links
                             'polygon', # coordinates of to vertex points of polygons that define a subcatchment boundary
                             'label',  # coordinates of user-defined map labels
                             'backdrop',  # coordinates of the bounding rectangle and file name of the backdrop
                             'symbol',  # coordinates of rain gage symbols
                             'profile',  # ?
                             'gwf',  # ?
                             'adjustment'] # ?
        # define junction container
        self.junction_values = ['x', 'y', 'elev', 'ymax', 'y0', 'ysur', 'apond']
        self.Junction = collections.namedtuple('Junction', self.junction_values)
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
