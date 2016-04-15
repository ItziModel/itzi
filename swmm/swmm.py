#! /usr/bin/python
# coding=utf8

'''A ctypes interface for the SWMM5 library
'''

from __future__ import division
import ctypes as c
from structs import NodeData, NodeType
import math
from collections import namedtuple
from swmm_error import SwmmError, NotOpenError

class Swmm5(object):
    '''A class implementing high-level swmm5 functions.
    '''
    def __init__(self, swmm_so='./source/swmm5.so'):
        self.c_swmm5 = c.CDLL(swmm_so)
        self.foot = 0.3048  # foot to metre
        self.is_open = False
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
            raise SwmmError(err)
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
        err = self.c_swmm5.swmm_start(c.c_int(save_results))
        if err != 0:
            raise SwmmError(err)
        return self

    def swmm_end(self):
        '''Ends a swmm simulation
        '''
        err = self.c_swmm5.swmm_end()
        if err != 0:
            raise SwmmError(err)
        return self

    def swmm_step(self):
        '''Advances the simulation by one routing time step
        '''
        c_elapsed_time = c.c_double(self.elapsed_time)
        err = self.c_swmm5.swmm_step(c.byref(c_elapsed_time))
        self.elapsed_time = c_elapsed_time.value
        if err != 0:
            raise SwmmError(err)
        return self

    def get_RouteModel(self):
        '''Get the node minimal surface area
        (storage node could be larger)
        '''
        if self.is_open == False:
            raise NotOpenError

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

    def get_MinSurfArea(self):
        '''Get the node minimal surface area in sqm
        (storage node could be larger)
        '''
        if self.is_open == False:
            raise NotOpenError
        if self.routing_model == 'DW':
            area = c.c_double.in_dll(self.c_swmm5, 'MinSurfArea').value
            return area * self.foot ** 2  # return SI value
        else:
            raise RuntimeError('MinSurfArea only valid for Dynamic Wave routing')

    def force_ponding(self):
        '''Force model to allow ponding
        '''
        if self.is_open == False:
            raise NotOpenError

        AllowPonding = c.c_int.in_dll(self.c_swmm5, 'AllowPonding').value
        if AllowPonding != 1:
            print('Forcing ponding at nodes...')
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

    def read_nodes_coordinates(self):
        """return a list of
        """
        is_coor = False
        self.coor = {}
        Point = namedtuple('Point', ['x','y'])
        with open(self.input_file, 'r') as inp:
            for line in inp:
                if line.startswith(';'):
                    continue
                if line.startswith('[COOR'):
                    is_coor = True
                    continue
                if is_coor and line.startswith('['):
                    is_coor = False
                    break
                if is_coor:
                    line = line.split()
                    self.coor[line[0]] = Point(x=float(line[1]),
                                               y=float(line[2]))
        return self

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
        if weighting_factor == None:
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
                raise SwmmError(err)
            return c_node_data

    def swmm_addNodeInflow(self, node_id=None, inflow=None):
        '''Add an inflow to a given node
        node_id: a node ID (string)
        inflow: an inflow in CFS (float)
        '''
        err = self.c_swmm5.swmm_addNodeInflow(c.c_char_p(node_id),
                                              c.c_double(inflow))
        if err != 0:
            raise SwmmError(err)
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
            raise SwmmError(err)
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
            raise SwmmError(err)
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
            raise SwmmError(err)
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

    def add_node_inflow(self, node_id=None, inflow=0):
        '''add an inflow in CMS to a given node'''
        # need to add a name validity check
        if node_id == None:
            raise ValueError('need a node_id')
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

    def run(self, input_file = None, report_file = None, output_file = None):
        '''Runs a SWMM simulation by calling Python functions
        '''
        # open a SWMM project
        self.swmm_open(input_file = input_file,
                       report_file = report_file,
                       output_file = output_file)
        # initialize all processing systems
        self.swmm_start()
        # Computes the first step
        err, elapsed_time = self.swmm_step()
        # step through the simulation
        while elapsed_time > 0.0 and err == 0:
            # extend the simulation by one routing time step
            err, elapsed_time = self.swmm_step(elapsed_time = elapsed_time)
        # close all processing systems
        self.swmm_end()
        # close the project
        self.swmm_close()
        return err

    def c_run(self, input_file = None, report_file = None, output_file = None):
        '''Runs a swmm simulation by calling the C funtion
        '''
        err = self.c_swmm5.swmm_run(c.c_char_p(input_file),
                                    c.c_char_p(report_file),
                                    c.c_char_p(output_file))
        if err != 0:
            raise SwmmError(err)
        return self


class SwmmNode(object):
    '''Define a SWMM node object
    should be defined by a swmm simulation object and a node ID
    '''
    def __init__(self, swmm_object=None, node_id=None):
        self.swmm_sim = swmm_object
        if not self.swmm_sim.is_open:
            raise NotOpenError
        # need to add a node validity check
        self.node_id = node_id
        self.linkage_flow = 0
        self.g = 9.80665
        # retrieve the node data in US units
        self.c_node_data = self.swmm_sim.swmm_getNodeData(node_id=self.node_id)
        self.node_type = self.c_node_data.type
        self.foot = self.swmm_sim.foot
        self.overflow_area = self.get_overflow_area()
        # calculate circumference from area (node considered circular)
        self.weir_width = math.pi * 2 * math.sqrt(self.overflow_area / math.pi)
        # length of weir in direction of flow
        self.weir_length = 0.1
        # set / update node data from SWMM
        self.update()
        self.set_coordinates()

    def update(self):
        '''Update node data in SI units.
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

    def get_linkage_type(self, wse=None):
        '''select the linkage type (Chen et al.,2007)
        wse = Water Surface Elevation (from 2D superficial model)
        '''
        if wse <= self.crest_elev and self.head < self.crest_elev:
            return 'no_linkage'
        elif (wse > self.crest_elev > self.head or
              wse < self.crest_elev <= self.head):
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
            raise RuntimeError('Unknown linkage type')

    def set_coordinates(self):
        '''Get the node coordinate
        '''
        self.coor = self.swmm_sim.coor[self.node_id]
        return self

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
        if self.node_type == NodeType.STORAGE:
            c_surf_area = self.swmm_sim.c_swmm5.node_getSurfArea(
                    c.c_int(self.node_id), c.c_double(self.full_depth))
            return c_surf_area.value * self.foot ** 2
        elif self.node_type == NodeType.OUTFALL:
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

    def set_linkage_flow(self, wse=None):
        '''Calculate the flow between superficial and drainage models
        Cf. Chen et al.(2007)
        flow sign is :
         - negative when entering the drainage (leaving the 2D model)
         - positive when leaving the drainage (entering the 2D model)
        '''
        water_surf_up = max(wse, self.head)
        water_surf_down = min(wse, self.head)
        upstream_depth = water_surf_up - self.crest_elev
        # get the weir coefficient
        weir_coeff = self.get_weir_coeff(upstream_depth=upstream_depth)
        orif_coeff = self.get_orifice_coeff()
        # calculate the flow
        if self.get_linkage_type(wse=wse) == 'no_linkage':
            q_linkage = 0

        elif self.get_linkage_type(wse=wse) == 'free_weir':
            unsigned_q = (weir_coeff * self.weir_width *
                          pow(upstream_depth, 3/2.) *
                          math.sqrt(2 * self.g))

        elif self.get_linkage_type(wse=wse) == 'submerged_weir':
            unsigned_q = (weir_coeff * self.weir_width * upstream_depth *
                          math.sqrt(2 * self.g *
                                    abs(water_surf_up - water_surf_down)))

        elif self.get_linkage_type(wse=wse) == 'orifice':
            unsigned_q = (orif_coeff * self.overflow_area *
                          math.sqrt(2 * self.g *
                                    abs(water_surf_up - water_surf_down)))
        else:
            assert False, "unknow linkage type"

        # set a max flow
        #~ unsigned_q = min(unsigned_q, 4)

        new_linkage_flow = math.copysign(unsigned_q, self.head - wse)
        #~ # force a step at zero whan changing sign
        #~ if (
            #~ (new_linkage_flow > 0 and self.linkage_flow < 0)
            #~ or
            #~ (new_linkage_flow < 0 and self.linkage_flow > 0)):
                #~ new_linkage_flow = 0

        self.linkage_flow = new_linkage_flow

        return self

    def get_weir_coeff(self, upstream_depth=None):
        '''Calculate the weir coefficient for linkage
        according to equation found in Bos (1985)
        upstream_depth: depend on the direction of the flow
           = depth of water in the inflow element above self.crest_elev
        '''
        if upstream_depth == None:
            raise RuntimeError('upstream_depth mandatory')
        #~ return 0.93 + 0.1 * upstream_depth / self.weir_length
        return 0.5

    def get_orifice_coeff(self):
        '''Calculate the orifice discharge coefficient
        '''
        # need to find an actual formula
        return 0.5
