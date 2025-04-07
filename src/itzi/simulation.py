# coding=utf8
"""
Copyright (C) 2015-2020 Laurent Courty

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
import warnings
from datetime import datetime, timedelta
import copy
import numpy as np
import pyswmm

from itzi.surfaceflow import SurfaceFlowSimulation
import itzi.rasterdomain as rasterdomain
from itzi.massbalance import MassBal
from itzi.drainage import DrainageSimulation, SwmmInputParser
import itzi.messenger as msgr
import itzi.infiltration as infiltration
import itzi.hydrology as hydrology
from itzi.itzi_error import NullError


def get_linked_nodes_list(node_objs, nodes_coor_dict, igis):
    """Check if the drainage nodes are inside the region and can be linked.
    Return a list of (node_obj, row, col)
    """
    linked_nodes_list = []
    for node in node_objs:
        node_id = node.nodeid
        coors = nodes_coor_dict[node_id]
        # a node without coordinates cannot be linked
        if coors is None or not igis.is_in_region(coors.x, coors.y):
            continue
        else:
            # get row and column
            row, col = igis.coor2pixel(coors)
            # populate list
            node_tuple = (node, int(row), int(col))
            linked_nodes_list.append(node_tuple)
    return linked_nodes_list


# correspondance between internal numpy arrays and map names
in_k_corresp = {'dem': 'dem', 'friction': 'friction', 'h': 'start_h',
                'y': 'start_y',
                'effective_porosity': 'effective_porosity',
                'capillary_pressure': 'capillary_pressure',
                'hydraulic_conductivity': 'hydraulic_conductivity',
                'in_inf': 'infiltration',
                'losses': 'losses',
                'rain': 'rain', 'inflow': 'inflow',
                'bcval': 'bcval', 'bctype': 'bctype'}


def create_simulation(sim_times, input_maps, output_maps, sim_param,
                      drainage_params, grass_params,
                      dtype=np.float32, stats_file=None):
    """A factory function that returns a SimulationManager object.
    """
    import itzi.gis as gis
    msgr.verbose(u"Setting up models...")
    # return error if output files exist
    gis.check_output_files(output_maps.values())
    msgr.debug('Output files OK')
    # GIS interface
    igis = gis.Igis(start_time=sim_times.start,
                    end_time=sim_times.end,
                    dtype=dtype,
                    mkeys=input_maps.keys(),
                    region_id=grass_params['region'],
                    raster_mask_id=grass_params['mask'])
    arr_mask = igis.get_npmask()
    msgr.verbose(u"Reading maps information from GIS...")
    igis.read(input_maps)
    # Timed arrays
    tarr = {}
    zeros_array = lambda: np.zeros(shape=raster_shape, dtype=dtype)
    for k in in_k_corresp.keys():
        tarr[k] = rasterdomain.TimedArray(in_k_corresp[k], igis, zeros_array)
    msgr.debug(u"Setting up raster domain...")
    # RasterDomain
    raster_shape = (igis.yr, igis.xr)
    try:
        raster_domain = rasterdomain.RasterDomain(dtype=dtype, arr_mask=arr_mask,
                                                  cell_shape=(igis.dx, igis.dy),
                                                  )
    except MemoryError:
        msgr.fatal(u"Out of memory.")
    # Infiltration
    inf_model = sim_param['inf_model']
    dtinf = sim_param['dtinf']
    msgr.debug(u"Setting up raster infiltration...")
    inf_class = {'constant': infiltration.InfConstantRate,
                 'green-ampt': infiltration.InfGreenAmpt,
                 'null': infiltration.InfNull}
    try:
        infiltration_model = inf_class[inf_model](raster_domain, dtinf)
    except KeyError:
        assert False, f"Unknow infiltration model: {inf_model}"
    # Hydrology
    msgr.debug(u"Setting up hydrologic model...")
    hydrology_model = hydrology.Hydrology(raster_domain, dtinf, infiltration_model)
    # Surface flows simulation
    msgr.debug(u"Setting up surface model...")
    surface_flow = SurfaceFlowSimulation(raster_domain, sim_param)
    # Instantiate Massbal object
    if stats_file:
        msgr.debug(u"Setting up mass balance object...")
        massbal = MassBal(stats_file, raster_domain,
                            sim_times.start, sim_times.temporal_type)
    else:
        massbal = None
    # Drainage
    if drainage_params['swmm_inp']:
        msgr.debug(u"Setting up drainage model...")
        swmm_sim = pyswmm.Simulation(drainage_params['swmm_inp'])
        swmm_inp = SwmmInputParser(drainage_params['swmm_inp'])
        # Select only the nodes inside the domain
        all_nodes = pyswmm.Nodes(swmm_sim)
        nodes_coors_dict = swmm_inp.get_nodes_id_as_dict()
        linked_nodes_list = get_linked_nodes_list(all_nodes, nodes_coors_dict, igis)
        drainage = DrainageSimulation(raster_domain,
                                      swmm_sim,
                                      drainage_params,
                                      linked_nodes_list,
                                      sim_param['g'])
    else:
        drainage = None
    # reporting object
    msgr.debug(u"Setting up reporting object...")
    report = Report(igis, sim_times.temporal_type,
                    sim_param['hmin'], massbal,
                    output_maps, raster_domain,
                    drainage, drainage_params['output'],
                    sim_times.record_step)
    msgr.verbose(u"Models set up")
    simulation = Simulation(sim_times.start, sim_times.end, raster_domain,
                            hydrology_model, surface_flow, drainage, report)
    return (simulation, tarr)


class Simulation():
    """
    """

    _array_keys = ['elevation',
                   'manning_n',
                   'depth',
                   'effective_porosity', 'capillary_pressure',
                   'hydraulic_conductivity',
                   'infiltration', 'losses', 'rain',
                   'etp', 'effective_precipitation',
                   'inflow',
                   'bcval', 'bctype',
                   'hmax',
                   'ext',
                   'hfe', 'hfs',
                   'qe', 'qs', 'qe_new', 'qs_new',
                   'ue', 'us',
                   'v', 'vdir',
                   'vmax',
                   'froude',
                   'n_drain',
                   'capped_losses',
                   'dire', 'dirs']

    def __init__(self, start_time, end_time, raster_domain,
                 hydrology_model, surface_flow, drainage_model, report):
        self.raster_domain = raster_domain
        self.start_time = start_time
        self.end_time = end_time
        # set simulation time to start_time
        self.sim_time = self.start_time
        self.raster_domain = raster_domain
        self.hydrology_model = hydrology_model
        self.drainage_model = drainage_model
        self.surface_flow = surface_flow
        self.report = report
        # First time-step is forced
        self.dt = timedelta(seconds=0.001)
        self.nextstep = self.sim_time + self.dt
        # dict of next time-step (datetime object)
        self.next_ts = {'end': self.end_time}
        for k in ['hydrology', 'surface_flow', 'drainage', 'record']:
            self.next_ts[k] = self.start_time
        # case if no drainage simulation
        if not self.drainage_model:
            self.next_ts['drainage'] = self.end_time
        # Grid spacing
        self.spacing = (self.raster_domain.dy, self.raster_domain.dx)

    def update(self):
        # Reporting #
        if self.sim_time == self.next_ts['record']:
            msgr.verbose(u"{}: Writing output maps...".format(self.sim_time))
            self.report.step(self.sim_time)
            self.next_ts['record'] += self.report.dt

        # hydrology #
        if self.sim_time == self.next_ts['hydrology']:
            self.hydrology_model.solve_dt()
            # calculate when will happen the next time-step
            self.next_ts['hydrology'] += self.hydrology_model.dt
            self.hydrology_model.step()
            # update stat array
            self.raster_domain.populate_stat_array('inf', self.sim_time)
            self.raster_domain.populate_stat_array('capped_losses', self.sim_time)

        # drainage #
        if self.sim_time == self.next_ts['drainage'] and self.drainage_model:
            # self.drainage.solve_dt()
            self.drainage_model.step()
            self.drainage_model.apply_linkage(self.dt.total_seconds())
            # update stat array
            self.raster_domain.populate_stat_array('n_drain', self.sim_time)
            # calculate when will happen the next time-step
            self.next_ts['drainage'] += self.drainage_model.dt

        # surface flow #
        # update arrays of infiltration, rainfall etc.
        self.raster_domain.update_ext_array()
        # force time-step to be the general time-step
        self.surface_flow.dt = self.dt
        # surface_flow.step() raise NullError in case of NaN/NULL cell
        # if this happen, stop simulation and
        # output a map showing the errors
        try:
            self.surface_flow.step()
        except NullError:
            self.report.write_error_to_gis(self.surface_flow.arr_err)
            msgr.fatal(u"{}: Null value detected in simulation, "
                       u"terminating".format(self.sim_time))
        # calculate when should happen the next surface time-step
        self.surface_flow.solve_dt()
        self.next_ts['surface_flow'] += self.surface_flow.dt

        # send current time-step duration to mass balance object
        if self.report.massbal:
            self.report.massbal.add_value('tstep', self.dt.total_seconds())

        self.find_dt()
        # update simulation time
        self.sim_time += self.dt
        return self

    def update_until(self, then):
        """Run the simulation until a time in seconds after start_time
        """
        assert isinstance(then, timedelta)
        end_time = self.start_time + then
        if end_time <= self.sim_time:
            raise ValueError('End time must be superior to current time')
        # Set temp end time (shorten last time step if necessary)
        self.next_ts['temp_end'] = end_time
        while self.sim_time < end_time:
            self.update()
        del self.next_ts['temp_end']
        # Make sure everything went well
        assert self.sim_time == end_time
        return self

    def finalize(self):
        # write final report
        self.report.end(self.sim_time)

    def set_array(self, arr_id, arr):
        """Set an array of the simulation domain
        """
        assert isinstance(arr_id, str)
        assert isinstance(arr, np.ndarray)
        if arr_id in ['inflow', 'rain']:
            self.raster_domain.populate_stat_array(arr_id, self.sim_time)
        self.raster_domain.update_array(arr_id, arr)
        if arr_id == 'dem':
            self.surface_flow.update_flow_dir()
        return self

    def get_array(self, arr_id):
        """
        """
        assert isinstance(arr_id, str)
        return self.raster_domain.get_array(arr_id)

    def find_dt(self):
        """find next step"""
        self.nextstep = min(self.next_ts.values())
        # force the surface time-step to the lowest time-step
        self.next_ts['surface_flow'] = self.nextstep
        self.dt = self.nextstep - self.sim_time
        return self


class Report():
    """In charge of results reporting and writing
    """
    def __init__(self, igis, temporal_type, hmin, massbal, out_map_names,
                rast_dom, drainage_sim, drainage_out, dt):
        self.record_counter = 0
        self.gis = igis
        self.temporal_type = temporal_type
        self.out_map_names = out_map_names
        self.hmin = hmin
        self.rast_dom = rast_dom
        self.massbal = massbal
        self.drainage_sim = drainage_sim
        self.drainage_out = drainage_out
        self.drainage_values = {'records': []}
        # a dict containing lists of maps written to gis to be registered
        self.output_maplist = {k: [] for k in self.out_map_names.keys()}
        self.vector_drainage_maplist = []
        self.output_arrays = {}
        self.dt = dt
        self.last_step = copy.copy(self.gis.start_time)

    def step(self, sim_time):
        """write results at given time-step
        """
        assert isinstance(sim_time, datetime)
        interval_s = (sim_time-self.last_step).total_seconds()
        self.get_output_arrays(interval_s, sim_time)
        self.write_results_to_gis(sim_time)
        if self.massbal:
            self.write_mass_balance(sim_time)
        self.record_counter += 1
        self.last_step = copy.copy(sim_time)
        self.rast_dom.reset_stats(sim_time)
        return self

    def end(self, sim_time):
        """Perform the last step
        register maps in gis
        write max level maps
        """
        assert isinstance(sim_time, datetime)
        # do the last step
        self.step(sim_time)
        # Make sure all maps are written in the background process
        self.gis.finalize()
        # register maps and write max maps
        self.register_results_in_gis()
        if self.out_map_names['h']:
            self.write_hmax_to_gis()
        if self.out_map_names['v']:
            self.write_vmax_to_gis()
        # Cleanup the GIS state
        self.gis.cleanup()
        return self

    def get_output_arrays(self, interval_s, sim_time):
        """Returns a dict of unmasked arrays to be written to the disk
        """
        for k in self.out_map_names:
            if self.out_map_names[k] is not None:
                if k == 'wse':
                    h = self.rast_dom.get_unmasked('h')
                    wse = h + self.rast_dom.get_array('dem')
                    self.output_arrays['wse'] = wse
                elif k == 'qx':
                    qx = self.rast_dom.get_unmasked('qe_new') * self.rast_dom.dy
                    self.output_arrays['qx'] = qx
                elif k =='qy':
                    qy = self.rast_dom.get_unmasked('qs_new') * self.rast_dom.dx
                    self.output_arrays['qy'] = qy
                # Created volume (total since last record)
                elif k == 'verror':
                    self.rast_dom.populate_stat_array('capped_losses', sim_time)  # This is weird
                    verror = self.rast_dom.get_unmasked('st_herr') * self.rast_dom.cell_surf
                    self.output_arrays['verror'] = verror
                elif k == 'drainage_stats' and interval_s:
                    self.rast_dom.populate_stat_array('n_drain', sim_time)
                    self.output_arrays['drainage_stats'] = self.rast_dom.get_unmasked('st_ndrain') / interval_s
                elif k not in ['drainage_stats','boundaries', 'inflow', 'infiltration', 'rainfall']:
                    self.output_arrays[k] = self.rast_dom.get_unmasked(k)
                else:
                    continue
        # statistics (average of last interval)
        if interval_s:
            mmh_to_ms = 1000. * 3600.
            if self.out_map_names['boundaries'] is not None:
                self.output_arrays['boundaries'] = self.rast_dom.get_unmasked('st_bound') / interval_s
            if self.out_map_names['inflow'] is not None:
                self.rast_dom.populate_stat_array('inflow', sim_time)
                self.output_arrays['inflow'] = self.rast_dom.get_unmasked('st_inflow') / interval_s
            if self.out_map_names['losses'] is not None:
                self.rast_dom.populate_stat_array('capped_losses', sim_time)
                self.output_arrays['losses'] = self.rast_dom.get_unmasked('st_losses') / interval_s

            if self.out_map_names['infiltration'] is not None:
                self.rast_dom.populate_stat_array('inf', sim_time)
                self.output_arrays['infiltration'] = (self.rast_dom.get_unmasked('st_inf') /
                                              interval_s) * mmh_to_ms
            if self.out_map_names['rainfall'] is not None:
                self.rast_dom.populate_stat_array('rain', sim_time)
                self.output_arrays['rainfall'] = (self.rast_dom.get_unmasked('st_rain') /
                                          interval_s) * mmh_to_ms
        return self

    def write_mass_balance(self, sim_time):
        """Append mass balance values to file
        """
        self.massbal.write_values(sim_time)
        return self

    def write_results_to_gis(self, sim_time):
        """Format the name of each maps using the record number as suffix
        Send a tuple (array, name, key) to the GIS writing function.
        """
        for k, arr in self.output_arrays.items():
            if isinstance(arr, np.ndarray):
                suffix = str(self.record_counter).zfill(4)
                map_name = "{}_{}".format(self.out_map_names[k], suffix)
                # Export depth if above hmin. If not, export NaN
                if k == 'h':
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        arr[arr <= self.hmin] = np.nan
                # write the raster
                self.gis.write_raster_map(arr, map_name, k)
                # add map name and time to the corresponding list
                self.output_maplist[k].append((map_name, sim_time))
        return self

    def write_error_to_gis(self, arr_error):
        '''Write a given depth array to the GIS
        '''
        map_h_name = "{}_error".format(self.out_map_names['h'])
        self.gis.write_raster_map(self.rast_dom.get_unmasked('h'),
                                  map_h_name, 'h')
        # add map name to the revelant list
        self.output_maplist['h'].append(map_h_name)
        return self

    def write_hmax_to_gis(self):
        '''Write a max depth array to the GIS
        '''
        arr_hmax_unmasked = self.rast_dom.get_unmasked('hmax')
        map_hmax_name = "{}_max".format(self.out_map_names['h'])
        self.gis.write_raster_map(arr_hmax_unmasked, map_hmax_name, 'h')
        return self

    def write_vmax_to_gis(self):
        '''Write a max flow speed array to the GIS
        '''
        arr_vmax_unmasked = self.rast_dom.get_unmasked('vmax')
        map_vmax_name = "{}_max".format(self.out_map_names['v'])
        self.gis.write_raster_map(arr_vmax_unmasked, map_vmax_name, 'v')
        return self

    def register_results_in_gis(self):
        """Register the generated maps in the temporal database
        Loop through output names
        if no output name is provided, don't do anything
        if name is populated, create a strds of the right temporal type
        and register the corresponding listed maps
        """
        # rasters
        for mkey, lst in self.output_maplist.items():
            strds_name = self.out_map_names[mkey]
            if strds_name is None:
                continue
            self.gis.register_maps_in_stds(mkey, strds_name, lst, 'strds',
                                           self.temporal_type)
        return self
