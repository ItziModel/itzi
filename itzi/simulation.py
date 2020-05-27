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

from itzi.surfaceflow import SurfaceFlowSimulation
from itzi.rasterdomain import RasterDomain
from itzi.massbalance import MassBal
from itzi.drainage import DrainageSimulation
import itzi.messenger as msgr
import itzi.gis as gis
import itzi.infiltration as infiltration
import itzi.hydrology as hydrology
from itzi.itzi_error import NullError


class SimulationManager():
    """Manage the general simulation:
    - update input values for each time-step
    - trigger the writing of results and statistics
    Accessed via the run() method
    """

    def __init__(self, sim_times, input_maps, output_maps, sim_param,
                 drainage_params, grass_params,
                 dtype=np.float32,
                 dtmin=timedelta(seconds=0.01),
                 stats_file=None):

        # read time parameters
        self.start_time = sim_times.start
        self.end_time = sim_times.end
        self.duration = sim_times.duration
        self.record_step = sim_times.record_step
        self.temporal_type = sim_times.temporal_type

        # set simulation time to start_time
        self.sim_time = self.start_time
        # First time-step is forced
        self.dt = dtmin  # Global time-step
        self.dtinf = sim_param['dtinf']

        # Temp region and mask
        self.region_id = grass_params['region']
        self.raster_mask_id = grass_params['mask']

        # dictionaries of map names
        self.in_map_names = input_maps
        self.out_map_names = output_maps

        # data type of arrays
        self.dtype = dtype
        # simulation parameters
        self.sim_param = sim_param
        self.inf_model = self.sim_param['inf_model']

        # drainage parameters
        self.drainage_params = drainage_params

        # statistic file name
        self.stats_file = stats_file

    def __set_models(self, igis):
        """Instantiate models objects
        """
        # RasterDomain
        msgr.debug(u"Setting up raster domain...")
        try:
            self.rast_domain = RasterDomain(self.dtype, igis,
                                            self.in_map_names, self.out_map_names)
        except MemoryError:
            msgr.fatal(u"Out of memory.")
        # Infiltration
        msgr.debug(u"Setting up raster infiltration...")
        inf_class = {'constant': infiltration.InfConstantRate,
                     'green-ampt': infiltration.InfGreenAmpt,
                     'null': infiltration.InfNull}
        try:
            self.infiltration = inf_class[self.inf_model](self.rast_domain,
                                                          self.dtinf)
        except KeyError:
            assert False, u"Unknow infiltration model: {}".format(self.inf_model)
        # Hydrology
        msgr.debug(u"Setting up hydrologic model...")
        self.hydrology = hydrology.Hydrology(self.rast_domain, self.dtinf,
                                             self.infiltration)
        # Surface flows simulation
        msgr.debug(u"Setting up surface model...")
        self.surf_sim = SurfaceFlowSimulation(self.rast_domain,
                                              self.sim_param)
        # Instantiate Massbal object
        if self.stats_file:
            msgr.debug(u"Setting up mass balance object...")
            self.massbal = MassBal(self.stats_file, self.rast_domain,
                                   self.start_time, self.temporal_type)
        else:
            self.massbal = None

        # Drainage
        if self.drainage_params['swmm_inp']:
            msgr.debug(u"Setting up drainage model...")
            self.drainage = DrainageSimulation(self.rast_domain,
                                               self.drainage_params,
                                               igis, self.sim_param['g'])
        else:
            self.drainage = None

        # reporting object
        msgr.debug(u"Setting up reporting object...")
        self.report = Report(igis, self.temporal_type,
                             self.sim_param['hmin'], self.massbal,
                             self.rast_domain,
                             self.drainage, self.drainage_params['output'])
        return self

    def initialize(self):
        """Set-up the simulation.
        """
        # instantiate simulation objects
        gis_kwargs = dict(start_time=self.start_time,
                          end_time=self.end_time,
                          dtype=self.dtype,
                          mkeys=self.in_map_names.keys(),
                          region_id=self.region_id,
                          raster_mask_id=self.raster_mask_id)
        self.igis = gis.Igis(**gis_kwargs)
        msgr.verbose(u"Setting up models...")
        self.__set_models(self.igis)
        msgr.verbose(u"Models set up")
        msgr.verbose(u"Reading maps information from GIS...")
        self.igis.read(self.in_map_names)

        # dict of next time-step (datetime object)
        self.next_ts = {'end': self.end_time,
                        'rec': self.start_time + self.record_step}
        for k in ['hyd', 'surf', 'drain']:
            self.next_ts[k] = self.start_time
        # case if no drainage simulation
        if not self.drainage:
            self.next_ts['drain'] = self.end_time
        # First time-step is forced
        self.nextstep = self.sim_time + self.dt
        # record initial state
        self.rast_domain.update_input_arrays(self.sim_time)
        self.report.step(self.sim_time)
        self.rast_domain.reset_stats(self.sim_time)
        return self

    def run(self):
        """Perform a full simulation
        including infiltration, surface flow etc.,
        recording of data and mass_balance calculation
        """
        sim_start_time = datetime.now()
        msgr.verbose(u"Starting time-stepping...")
        while self.sim_time < self.end_time:
            # display advance of simulation
            msgr.percent(self.start_time, self.end_time,
                         self.sim_time, sim_start_time)
            # step models
            self.step()
        return self

    def finalize(self):
        """Perform all operations after time stepping.
        """
        # write final report
        self.report.end(self.sim_time)
        self.igis.cleanup()
        return self

    def step(self):
        """Step each of the models if needed
        """
        # recalculate the flow direction if DEM changed
        if self.rast_domain.isnew['z']:
            self.surf_sim.update_flow_dir()

        # hydrology #
        if self.sim_time == self.next_ts['hyd']:
            self.hydrology.solve_dt()
            # calculate when will happen the next time-step
            self.next_ts['hyd'] += self.hydrology.dt
            self.hydrology.step()
            # update stat array
            self.rast_domain.populate_stat_array('inf', self.sim_time)
            self.rast_domain.populate_stat_array('capped_losses', self.sim_time)

        # drainage #
        if self.sim_time == self.next_ts['drain'] and self.drainage:
            self.drainage.solve_dt()
            # calculate when will happen the next time-step
            self.next_ts['drain'] += self.drainage.dt
            self.drainage.step()
            self.drainage.apply_linkage(self.dt.total_seconds())
            self.rast_domain.isnew['n_drain'] = True
            # update stat array
            self.rast_domain.populate_stat_array('n_drain', self.sim_time)
        else:
            self.rast_domain.isnew['n_drain'] = False

        # surface flow #
        # update arrays of infiltration, rainfall etc.
        self.rast_domain.update_ext_array()
        # force time-step to be the general time-step
        self.surf_sim.dt = self.dt
        # surf_sim.step() raise NullError in case of NaN/NULL cell
        # if this happen, stop simulation and
        # output a map showing the errors
        try:
            self.surf_sim.step()
        except NullError:
            self.report.write_error_to_gis(self.surf_sim.arr_err)
            msgr.fatal(u"{}: Null value detected in simulation, "
                       u"terminating".format(self.sim_time))
        # calculate when should happen the next surface time-step
        self.surf_sim.solve_dt()
        self.next_ts['surf'] += self.surf_sim.dt

        # send current time-step duration to mass balance object
        if self.massbal:
            self.massbal.add_value('tstep', self.dt.total_seconds())

        # Reporting #
        if self.sim_time >= self.next_ts['rec']:
            msgr.verbose(u"{}: Writing output maps...".format(self.sim_time))
            self.report.step(self.sim_time)
            self.next_ts['rec'] += self.record_step
            # reset statistic maps
            self.rast_domain.reset_stats(self.sim_time)

        # update input arrays. This is done at initialization as well.
        self.rast_domain.update_input_arrays(self.sim_time)
        # find next step
        self.nextstep = min(self.next_ts.values())
        # force the surface time-step to the lowest time-step
        self.next_ts['surf'] = self.nextstep
        self.dt = self.nextstep - self.sim_time
        # update simulation time
        self.sim_time += self.dt
        return self


class Report():
    """In charge of results reporting and writing
    """
    def __init__(self, igis, temporal_type, hmin, massbal, rast_dom,
                 drainage_sim, drainage_out):
        self.record_counter = 0
        self.gis = igis
        self.temporal_type = temporal_type
        self.out_map_names = rast_dom.out_map_names
        self.hmin = hmin
        self.rast_dom = rast_dom
        self.massbal = massbal
        self.drainage_sim = drainage_sim
        self.drainage_out = drainage_out
        self.drainage_values = {'records': []}
        # a dict containing lists of maps written to gis to be registered
        self.output_maplist = {k: [] for k in self.out_map_names.keys()}
        self.vector_drainage_maplist = []
        self.last_step = copy.copy(self.gis.start_time)

    def step(self, sim_time):
        """write results at given time-step
        """
        assert isinstance(sim_time, datetime)
        interval_s = (sim_time-self.last_step).total_seconds()
        self.output_arrays = self.rast_dom.get_output_arrays(interval_s, sim_time)
        self.write_results_to_gis(sim_time)
        if self.massbal:
            self.write_mass_balance(sim_time)
        if self.drainage_sim and self.drainage_out:
            self.save_drainage_values(sim_time)
        self.record_counter += 1
        self.last_step = copy.copy(sim_time)
        return self

    def end(self, sim_time):
        """Perform the last step
        register maps in gis
        write max level maps
        """
        assert isinstance(sim_time, datetime)
        # do the last step
        self.step(sim_time)
        self.gis.finalize()  # Make sure all maps are written
        # register maps and write max maps
        self.register_results_in_gis()
        if self.out_map_names['h']:
            self.write_hmax_to_gis()
        if self.out_map_names['v']:
            self.write_vmax_to_gis()
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
        # vector
        if self.drainage_sim and self.drainage_out:
            self.gis.register_maps_in_stds("Itzï drainage results",
                                           self.drainage_out,
                                           self.vector_drainage_maplist,
                                           'stvds',
                                           self.temporal_type)
        return self

    def save_drainage_values(self, sim_time):
        """Write vector map of drainage network
        """
        drainage_network = self.drainage_sim.drainage_network
        linking_elem = self.drainage_sim.linking_elements
        # format map name
        suffix = str(self.record_counter).zfill(4)
        map_name = "{}_{}".format(self.drainage_out, suffix)
        # write the map
        self.gis.write_vector_map(drainage_network, map_name, linking_elem)
        #
        self.vector_drainage_maplist.append((map_name, sim_time))
        return self
