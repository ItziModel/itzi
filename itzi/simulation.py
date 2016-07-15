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
from __future__ import print_function
import warnings
from datetime import datetime, timedelta
import numpy as np
import copy

from superficialflow import SuperficialSimulation
from rasterdomain import RasterDomain
from massbalance import MassBal
import gis
import flow
import infiltration
from itzi_error import NullError


class SimulationManager(object):
    """Manage the general simulation:
    - update input values for each time-step
    - trigger the writing of results and statistics
    Accessed via the run() method
    """

    def __init__(self, sim_times, input_maps, output_maps, sim_param,
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

        # dictionaries of map names
        self.in_map_names = input_maps
        self.out_map_names = output_maps

        # data type of arrays
        self.dtype = dtype
        # simulation parameters
        self.sim_param = sim_param
        self.inf_model = self.sim_param['inf_model']

        # statistic file name
        self.stats_file = stats_file

        # instantiate a Igis object
        self.gis = gis.Igis(start_time=self.start_time,
                            end_time=self.end_time,
                            dtype=self.dtype,
                            mkeys=self.in_map_names.keys())
        self.gis.msgr.verbose(_(u"Reading maps information from GIS..."))
        self.gis.read(self.in_map_names)

        # instantiate simulation objects
        self.__set_models()

    def __set_models(self):
        """Instantiate models objects
        """
        # RasterDomain
        self.rast_domain = RasterDomain(self.dtype, self.gis,
                                        self.in_map_names, self.out_map_names)

        # Infiltration
        if self.inf_model == 'constant':
            self.infiltration = infiltration.InfConstantRate(self.rast_domain,
                                                             self.dtinf)
        elif self.inf_model == 'green-ampt':
            self.infiltration = infiltration.InfGreenAmpt(self.rast_domain,
                                                          self.dtinf)
        elif self.inf_model is None:
            self.infiltration = infiltration.InfNull(self.rast_domain,
                                                     self.dtinf)
        else:
            assert False, u"Unknow infiltration model: {}".format(self.inf_model)

        # SuperficialSimulation
        self.surf_sim = SuperficialSimulation(self.rast_domain,
                                              self.sim_param)

        # Instantiate Massbal object
        if self.stats_file:
            self.massbal = MassBal(self.stats_file, self.rast_domain,
                                   self.start_time, self.temporal_type)
        else:
            self.massbal = None
        # reporting object
        self.report = Report(self.gis, self.temporal_type, self.sim_param['hmin'],
                             self.massbal, self.rast_domain, self.start_time)
        return self

    def run(self):
        """Perform a full simulation
        including infiltration, superficial flow etc.,
        recording of data and mass_balance calculation
        """
        duration_s = self.duration.total_seconds()

        # dict of next time-step (datetime object)
        self.next_ts = {'end': self.end_time,
                        'rec': self.start_time + self.record_step}
        for k in ['inf', 'surf']:
            self.next_ts[k] = self.start_time
        # First time-step is forced
        self.nextstep = self.sim_time + self.dt

        self.gis.msgr.verbose(_(u"Starting time-stepping..."))
        while self.sim_time < self.end_time:
            self.sim_time_s = (self.sim_time - self.start_time).total_seconds()
            # display advance of simulation
            self.gis.msgr.percent(self.sim_time_s, duration_s, 1)
            # update input arrays
            self.rast_domain.update_input_arrays(self.sim_time)
            # recalculate the flow direction if DEM changed
            if self.rast_domain.isnew['z']:
                self.surf_sim.update_flow_dir()
            # step models
            self.step()
            # update simulation time
            self.sim_time += self.dt
        # write final report
        self.report.end(self.sim_time)
        return self

    def step(self):
        """Step each of the model if needed
        """
        # calculate infiltration
        if self.sim_time == self.next_ts['inf']:
            self.infiltration.solve_dt()
            # calculate when will happen the next time-step
            self.next_ts['inf'] += self.infiltration.dt
            self.rast_domain.populate_stat_array('inf', self.sim_time)
            self.infiltration.step()
            self.rast_domain.isnew['inf'] = True
        else:
            self.rast_domain.isnew['inf'] = False

        # calculate superficial flow #
        # update arrays of infiltration, rainfall etc.
        self.rast_domain.update_ext_array()
        # force time-step to be the general time-step
        self.surf_sim.dt = self.dt
        # step() raise NullError in case of NaN/NULL cell
        # if this happen, stop simulation and
        # output a map showing the errors
        try:
            self.surf_sim.step()
        except NullError:
            self.report.write_error_to_gis(self.surf_sim.arr_err)
            self.gis.msgr.fatal(_(u"{}: "
                                  u"Null value detected in simulation, "
                                  u"terminating").format(self.sim_time))
        # calculate when should happen the next surface time-step
        self.surf_sim.solve_dt()
        self.next_ts['surf'] += self.surf_sim.dt

        if self.massbal:
            self.massbal.add_value('tstep', self.dt.total_seconds())
        # write simulation results
        if self.sim_time >= self.next_ts['rec']:
            self.gis.msgr.verbose(_(u"{}: Writting output maps...".format(self.sim_time)))
            self.report.step(self.sim_time)
            self.next_ts['rec'] += self.record_step
            # reset statistic maps
            self.rast_domain.reset_stats(self.sim_time)

        # find next step
        self.nextstep = min(self.next_ts.values())
        self.next_ts['surf'] = self.nextstep
        self.dt = self.nextstep - self.sim_time
        return self


class Report(object):
    """In charge of results reporting and writing
    """
    def __init__(self, igis, temporal_type, hmin, massbal, rast_dom, start_time):
        self.record_counter = 0
        self.gis = igis
        self.temporal_type = temporal_type
        self.out_map_names = rast_dom.out_map_names
        self.hmin = hmin
        self.rast_dom = rast_dom
        self.massbal = massbal
        # a dict containing lists of maps written to gis to be registered
        self.output_maplist = {k: [] for k in self.out_map_names.keys()}
        self.last_step = start_time

    def step(self, sim_time):
        """write results at given time-step
        """
        assert isinstance(sim_time, datetime)
        interval_s = (sim_time-self.last_step).total_seconds()
        self.output_arrays = self.rast_dom.get_output_arrays(interval_s, sim_time)
        self.write_results_to_gis(sim_time)
        if self.massbal:
            self.write_mass_balance(sim_time)
        self.record_counter += 1
        self.last_step = copy.copy(sim_time)
        return self

    def end(self, sim_time):
        """write last mass balance
        register maps in gis
        write max level maps
        """
        assert isinstance(sim_time, datetime)
        if self.massbal:
            self.write_mass_balance(sim_time)
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
        for k, arr in self.output_arrays.iteritems():
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
        for mkey, lst in self.output_maplist.iteritems():
            strds_name = self.out_map_names[mkey]
            if strds_name is None:
                continue
            self.gis.register_maps_in_strds(mkey, strds_name, lst,
                                            self.temporal_type)
        return self
