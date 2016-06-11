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
import csv
import warnings
from datetime import datetime, timedelta
import numpy as np

import domain
from rasterdomain import RasterDomain
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

    def __init__(self, record_step, input_maps, output_maps,
                 dtype=np.float32,
                 stats_file=None,
                 start_time=datetime(1, 1, 1),
                 end_time=datetime(1, 1, 1),
                 sim_duration=timedelta(0),
                 sim_param=None):
        assert isinstance(start_time, datetime), \
            "start_time not a datetime object!"
        assert isinstance(sim_duration, timedelta), \
            "sim_duration not a timedelta object!"
        assert sim_duration >= timedelta(0), "sim_duration is negative!"
        assert isinstance(record_step, timedelta), \
            "record_step not a timedelta object!"
        assert record_step > timedelta(0), "record_step must be > 0"
        assert sim_param is not None

        self.record_step = record_step
        self.start_time = start_time
        self.set_duration(end_time, sim_duration)
        # set simulation time to start_time a the beginning
        self.sim_time = self.start_time
        # dt
        self.dt = sim_param['dtmax']
        self.dtinf = sim_param['dtinf']
        # set temporal type of results
        self.set_temporal_type()

        # dictionaries of map names
        self.in_map_names = input_maps
        self.out_map_names = output_maps

        self.dtype = dtype
        # simulation parameters
        self.sim_param = sim_param

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
        self.set_models()

        # a dict containing lists of maps written to gis to be registered
        self.output_maplist = {k: [] for k in self.out_map_names.keys()}

    def set_duration(self, end_time, sim_duration):
        """If sim_duration is given, end_time is ignored
        This is normally checked upstream
        """
        if not sim_duration:
            self.duration = end_time - self.start_time
            self.end_time = end_time
        else:
            self.duration = sim_duration
            self.end_time = self.start_time + sim_duration
        return self

    def set_temporal_type(self):
        """A start_time equal to datetime.min means user did not
        provide a start_time. Therefore a relative temporal type
        is set for results writing.
        It's a potential problem in case a simulation is set to actually
        starts on 0001-01-01 00:00, as the results will be written
        in relative STDS.
        """
        self.temporal_type = 'absolute'
        if self.start_time == datetime.min:
            self.temporal_type = 'relative'
        return self

    def set_models(self):
        """Instantiate models objects
        """
        # RasterDomain
        self.rast_domain = RasterDomain(self.dtype, self.gis,
                                        self.in_map_names, self.out_map_names)

        # Infiltration. Coherence of input maps is checked upstream
        if self.in_map_names['in_inf']:
            self.infiltration = infiltration.InfConstantRate(self.rast_domain)
        elif self.in_map_names['in_cap_pressure']:
            self.infiltration = infiltration.InfGreenAmpt(self.rast_domain)
        else:
            self.infiltration = infiltration.InfNull(self.rast_domain)

        # SuperficialSimulation
        self.surf_sim = domain.SuperficialSimulation(self.rast_domain,
                                                     self.sim_param)

        # Instantiate Massbal object
        if self.stats_file:
            self.massbal = MassBal(self.stats_file, self.rast_domain)
        else:
            self.massbal = None
        return self

    def run(self):
        """Perform a full simulation
        including infiltration, superficial flow and drainage,
        recording of data and mass_balance calculation
        """
        record_counter = 1
        last_inf = 0.
        duration_s = self.duration.total_seconds()

        self.gis.msgr.verbose(_(u"Starting time-stepping..."))
        while self.sim_time < self.end_time:
            # display advance of simulation
            self.gis.msgr.percent(self.surf_sim.sim_clock, duration_s, 1)

            # Calculate when will happen the next records writing
            next_record = record_counter * self.record_step.total_seconds()

            # update arrays
            self.rast_domain.update_input_arrays(self.sim_time)
            # recalculate the flow direction if DEM changed
            if self.rast_domain.isnew['z']:
                self.surf_sim.update_flow_dir()

            # calculate infiltration
            self.set_inf_forced_timestep(next_record)
            self.infiltration.set_dt(self.dtinf, self.surf_sim.sim_clock,
                                     self.inf_forced_ts)
            if last_inf + self.infiltration.dt >= self.surf_sim.sim_clock:
                self.infiltration.step()
                last_inf = float(self.surf_sim.sim_clock)
                self.rast_domain.isnew['inf'] = True
            else:
                self.rast_domain.isnew['inf'] = False

            # Update external arrays
            self.rast_domain.update_ext_array()

            # next forced flow time-step
            next_ts = self.next_forced_timestep()
            # step() raise NullError in case of NaN/NULL cell
            # if this happen, stop simulation and
            # output a map showing the errors
            try:
                self.surf_sim.step(next_ts)
            except NullError:
                self.write_error_to_gis(self.surf_sim.arr_err)
                self.gis.msgr.fatal(_(u"Null value detected "
                                      u"in simulation at time {}, "
                                      u"terminating").format(self.sim_time))

            # update simulation time and dt
            self.sim_time = (self.start_time +
                             timedelta(seconds=self.surf_sim.sim_clock))
            self.dt = self.surf_sim.dt
            if self.massbal:
                self.massbal.add_value('tstep', self.dt)
            # write simulation results
            rec_time = self.surf_sim.sim_clock / self.record_step.total_seconds()
            if rec_time >= record_counter:
                self.gis.msgr.verbose(_(u"Writting output map..."))
                self.output_arrays = self.surf_sim.get_output_arrays(self.out_map_names)
                self.write_results_to_gis(record_counter)
                record_counter += 1
                if self.massbal:
                    self.write_mass_balance(self.surf_sim.sim_clock)
        # register generated maps in GIS
        self.register_results_in_gis()
        if self.out_map_names['out_h']:
            self.write_hmax_to_gis()
        if self.out_map_names['out_v']:
            self.write_vmax_to_gis()
        return self

    def write_mass_balance(self, sim_clock):
        '''
        '''
        if self.temporal_type == 'absolute':
            self.massbal.write_values(self.sim_time)
        elif self.temporal_type == 'relative':
            self.massbal.write_values(sim_clock)
        else:
            assert False, "unknown temporal type!"
        return self

    def next_forced_timestep(self):
        """return the future time in seconds at which the superficial flow
        model will be forced to step.
        default to next forced time-step of infiltration model
        """
        return self.inf_forced_ts

    def set_inf_forced_timestep(self, next_record):
        """Given a next superficial flow time-step in seconds as entry,
        return the future time in seconds at which the infiltration
        model will be forced to step.
        """
        self.inf_forced_ts = min(next_record, self.duration.total_seconds())
        return self

    def write_results_to_gis(self, record_counter):
        """Format the name of each maps using the record number as suffix
        Send a couple array, name to the GIS writing function.
        """
        for k, arr in self.output_arrays.iteritems():
            if isinstance(arr, np.ndarray):
                suffix = str(record_counter).zfill(4)
                map_name = "{}_{}".format(self.out_map_names[k], suffix)
                # Export depth if above hfmin. If not, export NaN
                if k == 'out_h':
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        arr[arr <= self.sim_param['hmin']] = np.nan
                # write the raster
                self.gis.write_raster_map(arr, map_name, k)
                # add map name and time to the corresponding list
                self.output_maplist[k].append((map_name, self.sim_time))
        return self

    def write_error_to_gis(self, arr_error):
        '''Write a given depth array and boolean error array to the GIS
        '''
        map_h_name = "{}_error".format(self.out_map_names['out_h'])
        self.gis.write_raster_map(self.rast_domain.get_unmasked('h'),
                                  map_h_name, 'out_h')
        # add map name to the revelant list
        self.output_maplist['out_h'].append(map_h_name)
        return self

    def write_hmax_to_gis(self):
        '''Write a given depth array to the GIS
        '''
        arr_hmax_unmasked = self.rast_domain.get_unmasked('hmax')
        map_hmax_name = "{}_max".format(self.out_map_names['out_h'])
        self.gis.write_raster_map(arr_hmax_unmasked, map_hmax_name, 'out_h')
        return self

    def write_vmax_to_gis(self):
        '''Write a given flow speed array to the GIS
        '''
        arr_vmax_unmasked = self.rast_domain.get_unmasked('vmax')
        map_vmax_name = "{}_max".format(self.out_map_names['out_v'])
        self.gis.write_raster_map(arr_vmax_unmasked, map_vmax_name, 'out_v')
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


class MassBal(object):
    """Follow-up the mass balance during the simulation run
    Mass balance error is the difference between the actual volume and
    the theoretical volume. The later is the old volume + input - output.
    Intended use:
    at each time-step, using add_value():
    individual simulation operations send relevant values to the MassBal object
    at each record time, using write_values():
    averaged or cumulated values for the considered time difference are
    written to a CSV file
    """
    def __init__(self, file_name, rast_dom):
        self.dom = rast_dom
        # values to be written on each record time
        self.fields = ['sim_time',  # either seconds or datetime
                       'avg_timestep', '#timesteps',
                       'boundary_vol', 'rain_vol', 'inf_vol',
                       'inflow_vol', 'hfix_vol',
                       'domain_vol', 'vol_error', '%error',
                       'comp_duration', 'avg_cell_per_sec']
        # data written to file as one line
        self.line = dict.fromkeys(self.fields)
        # data collected during simulation
        self.sim_data = {'tstep': [], 'boundary_vol': [],
                         'rain_vol': [], 'inf_vol': [], 'inflow_vol': [],
                         'old_dom_vol': [], 'new_dom_vol': [],
                         #~ 'step_duration': [],
                         'hfix_vol': []}
        # set file name and create file
        self.file_name = self.set_file_name(file_name)
        self.create_file()

    def set_file_name(self, file_name):
        '''Generate output file name
        '''
        if not file_name:
            file_name = "{}_stats.csv".format(
                str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')))
        return file_name

    def create_file(self):
        '''Create a csv file and write headers
        '''
        with open(self.file_name, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()
        return self

    def read_values(self, dt):
        """Read values from RasterDomain
        """
        self.sim_data['tstep'].append(dt)
        self.sim_data['boundary_vol'].append(self.dom.boundary_vol)
        self.sim_data['rain_vol'].append(self.dom.rain_q * dt)
        self.sim_data['inf_vol'].append(self.dom.inf_q * dt)
        self.sim_data['inflow_vol'].append(self.dom.inflow_q * dt)
        self.sim_data['old_dom_vol'].append(dt)
        self.sim_data['new_dom_vol'].append(dt)
        #~ self.sim_data['step_duration'].append(dt)
        self.sim_data['hfix_vol'].append(dt)
        return self

    def add_value(self, key, value):
        '''add a value to sim_data
        '''
        assert key in self.sim_data, "unknown key!"
        self.sim_data[key].append(value)
        return self

    def write_values(self, sim_time):
        '''Calculate statistics and write them to the file
        '''
        # check if all elements have the same number of records
        rec_len = [len(l) for l in self.sim_data.values()]
        assert rec_len[1:] == rec_len[:-1], "inconsistent number of records!"

        self.line['sim_time'] = sim_time
        # number of time-step during the interval is the number of records
        self.line['#timesteps'] = len(self.sim_data['tstep'])
        # average time-step calculation
        elapsed_time = sum(self.sim_data['tstep'])
        avg_timestep = elapsed_time / self.line['#timesteps']
        self.line['avg_timestep'] = '{:.3f}'.format(avg_timestep)

        # sum of inflow (positive) / outflow (negative) volumes
        boundary_vol = sum(self.sim_data['boundary_vol'])
        self.line['boundary_vol'] = '{:.3f}'.format(boundary_vol)
        rain_vol = sum(self.sim_data['rain_vol'])
        self.line['rain_vol'] = '{:.3f}'.format(rain_vol)
        inf_vol = - sum(self.sim_data['inf_vol'])
        self.line['inf_vol'] = '{:.3f}'.format(inf_vol)
        inflow_vol = sum(self.sim_data['inflow_vol'])
        self.line['inflow_vol'] = '{:.3f}'.format(inflow_vol)
        hfix_vol = sum(self.sim_data['hfix_vol'])
        self.line['hfix_vol'] = '{:.3f}'.format(hfix_vol)

        # For domain volume, take last value(i.e. current)
        last_vol = self.sim_data['new_dom_vol'][-1]
        self.line['domain_vol'] = '{:.3f}'.format(last_vol)

        # mass error is difference between theoretical volume and actual volume
        first_vol = self.sim_data['old_dom_vol'][0]
        sum_ext_vol = sum([boundary_vol, rain_vol, inf_vol,
                           inflow_vol, hfix_vol])
        dom_vol_theor = first_vol + sum_ext_vol
        vol_error = last_vol - dom_vol_theor
        self.line['vol_error'] = '{:.3f}'.format(vol_error)
        if last_vol <= 0:
            self.line['%error'] = '-'
        else:
            self.line['%error'] = '{:.2%}'.format(vol_error / last_vol)

        # Performance
        #~ comp_duration = sum(self.sim_data['step_duration'])
        #~ self.line['comp_duration'] = '{:.3f}'.format(comp_duration)
        #~ # Average step computation time
        #~ avg_comp_time = comp_duration / rec_len[0]
        #~ self.line['avg_cell_per_sec'] = int(self.dom_size / avg_comp_time)

        # Add line to file
        with open(self.file_name, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(self.line)

        # empty dictionaries
        self.sim_data = {k: [] for k in self.sim_data.keys()}
        self.line = dict.fromkeys(self.line.keys())
        return self
