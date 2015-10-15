#!/usr/bin/env python
# coding=utf8
"""
Copyright (C) 2015  Laurent Courty

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
import numpy as np
from datetime import datetime, timedelta
import domain
import gis

class SuperficialFlowSimulation(object):
    """
    """

    def __init__(self,record_step, input_maps, output_maps,
                dtype=np.float32,
                start_time=datetime(1,1,1),
                end_time=datetime(1,1,1),
                sim_duration=timedelta(0)):
        assert isinstance(start_time, datetime), \
            "start_time not a datetime object!"
        #~ assert isinstance(end_time, datetime), \
            #~ "end_time not a datetime object!"
        #~ assert start_time <= end_time, "start_time > end_time!"
        assert isinstance(sim_duration, timedelta), \
            "sim_duration not a timedelta object!"
        assert sim_duration >= timedelta(0), "sim_duration is negative!"
        assert isinstance(record_step, timedelta), \
            "record_step not a timedelta object!"
        assert record_step > timedelta(0), "record_step must be > 0"

        self.record_step = record_step
        self.start_time = start_time
        self.set_duration(end_time, sim_duration)
        # set simulation time to start_time a the beginning
        self.sim_time = self.start_time
        self.dt = 0
        # set temporal type of results
        self.set_temporal_type()
        self.in_map_names = input_maps
        self.out_map_names = output_maps
        self.dtype=dtype
        # instantiate a Igis object
        self.gis = gis.Igis(start_time=self.start_time,
                            end_time=self.end_time,
                            dtype=dtype, mkeys=self.in_map_names.keys())
        self.gis.msgr.verbose(_("Reading GIS..."))
        self.gis.read(self.in_map_names)
        # dict to store TimedArrays objects
        self.tarrays = dict.fromkeys(self.in_map_names.keys())
        # Populate it
        self.create_timed_arrays()
        # a dict containing lists of maps written to gis to be registered
        self.output_maplist = {k:[] for k in self.out_map_names.keys()}
        # Instantiate Massbal object
        self.massbal = MassBal()

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

    def create_timed_arrays(self):
        """Create a set of TimedArray objects
        Store the created objects in the tarrays dict
        """
        self.tarrays['in_z'] = TimedArray('in_z', self.gis, self.zeros_array)
        self.tarrays['in_n'] = TimedArray('in_n', self.gis, self.zeros_array)
        self.tarrays['in_h'] = TimedArray('in_h', self.gis, self.zeros_array)
        self.tarrays['in_inf'] = TimedArray('in_inf', self.gis, self.zeros_array)
        self.tarrays['in_rain'] = TimedArray('in_rain', self.gis, self.zeros_array)
        self.tarrays['in_q'] = TimedArray('in_q', self.gis, self.zeros_array)
        self.tarrays['in_bcval'] = TimedArray('in_bcval', self.gis, self.zeros_array)
        # Default of bctypes is an int array filled with ones
        self.tarrays['in_bctype'] = TimedArray('in_bctype', self.gis, self.ones_array)
        return self

    def run(self):
        """Perform a full superficial flow simulation
        including recording of data and mass_balance calculation
        """
        # Instantiate SurfaceDomain object
        rast_dom = domain.SurfaceDomain(
                dx=self.gis.dx,
                dy=self.gis.dy,
                arr_h=self.tarrays['in_h'].get_array(self.start_time),
                arr_def=self.zeros_array())
        record_counter = 1
        duration_s = self.duration.total_seconds()

        while self.sim_time < self.end_time:
            # display advance of simulation
            self.gis.msgr.percent(rast_dom.sim_clock, duration_s, 1)
            # update arrays
            self.update_domain_arrays(rast_dom)
            # time-stepping
            next_record = record_counter*self.record_step.total_seconds()
            rast_dom.step(self.next_timestep(next_record), self.massbal)
            # update simulation time and dt
            self.sim_time = self.start_time + timedelta(seconds=rast_dom.sim_clock)
            self.dt = rast_dom.dt
            self.massbal.add_value('tstep', self.dt)
            # write simulation results
            rec_time = rast_dom.sim_clock / self.record_step.total_seconds()
            if rec_time >= record_counter:
                self.output_arrays = rast_dom.get_output_arrays(self.out_map_names)
                self.write_results_to_gis(record_counter)
                record_counter += 1
                self.write_mass_balance(rast_dom.sim_clock)
        # register generated maps in GIS
        self.register_results_in_gis()
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

    def zeros_array(self):
        """return a np array of the domain dimension, filled with ones
        dtype is set to object's dtype.
        Intended to be used as default for most of the input model maps
        """
        return np.zeros(shape=(self.gis.ry, self.gis.xr), dtype=self.dtype)

    def ones_array(self):
        """return a np array of the domain dimension, filled with ones
        dtype is set to unsigned integer.
        Intended to be used as default for bctype map
        """
        return np.ones(shape=(self.gis.ry, self.gis.xr), dtype=np.uint8)

    def next_timestep(self, next_record):
        """Given a next record time in seconds as entry,
        return the future time at which the model will be forced to step.
        """
        return min(next_record, self.duration.total_seconds())

    def update_domain_arrays(self, rast_dom):
        """Takes a SurfaceDomain object as input
        set the input arrays of the given object using TimedArray
        auto-update capacity
        """
        assert isinstance(rast_dom, domain.SurfaceDomain), \
            "rast_dom not the expected object!"

        sim_time = self.start_time + timedelta(seconds=rast_dom.sim_clock)

        rast_dom.arr_z = self.tarrays['in_z'].get_array(sim_time)
        rast_dom.arr_n = self.tarrays['in_n'].get_array(sim_time)
        rast_dom.arr_bcval = self.tarrays['in_bcval'].get_array(sim_time)
        rast_dom.arr_bctype = self.tarrays['in_bctype'].get_array(sim_time)
        # Combine three arrays for the ext array
        rast_dom.arr_ext = self.set_ext_array(
            in_q=self.tarrays['in_q'].get_array(sim_time),
            in_rain=self.tarrays['in_rain'].get_array(sim_time),
            in_inf=self.tarrays['in_inf'].get_array(sim_time))
        return self

    def set_ext_array(self, in_q, in_rain, in_inf):
        """Combine rain, infiltration etc. into a unique array in m/s
        rainfall and infiltration are considered in mm/h
        send relevant values to MassBal
        """
        assert isinstance(in_q, np.ndarray), "not a np array!"
        assert isinstance(in_rain, np.ndarray), "not a np array!"
        assert isinstance(in_inf, np.ndarray), "not a np array!"

        # mass balance in m3
        cell_surf = self.gis.dx * self.gis.dy
        rain_vol = np.sum(in_rain / 1000. / 3600.) * cell_surf * self.dt
        inf_vol = np.sum(in_inf / 1000. / 3600.) * cell_surf * self.dt
        inflow_vol = np.sum(in_q) * cell_surf * self.dt
        self.massbal.add_value('rain_vol', rain_vol)
        self.massbal.add_value('inf_vol', inf_vol)
        self.massbal.add_value('inflow_vol', inflow_vol)

        return in_q + (in_rain - in_inf) / 1000. / 3600.

    def write_results_to_gis(self, record_counter):
        """Format the name of each maps using the record number as suffix
        Send a couple array, name to the GIS writing function.
        """
        for k,arr in self.output_arrays.iteritems():
            if arr != None:
                assert isinstance(arr, np.ndarray), "arr not a np array!"
                suffix = str(record_counter).zfill(6)
                map_name = "{}_{}".format(self.out_map_names[k], suffix)
                self.gis.write_raster_map(arr, map_name,
                                    self.sim_time, self.temporal_type)
                # add map name to the revelant list
                self.output_maplist[k].append(map_name)
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
            if strds_name == None:
                continue
            self.gis.register_maps_in_strds(mkey, strds_name, lst, self.temporal_type)
        return self


class TimedArray(object):
    """A container for np.ndarray along with time informations
    Update the array value according to the simulation time
    array is accessed via get_array()
    """
    def __init__(self, mkey, igis, f_arr_def):
        assert isinstance(mkey, basestring), "not a string!"
        assert hasattr(f_arr_def, '__call__'), "not a function!"
        self.mkey = mkey
        self.igis = igis  # GIS interface
        # A function to generate a default array
        self.f_arr_def = f_arr_def
        # default values for start and end
        # intended to trigger update when is_valid() is first called
        self.a_start = datetime(1,1,2)
        self.a_end = datetime(1,1,1)

    def get_array(self, sim_time):
        """Return a numpy array valid for the given time
        If the array stored is not valid, update the values of the object
        """
        assert isinstance(sim_time, datetime), "not a datetime object!"
        if not self.is_valid(sim_time):
            self.update_values_from_gis(sim_time)
        return self.arr

    def is_valid(self, sim_time):
        """input being a time in timedelta
        If the current stored array is within the range of the map,
        return True
        If not return False
        """
        if self.a_start <= sim_time <= self.a_end:
            return True
        else:
            return False

    def update_values_from_gis(self, sim_time):
        """Update array, start_time and end_time from GIS
        if GIS return None, set array to default value
        """
        # Retrieve values
        arr, arr_start, arr_end = self.igis.get_array(self.mkey, sim_time)
        # set to default if necessary
        if arr == None:
            arr = self.f_arr_def()
        # check retrieved values
        assert isinstance(arr, np.ndarray), "not a np.ndarray!"
        assert isinstance(arr_start, datetime), "not a datetime object!"
        assert isinstance(arr_end, datetime), "not a datetime object!"
        assert arr_start <= sim_time <= arr_end, "wrong time retrieved!"
        # update object values
        self.a_start = arr_start
        self.a_end = arr_end
        self.arr = arr
        return self


class MassBal(object):
    """Follow-up the mass balance during the simulation run
    Mass balance error is the difference between the actual volume and
    the theoretical volume. The later is the old volume + input - output.
    Intended use:
    at each time-step, using add_value():
    individual simulation operations send relevant values to the MassBal object
    at each record time, using write_values():
    averaged or cumulated values for the considered time difference are written to a CSV file
    """
    def __init__(self, file_name=''):
        self.name = file_name
        # values to be written on each record time
        self.fields = ['sim_time',  # either seconds or datetime
                'avg_timestep', 'min_timestep', '#timesteps',
                'boundary_vol', 'rain_vol', 'inf_vol', 'inflow_vol',
                'domain_vol', 'vol_error']
        # data written to file as one line
        self.line = dict.fromkeys(self.fields)
        # data collected during simulation
        self.sim_data = {'tstep': [], 'boundary_vol': [],
                        'rain_vol': [], 'inf_vol': [], 'inflow_vol': [],
                        'old_dom_vol': [], 'new_dom_vol': []}
        # set file name and create file
        self.file_name = self.set_file_name(file_name)
        self.create_file()

    def set_file_name(self, file_name):
        '''Generate output file name
        '''
        if not file_name:
            file_name = "{}_mass_balance.csv".format(
                str(datetime.now().strftime('%Y-%M-%dT%H:%M:%S')))
        return file_name

    def create_file(self):
        '''Create a csv file and write headers
        '''
        with open(self.file_name, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()
        return self

    def add_value(self, key, value):
        '''add a value to sim_data
        '''
        assert key in self.sim_data, "unknown key!"
        self.sim_data[key].append(value)
        return self

    def write_values(self, sim_time):
        '''prepare data line and write it to file
        '''
        # check if all elements have the same number of records
        rec_len = [len(l) for l in self.sim_data.values()]
        assert rec_len[1:] == rec_len[:-1], "inconsistent number of records!"

        self.line['sim_time'] = sim_time
        # number of time-step during the interval is the number of records
        self.line['#timesteps'] = len(self.sim_data['tstep'])
        self.line['min_timestep'] = min(self.sim_data['tstep'])
        # average time-step calculation
        elapsed_time = sum(self.sim_data['tstep'])
        self.line['avg_timestep'] = elapsed_time / self.line['#timesteps']
        # sum of inflow (positive) / outflow (negative) volumes
        self.line['boundary_vol'] = sum(self.sim_data['boundary_vol'])
        self.line['rain_vol'] = sum(self.sim_data['rain_vol'])
        self.line['inf_vol'] = sum(self.sim_data['inf_vol'])
        self.line['inflow_vol'] = sum(self.sim_data['inflow_vol'])
        # For domain volume, take last value(i.e. current)
        last_vol = self.sim_data['new_dom_vol'][-1]
        self.line['domain_vol'] = last_vol
        # mass error is the diff. between the theor. vol and the actual vol
        first_vol = self.sim_data['old_dom_vol'][0]
        sum_ext_vol = sum([self.line['boundary_vol'],
                                self.line['rain_vol'],
                                - self.line['inf_vol'],
                                self.line['inflow_vol']])
        dom_vol_theor = first_vol + sum_ext_vol
        self.line['vol_error'] = last_vol - dom_vol_theor

        # Add line to file
        with open(self.file_name, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(self.line)

        # empty dictionaries
        self.sim_data = {k:[] for k in self.sim_data.keys()}
        self.line = dict.fromkeys(self.line.keys())
        return self
