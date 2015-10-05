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
        self.set_temporal_type()
        self.set_duration(end_time, sim_duration)
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
        # this one is set a default to ones
        self.tarrays['in_bctype'] = TimedArray('in_bctype', self.gis, self.ones_array)
        return self

    def run(self):
        """Perform a full superficial flow simulation
        including recording of data and mass_balance calculation
        """
        rast_dom = domain.SurfaceDomain(
                dx=self.gis.dx,
                dy=self.gis.dy,
                arr_h=self.tarrays['in_h'].get_array(self.start_time),
                arr_def=self.zeros_array())
        record_counter = 0
        duration_s = self.duration.total_seconds()

        while rast_dom.sim_clock < duration_s:
            # display advance of simulation
            self.gis.msgr.percent(rast_dom.sim_clock, duration_s, 1)
            # update arrays
            self.update_domain_arrays(rast_dom)
            # time-stepping
            next_record = record_counter*self.record_step.total_seconds()
            rast_dom.step(self.next_timestep(next_record))
            # write simulation results
            rec_time = rast_dom.sim_clock / self.record_step.total_seconds()
            if rec_time >= record_counter:
                self.output_arrays = rast_dom.get_output_arrays(self.out_map_names)
                self.write_results_to_gis(record_counter)
                record_counter += 1
        # register generated maps in GIS
        self.register_results_in_gis()
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
        """Combine rain, infiltration etc. into a unique array
        rainfall and infiltration are considered in mm/h
        """
        return in_q + (in_rain + in_inf) / 1000 / 3600

    def write_results_to_gis(self, record_counter):
        """
        """
        for k,arr in self.output_arrays.iteritems():
            if arr != None:
                assert isinstance(arr, np.ndarray), "arr not a np array!"
                suffix = timestamp = str(record_counter).zfill(6)
                map_name = "{}_{}".format(self.out_map_names[k], suffix)
                #~ print map_name
                self.gis.write_raster_map(arr, map_name)
        return self

    def register_results_in_gis(self):
        """
        """
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
        # update object values
        self.a_start = arr_start
        self.a_end = arr_end
        self.arr = arr
        return self
