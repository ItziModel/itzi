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

    def __init__(self,
                start_time=datetime(1,1,1), end_time=datetime(1,1,1),
                sim_duration=timedelta(0), record_step, dtype=np.float32,
                input_maps, output_maps):
        assert isinstance(start_time, datetime), \
            "start_time not a datetime object!"
        assert isinstance(end_time, datetime), \
            "end_time not a datetime object!"
        assert start_time <= end_time, "start_time > end_time!"
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
        # dict of arrays accepted by SurfaceDomain
        self.dom_arrays = {'z': None, 'n': None, 'ext': None,
                    'h_old': None, 'bcval': None, 'bctype': None}
        self.in_map_names = input_maps
        self.out_map_names = output_maps
        self.dtype=dtype
        # instantiate a Igis object
        self.gis = gis.Igis(start_time=self.start_time,
                            end_time=self.end_time,
                            dtype=dtype)

    def set_duration(self, end_time, sim_duration):
        """If sim_duration is given, end_time is ignored
        This is normally checked upstream
        """
        if not sim_duration:
            self.duration = end_time - self.sim_start
            self.end_time = end_time
        else:
            self.duration = sim_duration
            self.end_time = start_time + sim_duration
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

    def run(self):
        """Perform a full superficial flow simulation
        including recording of data and mass_balance calculation
        """
        rast_dom = domain.SurfaceDomain(dx=self.gis.dx,
                                        dy=self.gis.dy
                                        arr_def=self.zeros_array())
        record_counter = 0
        duration_s = self.duration.total_seconds()

        while rast_dom.sim_clock <= duration_s:
            # display advance of simulation
            gis.msgr.percent(rast_dom.sim_clock, duration_s, 1)
            # update arrays
            self.set_arrays(timedelta(seconds=rast_dom.sim_clock))
            # time-stepping
            rast_dom.set_input_arrays(self.dom_arrays)
            rast_dom.step(self.next_timestep())
            # write simulation results
            rec_time = rast_dom.sim_clock / self.record_step.total_seconds()
            if rec_time >= record_counter:
                self.write_results_to_gis()
                record_counter += 1
        # register generated maps in GIS
        self.register_results_in_gis()
        return self

    def zeros_array(self):
        """
        """
        return np.zeros(shape=(self.gis.ry, self.gis.xr), dtype=self.dtype)

    def next_timestep(self):
        """
        """
        return next_ts

    def set_arrays(self, td_clock):
        """load arrays from the GIS as a dict
        translate the loaded dict into one ready to be feeded to the
        domain for next time-step
        """
        assert isinstance(td_clock, timedelta), \
            "td_clock not a timedelta object!"
        assert td_clock >= timedelta(0), "td_clock is negative!"

        sim_time = self.start_time + td_clock
        lst_key = self.in_map_names.keys()
        arrays = self.gis.get_input_arrays(lst_key, sim_time)
        # dictionary translation
        self.dom_arrays['ext'] = set_ext_array(arrays['in_q'],
                                            arrays['in_rain'],
                                            arrays['in_inf'])
        self.dom_arrays['z'] = arrays['in_z']
        self.dom_arrays['n'] = arrays['in_n']
        self.dom_arrays['bcval'] = arrays['in_bcval']
        self.dom_arrays['bctype'] = arrays['in_bctype'].astype(np.uint8)
        # update in_h only at the first time-step
        if not td_clock:
            self.dom_arrays['h_old'] = arrays['in_h']
        return self

    def set_ext_array(self, q, rain, inf):
        """Combine rain, infiltration etc. into a unique array
        rainfall and infiltration are considered in mm/h
        """
        ext = q + (rain + inf) / 1000 / 3600
        return ext

    def write_results_to_gis(self):
        """
        """
        return self

    def register_results_in_gis(self):
        """
        """
        return self
