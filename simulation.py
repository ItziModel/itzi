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

    def __init__(self, start_time=datetime(1,1,1), end_time=datetime(1,1,1),
                    sim_duration=timedelta(0), record_step):
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
        self.set_duration(end_time, sim_duration)
        self.arrays = {'z': None, 'n': None, 'ext': None,
                    'h_old': None, 'h_new': None,
                    'bcval': None, 'bctype': None,
                    'hfw': None, 'hfn': None,
                    'qw_old': None, 'qn_old': None,
                    'qw_new': None, 'qn_new': None}
        self.gis = gis.Igis(start_time=self.start_time,
                            end_time=self.end_time)

    def set_duration(self, end_time, sim_duration):
        """If sim_duration is not zero, end_time is ignored
        """
        if not sim_duration:
            self.duration = end_time - self.sim_start
            self.end_time = end_time
        else:
            self.duration = sim_duration
            self.end_time = start_time + sim_duration
        return self

    def run(self):
        """Perform a full superficial flow simulation
        including recording of data and mass_balance calculation
        """
        rast_dom = domain.SurfaceDomain(dx=self.gis.dx, dy=self.gis.dy)
        record_counter = 0
        while rast_dom.sim_clock <= self.duration.total_seconds():
            self.set_arrays(timedelta(seconds=rast_dom.sim_clock))
            # write simulation records
            rec_time = rast_dom.sim_clock / self.record_step.total_seconds()
            if rec_time >= record_counter:
                self.write_results()
                record_counter += 1
            # time-stepping
            rast_dom.set_input_arrays(self.arrays)
            rast_dom.step(self.next_timestep())
        return self

    def next_timestep(self):
        """
        """
        return next_ts

    def set_arrays(self, timedelta_clock):
        """
        set objects arrays ready for next time-step
        """
        assert isinstance(timedelta_clock, timedelta), \
            "timedelta_clock not a timedelta object!"
        assert timedelta_clock >= timedelta(0), "timedelta_clock is negative!"

        if timedelta_clock == self.start_time:
            self.arrays = self.load_starting_arrays()
        elif self.start_time < timedelta_clock <= self.end_time:
            self.arrays = self.load_running_arrays(timedelta_clock)
            self.copy_arrays_values_for_next_timestep()
        else:
            assert False, "The simulation should be over!"
        return self

    def load_starting_arrays(self):
        """Get a dict of arrays from the GIS
        """
        lst_key = ['z', 'n', 'rain', 'inf', 'bcval', 'bctype', 'h_old']
        return self.gis.get_input_arrays(lst_key, self.start_time)

    def load_running_arrays(self):
        """Get a dict of arrays from the GIS
        """
        current_sim_time = self.start_time + timedelta_clock
        lst_key = ['z', 'n', 'rain', 'inf', 'bcval', 'bctype']
        return self.gis.get_input_arrays(lst_key, current_sim_time)

    def copy_arrays_values_for_next_timestep(self):
        """Copy values from calculated arrays to input arrays
        """
        self.arrays['qw_old'][:] = self.arrays['qw_new']
        self.arrays['qn_old'][:] = self.arrays['qn_new']
        self.arrays['h_old'][:] = self.arrays['h_new']
        return self

    def write_results(self):
        """
        """
        return self
