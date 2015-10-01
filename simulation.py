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
from datetime import datetime
import domain
import gis

class SimulationManager(object):
    """
    """

    def __init__(self, start_time=datetime(1,1,1), end_time=0,
                    sim_duration=None):
        self.gis = gis.Igis()
        assert isinstance(start_time, datetime), \
            "start_time should be a datetime object"
        self.start_time
        try:
            int(sim_duration)
        except ValueError:
            assert False, "sim_duration should be integer"
        self.set_sim_duration(end_time, sim_duration)
        self.arrays = {'z': None, 'n': None, 'ext': None,
                    'h_old': None, 'h_new': None,
                    'hfw': None, 'hfn': None,
                    'qw_old': None, 'qn_old': None,
                    'qw_new': None, 'qn_new': None}

    def set_duration(self, end_time, sim_duration):
        """
        """
        if end_time:
            self.duration = (sim_end - self.sim_start).total_seconds()
        return self

    def run_superficial_flow(self):
        """Perform a full superficial flow simulation
        including recording of data and mass_balance calculation
        """
        rast_dom = domain.SurfaceDomain(dx=None, dy=None)

        while rast_dom.sim_clock <= self.duration:
            rast_dom.set_input_arrays(self.arrays)
            rast_dom.step(self.next_timestep())
        return self
    
    def next_timestep(self):
        """
        """
        return next_ts

    def set_arrays(self):
        """
        """
        self.arrays
