# coding=utf8
"""
Copyright (C) 2016  Laurent Courty

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
from datetime import timedelta

import flow
import infiltration
from itzi_error import DtError


class Hydrology(object):
    """
    """
    def __init__(self, raster_domain, dt, infiltration):
        self.dom = raster_domain
        self.def_dt = dt
        # an infiltration model object
        self.infiltration = infiltration

    def solve_dt(self):
        """time-step is by default equal to the default time-step
        """
        self._dt = self.def_dt
        self.infiltration.solve_dt()
        return self

    @property
    def dt(self):
        return timedelta(seconds=self._dt)

    @dt.setter
    def dt(self, newdt):
        """return an error if new dt is higher than current one
        """
        newdt_s = newdt.total_seconds()
        fudge = timedelta.resolution.total_seconds()
        if newdt_s > self._dt + fudge:
            raise DtError("new dt cannot be longer than current one")
        else:
            self._dt = newdt_s
            self.infiltration.dt = newdt_s

    def step(self):
        """Run hydrologic models and update the water depth map
        """
        # calculate flows
        self.infiltration.step()
        self.apply_hydrology()
        return self

    def apply_hydrology(self):
        """
        """
        flow.apply_hydrology(arr_rain=self.dom.get('rain'),
                             arr_inf=self.dom.get('inf'),
                             arr_etp=self.dom.get('etp'),
                             arr_drain_cap=self.dom.get('s_drain'),
                             arr_h=self.dom.get('h'),
                             dt=self._dt)
        return self

    #~ def calculate_simple_drainage(self):
        #~ """This is used as an approximation of the drainage capacity of the area
        #~ It is done by removing a user-defined amount of water from the domain.
        #~ """
        #~ flow.inf_user(arr_h=self.dom.get('h'),
              #~ arr_inf_in=self.dom.get('s_drain'),
              #~ arr_inf_out=self.dom.get('inf'),
              #~ dt=self._dt)
