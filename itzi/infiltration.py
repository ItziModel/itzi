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
from itzi_error import DtError


class Infiltration(object):
    """Base class for Infiltration
    infiltration is calculated in mm/h
    """
    def __init__(self, raster_domain, dt_inf):
        self.dom = raster_domain
        self.def_dt = dt_inf

    def solve_dt(self):
        """time-step is by default equal to the default time-step
        """
        self._dt = self.def_dt
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


class InfConstantRate(Infiltration):
    """Calculate infiltration using a constant user-defined infiltration
    rate given by a raster map or serie of maps.
    """
    def step(self):
        """Update infiltration rate map in mm/h
        """
        flow.inf_user(arr_h=self.dom.get('h'),
                      arr_inf_in=self.dom.get('in_inf'),
                      arr_inf_out=self.dom.get('inf'),
                      dt=self._dt)
        return self


class InfGreenAmpt(Infiltration):
    """Calculate infiltration using Green-Ampt formula
    """
    def __init__(self, raster_domain, dt_inf):
        Infiltration.__init__(self, raster_domain, dt_inf)
        # Initial water soil content set to zero
        self.init_wat_soil_content = np.zeros(shape=self.dom.shape,
                                              dtype=self.dom.dtype)
        # Initial cumulative infiltration set to one mm
        # (prevent division by zero)
        self.infiltration_amount = np.ones(shape=self.dom.shape,
                                           dtype=self.dom.dtype)

    def step(self):
        """update infiltration rate map in mm/h.
        """
        flow.inf_ga(arr_h=self.dom.get('h'),
                    arr_eff_por=self.dom.get('por'),
                    arr_pressure=self.dom.get('pres'),
                    arr_conduct=self.dom.get('con'),
                    arr_inf_amount=self.infiltration_amount,
                    arr_water_soil_content=self.init_wat_soil_content,
                    arr_inf_out=self.dom.get('inf'), dt=self._dt)
        return self


class InfNull(Infiltration):
    """Dummy class for cases where no inflitration is calculated
    """
    def step(self):
        pass
        return self
