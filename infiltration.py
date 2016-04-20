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
import flow


class Infiltration(object):
    """Base class for Infiltration
    infiltration is calculated in mm/h
    """
    def __init__(self, raster_domain):
        self.dom = raster_domain

    def set_dt(self, dt_inf, sim_clock, next_inf_ts):
        """adjust infiltration time-step to not overstep a forced time-step.
        dt_inf, sim_clock and next_ts in seconds
        """
        self.dt = dt_inf
        if sim_clock + self.dt > next_inf_ts:
            self.dt = next_inf_ts - sim_clock
        return self


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
                      dt=self.dt)
        return self


class InfGreenAmpt(Infiltration):
    """Calculate infiltration using Green-Ampt formula
    """
    def __init__(self, raster_domain):
        Infiltration.__init__(self, raster_domain)
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
                    arr_inf_out=self.dom.get('inf'), dt=self.dt)
        return self


class InfNull(Infiltration):
    """Dummy class for cases where no inflitration is calculated
    """
    def step(self):
        pass
        return self

#~ class InfHoltan(Infiltration):
    #~ """
    #~ """

    #~ def get_infiltration(self):
        #~ f = fc + a * pow(So - F, n)
