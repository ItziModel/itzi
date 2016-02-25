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
    def __init__(self):
        pass

    def set_dt(self, dt_inf, sim_clock, next_inf_ts):
        """adjust infiltration time-step to not overstep a forced time-step
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
    def __init__(self):
        pass

    def update_input(self, arr_inf):
        assert isinstance(arr_inf, np.ndarray), "not a np array!"
        self.inf_in = arr_inf
        self.inf_out = np.copy(arr_inf)

    def get_inf_rate(self, arr_h):
        """Used to get the infiltration rate at each time step
        """
        assert isinstance(arr_h, np.ndarray), "not a np array!"
        flow.inf_user(arr_h, self.inf_in, self.inf_out, self.dt)
        return self.inf_out


class InfGreenAmpt(Infiltration):
    """Calculate infiltration using Green-Ampt formula
    """
    def __init__(self, xr, yr):
        self.eff_porosity = None
        self.capilary_pressure = None
        self.hyd_conduct = None
        self.infrate = np.zeros(shape=(yr, xr), dtype=np.float32)
        # Initial water soil content set to zero
        self.init_wat_soil_content = np.zeros(shape=(yr, xr), dtype=np.float32)
        # Initial cumulative infiltration set to one mm (prevent division by zero)
        self.infiltration_amount = np.ones(shape=(yr, xr), dtype=np.float32)
        assert self.infrate.shape == self.init_wat_soil_content.shape == self.infiltration_amount.shape

    def update_input(self, eff_por, cap_pressure, hyd_conduct):
        assert isinstance(eff_por, np.ndarray), "not a np array!"
        assert isinstance(cap_pressure, np.ndarray), "not a np array!"
        assert isinstance(hyd_conduct, np.ndarray), "not a np array!"
        assert eff_por.shape == cap_pressure.shape == hyd_conduct.shape, "inconsistant shape!"
        self.eff_porosity = eff_por
        self.capilary_pressure = cap_pressure
        self.hyd_conduct = hyd_conduct

    def get_inf_rate(self, arr_h):
        """Used to get the infiltration rate at each time step
        """
        assert self.eff_porosity.shape == arr_h.shape
        assert self.eff_porosity.shape == self.infiltration_amount.shape
        assert self.eff_porosity.shape == self.init_wat_soil_content.shape

        flow.inf_ga(arr_h, self.eff_porosity, self.capilary_pressure,
        self.hyd_conduct, self.infiltration_amount, self.init_wat_soil_content,
        self.infrate, self.dt)
        return self.infrate


#~ class InfHoltan(Infiltration):
    #~ """
    #~ """

    #~ def get_infiltration(self):
        #~ f = fc + a * pow(So - F, n)
