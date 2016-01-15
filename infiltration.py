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

class Infiltration(object):
    """Base class for Infiltration
    infiltration is calculated in mm/h
    """
    def __init__(self, dt):
        self.infrate = None
        self.dt_h = None

    def cap_rate(self, arr_h, dt):
        """Cap the infiltration rate to not create negative depths
        """
        # max rate in mm/h
        self.dt_h = dt / 3600
        arr_h_mm = arr_h * 1000
        arr_max_rate = arr_h_mm / self.dt_h
        # cap the rate
        self.infrate = np.minimum(arr_max_rate, self.infrate)


class InfConstantRate(Infiltration):
    """Calculate infiltration using a constant user-defined infiltration
    rate given by a raster map or serie of maps.
    """
    def __init__(self, arr_inf):
        assert isinstance(arr_inf, np.ndarray), "not a np array!"
        self.infrate = arr_inf

    def get_inf_rate(self, arr_h, dt):
        """Used to get the infiltration rate at each time step
        """
        assert isinstance(arr_h, np.ndarray), "not a np array!"
        assert isinstance(dt, float), "not a float!"
        self.cap_rate(arr_h, dt)
        return self.infrate


class InfGreenAmpt(Infiltration):
    """Calculate infiltration using Green-Ampt formula
    """
    def __init__(self, total_por, eff_por, cap_pressure, hyd_conduct):
        assert isinstance(total_por, np.ndarray), "not a np array!"
        assert isinstance(eff_por, np.ndarray), "not a np array!"
        assert isinstance(cap_pressure, np.ndarray), "not a np array!"
        assert isinstance(hyd_conduct, np.ndarray), "not a np array!"

        self.total_porosity = total_por
        self.eff_porosity = eff_por
        self.capilary_pressure = cap_pressure
        self.hyd_conduct = hyd_conduct
        # starting values (at zero)
        self.init_wat_soil_content = np.zeros_like(total_porosity)
        self.infiltration_amount = np.zeros_like(total_porosity)

    def solve_green_ampt(self):
        avail_porosity = self.eff_porosity - self.init_wat_soil_content
        poros_cappress = avail_porosity * self.capilary_pressure
        self.infrate = self.hyd_conduct * (1 +
                            (poros_cappress / self.infiltration_amount))

    def get_inf_rate(self, arr_h, dt):
        """Used to get the infiltration rate at each time step
        """
        self.solve_green_ampt()
        self.cap_rate(arr_h, dt)
        self.infiltration_amount += self.infrate * self.dt_h
        return self.infrate


#~ class InfHoltan(Infiltration):
    #~ """
    #~ """

    #~ def get_infiltration(self):
        #~ f = fc + a * pow(So - F, n)
