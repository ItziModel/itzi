# coding=utf8
"""
Copyright (C) 2016-2025  Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

from datetime import timedelta
import numpy as np

import itzi.flow as flow
from itzi.itzi_error import DtError


class Infiltration:
    """Base class for Infiltration
    infiltration is calculated in m/s
    """

    def __init__(self, raster_domain, dt_inf):
        self.dom = raster_domain
        self.def_dt = dt_inf
        self._dt = self.def_dt

    def solve_dt(self):
        """time-step is by default equal to the default time-step"""
        self._dt = self.def_dt
        return self

    @property
    def dt(self):
        """return the time-step as a timedelta"""
        return timedelta(seconds=self._dt)

    @dt.setter
    def dt(self, newdt):
        """return an error if new dt is higher than current one"""
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
        """Update infiltration rate map in m/s"""
        flow.infiltration_user(
            arr_h=self.dom.get_array("water_depth"),
            arr_inf_in=self.dom.get_array("infiltration"),
            arr_inf_out=self.dom.get_array("computed_infiltration"),
            dt=self._dt,
        )
        return self


class InfGreenAmpt(Infiltration):
    """Calculate infiltration using Green-Ampt formula"""

    def __init__(self, raster_domain, dt_inf):
        Infiltration.__init__(self, raster_domain, dt_inf)
        # Initial cumulative infiltration set to tiny value
        # (prevent division by zero)
        self.infiltration_amount = np.full(
            shape=self.dom.shape, fill_value=(1e-4), dtype=self.dom.dtype
        )

    def step(self):
        """update infiltration rate map in m/s."""
        flow.infiltration_ga(
            arr_h=self.dom.get_array("water_depth"),
            arr_eff_por=self.dom.get_array("effective_porosity"),
            arr_pressure=self.dom.get_array("capillary_pressure"),
            arr_conduct=self.dom.get_array("hydraulic_conductivity"),
            arr_inf_amount=self.infiltration_amount,
            arr_water_soil_content=self.dom.get_array("soil_water_content"),
            arr_inf_out=self.dom.get_array("computed_infiltration"),
            dt=self._dt,
        )
        return self


class InfNull(Infiltration):
    """Dummy class for cases where no infiltration is calculated"""

    def step(self):
        """dummy time-step"""
        return self
