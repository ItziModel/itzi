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

import itzi.flow as flow
from itzi.itzi_error import DtError


class Hydrology:
    """ """

    def __init__(self, raster_domain, dt, infiltration):
        self.dom = raster_domain
        self.def_dt = dt
        # an infiltration model object
        self.infiltration = infiltration

    def solve_dt(self):
        """time-step is by default equal to the default time-step"""
        self._dt = self.def_dt
        self.infiltration.solve_dt()
        return self

    @property
    def dt(self):
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
            self.infiltration.dt = newdt_s

    def step(self):
        """Run hydrologic models and update the water depth map"""
        # calculate flows
        self.infiltration.step()
        self.cap_losses()
        self.apply_hydrology()
        return self

    def cap_losses(self):
        """User-defined losses are treated like user infiltration."""
        flow.infiltration_user(
            arr_h=self.dom.get_array("water_depth"),
            arr_inf_in=self.dom.get_array("losses"),
            arr_inf_out=self.dom.get_array("capped_losses"),
            dt=self._dt,
        )

    def apply_hydrology(self):
        """Update effective precipitation (eff_precip) by adding/removing depth from:
        rainfall, infiltration, evapotranspiration and lump-sum drainage.
        """
        flow.apply_hydrology(
            arr_rain=self.dom.get_array("rain"),
            arr_inf=self.dom.get_array("computed_infiltration"),
            arr_capped_losses=self.dom.get_array("capped_losses"),
            arr_h=self.dom.get_array("water_depth"),
            arr_eff_precip=self.dom.get_array("eff_precip"),
            dt=self._dt,
        )
        return self
