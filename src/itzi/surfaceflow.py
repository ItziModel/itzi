# coding=utf8
"""
Copyright (C) 2015-2025 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

import math
from datetime import timedelta
import numpy as np

import itzi.flow as flow
from itzi.itzi_error import NullError, DtError


class SurfaceFlowSimulation:
    """Surface flow simulation on staggered raster grid
    Accessed through step() methods
    By convention the flow is:
     - calculated at the East and South faces of each cell
     - positive from West to East and from North to South
    """

    def __init__(self, domain, param):
        self.dom = domain
        self.dtmax = param["dtmax"]
        self.cfl = param["cfl"]
        self.g = param["g"]
        self.theta = param["theta"]
        self.min_flow_depth = param["hmin"]
        self.sl_thresh = param["slmax"]
        self.v_routing = param["vrouting"]
        self.dx = domain.dx
        self.dy = domain.dy
        self.cell_surf = self.dx * self.dy

        self._dt = None

    def update_flow_dir(self):
        """Return arrays of flow directions used for rain routing
        each cell is assigned a direction in which it will drain
        0: the flow is going dowstream, index-wise
        1: the flow is going upstream, index-wise
        -1: no routing happening on that face
        """
        # get a padded array
        arrp_z = self.dom.get_padded("dem")
        # define differences in Z
        z0 = arrp_z[1:-1, 1:-1]
        zN = arrp_z[0:-2, 1:-1]
        zS = arrp_z[2:, 1:-1]
        zE = arrp_z[1:-1, 2:]
        zW = arrp_z[1:-1, 0:-2]
        dN = z0 - zN
        dE = z0 - zE
        dS = z0 - zS
        dW = z0 - zW
        # maximum altitude difference
        arr_max_dz = np.maximum(np.maximum(dN, dS), np.maximum(dE, dW))
        # y direction
        flow.flow_dir(arr_max_dz, dN, dS, self.dom.get_array("dirs"))
        # x direction
        flow.flow_dir(arr_max_dz, dW, dE, self.dom.get_array("dire"))

        return self

    def step(self):
        """Run a full simulation time-step"""
        self.solve_q()
        self.update_h()
        # in case of NaN/NULL cells, raise a NullError
        self.arr_err = np.isnan(self.dom.get_array("water_depth"))
        if np.any(self.arr_err):
            raise NullError
        self.swap_flow_arrays()
        return self

    def solve_dt(self):
        """Calculate the adaptive time-step
        The formula #15 in Almeida et al (2012) has been modified to
        accommodate non-square cells
        The time-step is limited by the maximum time-step dtmax.
        """
        maxh = np.amax(self.dom.get_array("water_depth"))  # max depth in domain
        min_dim = min(self.dx, self.dy)
        if maxh > 0:
            dt = self.cfl * (min_dim / (math.sqrt(self.g * maxh)))
            self._dt = min(self.dtmax, dt)
        else:
            self._dt = self.dtmax
        return self

    @property
    def dt(self):
        return timedelta(seconds=self._dt)

    @dt.setter
    def dt(self, newdt):
        """return an error if new dt is higher than current one or negative"""
        newdt_s = newdt.total_seconds()
        fudge = timedelta.resolution.total_seconds()
        if self._dt is None:
            self._dt = newdt_s
        elif newdt_s <= 0:
            raise DtError("dt must be positive ({})".format(newdt_s))
        elif newdt_s > self._dt + fudge:
            raise DtError(
                "new dt cannot be longer than current one (old: {}, new: {})".format(
                    self._dt, newdt_s
                )
            )
        else:
            self._dt = newdt_s

    def update_h(self):
        """Calculate new water depth, average velocity and Froude number"""
        flow.solve_h(
            arr_ext=self.dom.get_padded("ext"),
            arr_qe=self.dom.get_padded("qe_new"),
            arr_qs=self.dom.get_padded("qs_new"),
            arr_bct=self.dom.get_padded("bctype"),
            arr_bcv=self.dom.get_padded("bcval"),
            arr_h=self.dom.get_padded("water_depth"),
            arr_hmax=self.dom.get_padded("hmax"),
            arr_hfix=self.dom.get_padded("boundaries_accum"),
            arr_herr=self.dom.get_padded("error_depth_accum"),
            arr_hfe=self.dom.get_padded("hfe"),
            arr_hfs=self.dom.get_padded("hfs"),
            arr_v=self.dom.get_padded("v"),
            arr_vdir=self.dom.get_padded("vdir"),
            arr_vmax=self.dom.get_padded("vmax"),
            arr_fr=self.dom.get_padded("froude"),
            dx=self.dx,
            dy=self.dy,
            dt=self._dt,
            g=self.g,
        )
        assert not np.any(self.dom.get_array("water_depth") < 0)
        return self

    def solve_q(self):
        """Solve flow inside the domain using C/Cython function"""
        flow.solve_q(
            arr_dire=self.dom.get_padded("dire"),
            arr_dirs=self.dom.get_padded("dirs"),
            arr_z=self.dom.get_padded("dem"),
            arr_n=self.dom.get_padded("friction"),
            arr_h=self.dom.get_padded("water_depth"),
            arr_qe=self.dom.get_padded("qe"),
            arr_qs=self.dom.get_padded("qs"),
            arr_hfe=self.dom.get_padded("hfe"),
            arr_hfs=self.dom.get_padded("hfs"),
            arr_bctype=self.dom.get_padded("bctype"),
            arr_bcvalue=self.dom.get_padded("bcval"),
            arr_qe_new=self.dom.get_padded("qe_new"),
            arr_qs_new=self.dom.get_padded("qs_new"),
            arr_bcaccum=self.dom.get_padded("boundaries_accum"),
            dt=self._dt,
            dx=self.dx,
            dy=self.dy,
            g=self.g,
            theta=self.theta,
            hf_min=self.min_flow_depth,
            v_rout=self.v_routing,
            sl_thres=self.sl_thresh,
        )
        return self

    def swap_flow_arrays(self):
        """Swap flow arrays from calculated to input"""
        self.dom.swap_arrays("qe", "qe_new")
        self.dom.swap_arrays("qs", "qs_new")
        return self
