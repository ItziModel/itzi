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

        # Slices for upstream and downstream cells on a padded array
        self.su = slice(None, -2)
        self.sd = slice(2, None)
        # uniform crop, i.e. equivalent to non-padded array
        self.ss = slice(1, -1)
        # slice to crop last row or column of non-padded array
        # to not conflict with boundary condition
        self.sc = slice(None, -1)
        # equivalent in 2D
        self.s_i_0 = (slice(None), self.sc)
        self.s_j_0 = (self.sc, slice(None))
        # 'Downstream' slice: cells to the east or south
        self.s_i_1 = (slice(None), slice(1, None))
        self.s_j_1 = (slice(1, None), slice(None))

        # Instantiate boundary objects
        self.w_boundary = Boundary(self.dy, self.dx, boundary_pos="W")
        self.e_boundary = Boundary(self.dy, self.dx, boundary_pos="E")
        self.n_boundary = Boundary(self.dx, self.dy, boundary_pos="N")
        self.s_boundary = Boundary(self.dx, self.dy, boundary_pos="S")

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
        self.apply_boundary_conditions()
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
        flow_west = self.dom.get_padded("qe_new")[self.ss, self.su]
        flow_east = self.dom.get_array("qe_new")
        flow_north = self.dom.get_padded("qs_new")[self.su, self.ss]
        flow_south = self.dom.get_array("qs_new")
        assert flow_west.shape == flow_east.shape == flow_north.shape == flow_south.shape

        flow_depth_west = self.dom.get_padded("hfe")[self.ss, self.su]
        flow_depth_east = self.dom.get_array("hfe")
        flow_depth_north = self.dom.get_padded("hfs")[self.su, self.ss]
        flow_depth_south = self.dom.get_array("hfs")
        assert (
            flow_depth_west.shape
            == flow_depth_east.shape
            == flow_depth_north.shape
            == flow_depth_south.shape
        )

        flow.solve_h(
            arr_ext=self.dom.get_array("ext"),
            arr_qe=flow_east,
            arr_qw=flow_west,
            arr_qn=flow_north,
            arr_qs=flow_south,
            arr_bct=self.dom.get_array("bctype"),
            arr_bcv=self.dom.get_array("bcval"),
            arr_h=self.dom.get_array("water_depth"),
            arr_hmax=self.dom.get_array("hmax"),
            arr_hfix=self.dom.get_array("boundaries_accum"),
            arr_herr=self.dom.get_array("error_depth_accum"),
            arr_hfe=flow_depth_east,
            arr_hfw=flow_depth_west,
            arr_hfn=flow_depth_north,
            arr_hfs=flow_depth_south,
            arr_v=self.dom.get_array("v"),
            arr_vdir=self.dom.get_array("vdir"),
            arr_vmax=self.dom.get_array("vmax"),
            arr_fr=self.dom.get_array("froude"),
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
            arr_dire=self.dom.get_array("dire"),
            arr_dirs=self.dom.get_array("dirs"),
            arr_z=self.dom.get_array("dem"),
            arr_n=self.dom.get_array("friction"),
            arr_h=self.dom.get_array("water_depth"),
            arrp_qe=self.dom.get_padded("qe"),
            arrp_qs=self.dom.get_padded("qs"),
            arr_hfe=self.dom.get_array("hfe"),
            arr_hfs=self.dom.get_array("hfs"),
            arr_qe_new=self.dom.get_array("qe_new"),
            arr_qs_new=self.dom.get_array("qs_new"),
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

    def apply_boundary_conditions(self):
        """Select relevant 1D slices and apply boundary conditions.
        1D arrays passed to the boundary method include cells bordering
        the boundary on the inside of the domain.
        For the values applying at cells interface (flow depth and flow):
        'qboundary' is the flow at the very boundary and is updated
        'flow_depth' and 'qin' are the next value inside the domain
        Therefore, only 'qboundary' should need a padded array.
        """

        w_boundary_flow = self.dom.get_padded("qe_new")[1:-1, 0]
        self.w_boundary.get_boundary_flow(
            qin=self.dom.get_array("qe_new")[:, 0],
            flow_depth=self.dom.get_array("hfe")[:, 0],
            n=self.dom.get_array("friction")[:, 0],
            z=self.dom.get_array("dem")[:, 0],
            depth=self.dom.get_array("water_depth")[:, 0],
            bctype=self.dom.get_array("bctype")[:, 0],
            bcvalue=self.dom.get_array("bcval")[:, 0],
            qboundary=w_boundary_flow,
        )
        e_boundary_flow = self.dom.get_array("qe_new")[:, -1]
        self.e_boundary.get_boundary_flow(
            qin=self.dom.get_array("qe_new")[:, -2],
            flow_depth=self.dom.get_array("hfe")[:, -2],
            n=self.dom.get_array("friction")[:, -1],
            z=self.dom.get_array("dem")[:, -1],
            depth=self.dom.get_array("water_depth")[:, -1],
            bctype=self.dom.get_array("bctype")[:, -1],
            bcvalue=self.dom.get_array("bcval")[:, -1],
            qboundary=e_boundary_flow,
        )
        n_boundary_flow = self.dom.get_padded("qs_new")[0, 1:-1]
        self.n_boundary.get_boundary_flow(
            qin=self.dom.get_array("qs_new")[0],
            flow_depth=self.dom.get_array("hfs")[0],
            n=self.dom.get_array("friction")[0],
            z=self.dom.get_array("dem")[0],
            depth=self.dom.get_array("water_depth")[0],
            bctype=self.dom.get_array("bctype")[0],
            bcvalue=self.dom.get_array("bcval")[0],
            qboundary=n_boundary_flow,
        )
        s_boundary_flow = self.dom.get_array("qs_new")[-1]
        self.s_boundary.get_boundary_flow(
            qin=self.dom.get_array("qs_new")[-2],
            flow_depth=self.dom.get_array("hfs")[-2],
            n=self.dom.get_array("friction")[-1],
            z=self.dom.get_array("dem")[-1],
            depth=self.dom.get_array("water_depth")[-1],
            bctype=self.dom.get_array("bctype")[-1],
            bcvalue=self.dom.get_array("bcval")[-1],
            qboundary=s_boundary_flow,
        )
        # add to the accumulation array the equivalent water depth passing through the boundaries
        # Inflow is positive, outflow is negative
        # West (upstream) is inflow (+q)
        self.dom.get_array("boundaries_accum")[:, 0] += w_boundary_flow * self._dt / self.dx
        # East (downstream) is outflow (-q)
        self.dom.get_array("boundaries_accum")[:, -1] -= e_boundary_flow * self._dt / self.dx
        # North (upstream) is inflow (+q)
        self.dom.get_array("boundaries_accum")[0, :] += n_boundary_flow * self._dt / self.dy
        # South (downstream) is outflow (-q)
        self.dom.get_array("boundaries_accum")[-1, :] -= s_boundary_flow * self._dt / self.dy
        return self

    def swap_flow_arrays(self):
        """Swap flow arrays from calculated to input"""
        self.dom.swap_arrays("qe", "qe_new")
        self.dom.swap_arrays("qs", "qs_new")
        return self


class Boundary(object):
    """
    A boundary of the computation domain
    Privilegied access is through get_boundary_flow()
    """

    def __init__(self, cell_width, cell_length, boundary_pos):
        self.position = boundary_pos
        self.cell_width = cell_width
        self.cell_length = cell_length
        if self.position in ("W", "N"):
            self.position_type = "upstream"
        elif self.position in ("E", "S"):
            self.position_type = "downstream"
        else:
            assert False, "Unknown boundary position: {}".format(self.position)

    def get_boundary_flow(self, qin, qboundary, flow_depth, n, z, depth, bctype, bcvalue):
        """Take 1D numpy arrays as input
        Return an updated 1D array of flow through the boundary
        Type 2: flow depth (flow_depth) on the boundary is assumed equal
        to the water depth (depth). i.e. the water depth and terrain
        elevation equal on both sides of the boundary
        Type 3: flow depth is therefore equal to user-defined wse - z
        """
        # check sanity of input arrays
        assert qin.ndim == 1
        assert (
            qin.shape
            == qboundary.shape
            == flow_depth.shape
            == n.shape
            == z.shape
            == depth.shape
            == bctype.shape
            == bcvalue.shape
        )
        # select slices according to boundary types
        slice_closed = np.where((bctype == 0) | (bctype == 1))
        slice_open = np.where(bctype == 2)
        slice_wse = np.where(bctype == 3)
        # Boundary type 1 (closed)
        qboundary[slice_closed] = 0
        # Boundary type 2 (open)
        qboundary[slice_open] = self.get_flow_open_boundary(
            qin[slice_open], flow_depth[slice_open], depth[slice_open]
        )
        # Boundary type 3 (user-defined wse)
        slope = self.get_slope(depth[slice_wse], z[slice_wse], bcvalue[slice_wse])
        hf_boundary = bcvalue[slice_wse] - z[slice_wse]
        qboundary[slice_wse] = self.get_flow_wse_boundary(n[slice_wse], hf_boundary, slope)
        return self

    def get_flow_open_boundary(self, qin, flow_depth, flow_depth_boundary):
        """Velocity at the boundary equals to velocity inside domain.
        The computed flow inherits the sign of the input flow 'qin'.
        """
        result = np.zeros_like(qin)
        slice_over = np.where(flow_depth > 0)
        result[slice_over] = (
            qin[slice_over] / flow_depth[slice_over] * flow_depth_boundary[slice_over]
        )
        return result

    def get_slope(self, h, z, user_wse):
        """Return the slope between two water surface elevation.
        user_wse is outside teh domain.
        Therefore, when the slope is positive the flow goes into the domain."""
        slope = (user_wse - (h + z)) / self.cell_length
        max_slope = 0.5
        return np.minimum(np.fabs(slope), max_slope)

    def get_flow_wse_boundary(self, n, flow_depth, slope):
        """
        Gauckler-Manning-Strickler flow equation
        The sign of the flow depends on the slope direction and boundary
        flow in m2/s
        """
        # Calculate velocity using the magnitude (absolute value) of the slope
        v = (1.0 / n) * np.power(flow_depth, 2.0 / 3.0) * np.power(np.fabs(slope), 1.0 / 2.0)
        q_magnitude = v * flow_depth

        # Determine the sign of the flow based on direction and boundary position.
        # np.sign(slope) > 0 means flow is INTO the domain.
        # np.sign(slope) < 0 means flow is OUT of the domain.
        q_signed = q_magnitude * np.sign(slope)

        if self.position_type == "downstream":
            # For E/S boundaries, flow INTO domain is negative q, flow OUT is positive q.
            # This is the opposite of the slope sign.
            return -q_signed
        else:  # upstream
            # For W/N boundaries, flow INTO domain is positive q, flow OUT is negative q.
            # This is the same as the slope sign.
            return q_signed
