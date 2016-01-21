#! /usr/bin/python
# coding=utf8
"""
Copyright (C) 2015-2016 Laurent Courty

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
import math
import numpy as np
import bottleneck as bn
import flow
import time
from chac_error import NullError

class SurfaceDomain(object):
    """Represents a staggered grid where flow is simulated
    Accessed through step() and get_output_arrays() methods
    By convention the flow is:
     - calculated at the East and South faces of each cell
     - positive from West to East and from North to South
    """

    def __init__(self, dx, dy, arr_def, arr_h,
                sim_clock=0,
                dtmax=None,
                a=None,         # CFL constant
                g=9.80665,     # Standard gravity
                theta=None,
                hf_min=None,
                slope_threshold=None,
                v_routing=None):  # simple routing velocity m/s
        assert dtmax != None
        assert a != None
        assert theta != None
        assert hf_min != None
        assert slope_threshold != None
        assert v_routing != None

        self.sim_clock = sim_clock
        self.dtmax = dtmax
        self.cfl = a
        self.g = g
        self.theta = theta
        self.hf_min = hf_min
        self.sl_thresh = slope_threshold
        self.v_routing = v_routing
        self.dx = dx
        self.dy = dy
        self.cell_surf = dx * dy

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

        # Input water depth
        self.arr_h = arr_h
        # Set internal arrays to a provided default
        # Input arrays are set externally with set_input_arrays()
        # max depth
        self.arr_hmax = np.copy(arr_def)
        # flow depth
        self.arr_hfe = np.copy(arr_def)
        self.arr_hfs = np.copy(arr_def)
        # flows in m2/s
        self.arr_qe, self.arrp_qe = self.pad_array(np.copy(arr_def))
        self.arr_qs, self.arrp_qs = self.pad_array(np.copy(arr_def))
        self.arr_qe_new, self.arrp_qe_new = self.pad_array(np.copy(arr_def))
        self.arr_qs_new, self.arrp_qs_new = self.pad_array(np.copy(arr_def))
        # arrays of flow vector norm
        self.arr_qe_norm = (np.copy(arr_def))
        self.arr_qs_norm = (np.copy(arr_def))
        # direction arrays
        self.arr_dire = np.full(arr_def[self.s_i_0].shape, -1, dtype = np.int8)
        self.arr_dirs = np.full(arr_def[self.s_j_0].shape, -1, dtype = np.int8)
        del arr_def

        # Instantiate boundary objects
        self.w_boundary = Boundary(self.dy, self.dx, boundary_pos='W')
        self.e_boundary = Boundary(self.dy, self.dx, boundary_pos='E')
        self.n_boundary = Boundary(self.dx, self.dy, boundary_pos='N')
        self.s_boundary = Boundary(self.dx, self.dy, boundary_pos='S')

    @staticmethod
    def pad_array(arr):
        """Return the original input array
        as a slice of a larger padded array with one cell
        """
        arr_p = np.pad(arr, 1, 'edge')
        arr = arr_p[1:-1,1:-1]
        return arr, arr_p

    def update_flow_dir(self):
        ''' Return arrays of flow directions used for rain routing
        each cell is assigned a direction in which it will drain
        0: the flow is going dowstream, index-wise
        1: the flow is going upstream, index-wise
        -1: no routing happening on that face
        '''
        # get a padded array
        arrp_z = self.pad_array(self.arr_z)[1]
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
        flow.flow_dir(arr_max_dz[self.s_j_0], dN, dS, self.arr_dirs)
        # x direction
        flow.flow_dir(arr_max_dz[self.s_i_0], dW, dE, self.arr_dire)
        return self

    def step(self, next_ts, massbal):
        """Run a full simulation time-step
        """
        start_clock = time.clock()
        self.set_dt(next_ts)
        self.solve_q()
        boundary_vol = self.apply_boundary_conditions()
        if massbal:
            massbal.add_value('boundary_vol', boundary_vol)
            massbal.add_value('old_dom_vol', self.domain_volume())
        self.update_h()
        if massbal:
            massbal.add_value('hfix_vol', self.hfix_vol)
            massbal.add_value('new_dom_vol', self.domain_volume())
        # in case of NaN/NULL cells, raise a NullError to be catched by run()
        self.arr_err = np.isnan(self.arr_h)
        if np.any(self.arr_err):
            raise NullError

        self.copy_arrays_values_for_next_timestep()
        end_clock = time.clock()
        if massbal:
            massbal.add_value('step_duration', end_clock - start_clock)
        return self

    def set_dt(self, next_ts):
        """Calculate the adaptative time-step
        The formula #15 in almeida et al (2012) has been modified to
        accomodate non-square cells
        The time-step is limited by the maximum time-step dtmax.
        """
        hmax = np.amax(self.arr_h)  # max depth in domain
        min_dim = min(self.dx, self.dy)
        if hmax > 0:
            self.dt = min(self.dtmax,
                    self.cfl * ( min_dim / (math.sqrt(self.g * hmax))))
        else:
            self.dt = self.dtmax

        # set sim_clock and check if timestep is within forced_timestep
        if self.sim_clock + self.dt > next_ts:
            self.dt = next_ts - self.sim_clock
            self.sim_clock += self.dt
        else:
            self.sim_clock += self.dt
        return self

    def domain_volume(self):
        '''return domain volume
        '''
        return bn.nansum(self.arr_h) * self.cell_surf

    def update_h(self):
        """Calculate new water depth
        """
        flow_west = self.arrp_qe_new[self.ss, self.su]
        flow_east = self.arr_qe_new
        flow_north = self.arrp_qs_new[self.su, self.ss]
        flow_south = self.arr_qs_new
        assert flow_west.shape == flow_east.shape == flow_north.shape == flow_south.shape

        self.hfix_vol = flow.solve_h(arr_ext=self.arr_ext,
                    arr_qe=flow_east, arr_qw=flow_west,
                    arr_qn=flow_north, arr_qs=flow_south,
                    arr_bct=self.arr_bctype, arr_bcv=self.arr_bcval,
                    arr_h=self.arr_h, arr_hmax=self.arr_hmax,
                    dx=self.dx, dy=self.dy, dt=self.dt)
        assert not np.any(self.arr_h < 0)
        return self

    def solve_q(self):
        '''Solve flow inside the domain using C/Cython function
        prepare the arrays slices and pass them to the Cython function
        '''
        # Those are for padded flow arrays only.
        # Used to get the flow of the first boundary
        s_i_m1 = (self.ss, slice(0, -3))
        s_j_m1 = (slice(0, -3), self.ss)

        z_i0 = self.arr_z[self.s_i_0]
        z_i1 = self.arr_z[self.s_i_1]
        z_j0 = self.arr_z[self.s_j_0]
        z_j1 = self.arr_z[self.s_j_1]
        assert z_i0.shape == z_i1.shape
        assert z_j0.shape == z_j1.shape

        h_i0 = self.arr_h[self.s_i_0]
        h_i1 = self.arr_h[self.s_i_1]
        h_j0 = self.arr_h[self.s_j_0]
        h_j1 = self.arr_h[self.s_j_1]
        assert h_i0.shape == h_i1.shape
        assert h_j0.shape == h_j1.shape

        n_i0 = self.arr_n[self.s_i_0]
        n_i1 = self.arr_n[self.s_i_1]
        n_j0 = self.arr_n[self.s_j_0]
        n_j1 = self.arr_n[self.s_j_1]
        assert n_i0.shape == n_i1.shape
        assert n_j0.shape == n_j1.shape

        # flow depths
        hf_i = self.arr_hfe[self.s_i_0]
        hf_j = self.arr_hfs[self.s_j_0]

        # values in the Y dim, used to calculate qnorm in X dimension
        arr_qs_i_j = self.arr_qs
        arr_qs_i_ju = self.arrp_qs[self.su, self.ss]
        arr_qs_id_j = self.arrp_qs[self.ss, self.sd]
        arr_qs_id_ju = self.arrp_qs[self.su, self.sd]
        assert arr_qs_i_j.shape == arr_qs_i_ju.shape == arr_qs_id_j.shape
        assert arr_qs_i_j.shape == arr_qs_id_ju.shape

        # values in the X dim, used to calculate qnorm in Y dimension
        arr_qe_i_j = self.arr_qe
        arr_qe_iu_j = self.arrp_qe[self.ss, self.su]
        arr_qe_i_jd = self.arrp_qe[self.sd, self.ss]
        arr_qe_iu_jd = self.arrp_qe[self.sd, self.su]
        assert arr_qe_i_j.shape == arr_qe_iu_j.shape == arr_qe_i_jd.shape
        assert arr_qe_i_j.shape == arr_qe_iu_jd.shape

        # flows
        q_vect_i = self.arr_qe_norm[self.s_i_0]
        q_vect_j = self.arr_qs_norm[self.s_j_0]
        q_i0 = self.arr_qe[self.s_i_0]
        q_i1 = self.arr_qe[self.s_i_1]
        q_j0 = self.arr_qs[self.s_j_0]
        q_j1 = self.arr_qs[self.s_j_1]
        # Uses padded array to get boundary flow
        q_im1 = self.arrp_qe[s_i_m1]
        q_jm1 = self.arrp_qs[s_j_m1]
        assert q_vect_i.shape == q_i0.shape == q_i1.shape == q_im1.shape
        assert q_vect_j.shape == q_j0.shape == q_j1.shape == q_jm1.shape

        q_i0_new = self.arr_qe_new[self.s_i_0]
        q_j0_new = self.arr_qs_new[self.s_j_0]

        # flow in x direction
        assert z_i0.shape == z_i1.shape == n_i0.shape == n_i1.shape
        assert n_i0.shape == h_i0.shape == h_i1.shape == q_i0.shape
        assert q_i0.shape == q_i1.shape == q_im1.shape == q_vect_i.shape
        assert q_vect_i.shape == q_i0_new.shape
        flow.solve_q(arr_dir=self.arr_dire,
            arr_z0=z_i0, arr_z1=z_i1,
            arr_n0=n_i0, arr_n1=n_i1,
            arr_h0=h_i0, arr_h1=h_i1,
            arr_q0=q_i0, arr_q1=q_i1, arr_qm1=q_im1,
            arr_qn1=arr_qs_i_j, arr_qn2=arr_qs_i_ju,
            arr_qn3=arr_qs_id_j, arr_qn4=arr_qs_id_ju,
            arr_q0_new=q_i0_new, arr_hf=hf_i,
            dt=self.dt, cell_len=self.dx, g=self.g, theta=self.theta,
            hf_min=self.hf_min, v_rout=self.v_routing, sl_thres=self.sl_thresh)
        # flow in y direction
        assert z_j0.shape == z_j1.shape == n_j0.shape == n_j1.shape
        assert n_j0.shape == h_j0.shape == h_j1.shape == q_j0.shape
        assert q_j0.shape == q_j1.shape == q_jm1.shape == q_vect_j.shape
        assert q_vect_j.shape == q_j0_new.shape
        flow.solve_q(arr_dir=self.arr_dirs,
            arr_z0=z_j0, arr_z1=z_j1,
            arr_n0=n_j0, arr_n1=n_j1,
            arr_h0=h_j0, arr_h1=h_j1,
            arr_q0=q_j0, arr_q1=q_j1, arr_qm1=q_jm1,
            arr_qn1=arr_qe_iu_j, arr_qn2=arr_qe_i_jd,
            arr_qn3=arr_qe_i_jd, arr_qn4=arr_qe_iu_jd,
            arr_q0_new=q_j0_new, arr_hf=hf_j,
            dt=self.dt, cell_len=self.dy, g=self.g, theta=self.theta,
            hf_min=self.hf_min, v_rout=self.v_routing, sl_thres=self.sl_thresh)
        return self

    def apply_boundary_conditions(self):
        '''Select relevant 1D slices and apply boundary conditions.
        1D arrays passed to the boundary method include cells bordering
        the boundary on the inside of the domain.
        For the values applying at cells interface (flow depth and flow):
        'qboundary' is the flow at the very boundary and is updated
        'hflow' and 'qin' are the next value inside the domain
        Therefore, only 'qboundary' should need a padded array.
        '''

        w_boundary_flow = self.arrp_qe_new[1:-1, 0]
        self.w_boundary.get_boundary_flow(qin=self.arr_qe_new[:, 0],
                                    hflow=self.arr_hfe[:, 0],
                                    n=self.arr_n[:, 0],
                                    z=self.arr_z[:, 0],
                                    depth=self.arr_h[:, 0],
                                    bctype=self.arr_bctype[:, 0],
                                    bcvalue=self.arr_bcval[:, 0],
                                    qboundary=w_boundary_flow)
        e_boundary_flow = self.arr_qe_new[:, -1]
        self.e_boundary.get_boundary_flow(qin=self.arr_qe_new[:, -2],
                                    hflow=self.arr_hfe[:, -2],
                                    n=self.arr_n[:, -1],
                                    z=self.arr_z[:, -1],
                                    depth=self.arr_h[:, -1],
                                    bctype=self.arr_bctype[:, -1],
                                    bcvalue=self.arr_bcval[:, -1],
                                    qboundary=e_boundary_flow)
        n_boundary_flow = self.arrp_qs_new[0, 1:-1]
        self.n_boundary.get_boundary_flow(qin=self.arr_qs_new[0],
                                    hflow=self.arr_hfs[0],
                                    n=self.arr_n[0],
                                    z=self.arr_z[0],
                                    depth=self.arr_h[0],
                                    bctype=self.arr_bctype[0],
                                    bcvalue=self.arr_bcval[0],
                                    qboundary=n_boundary_flow)
        s_boundary_flow = self.arr_qs_new[-1]
        self.s_boundary.get_boundary_flow(qin=self.arr_qs_new[-2],
                                    hflow=self.arr_hfs[-2],
                                    n=self.arr_n[-1],
                                    z=self.arr_z[-1],
                                    depth=self.arr_h[-1],
                                    bctype=self.arr_bctype[-1],
                                    bcvalue=self.arr_bcval[-1],
                                    qboundary=s_boundary_flow)
        # calculate volume entering through boundaries
        x_boundary_len = (w_boundary_flow.shape[0] + e_boundary_flow.shape[0]) * self.dy
        y_boundary_len = (n_boundary_flow.shape[0] + s_boundary_flow.shape[0]) * self.dx
        x_boundary_flow = (bn.nansum(w_boundary_flow) - bn.nansum(e_boundary_flow)) * x_boundary_len
        y_boundary_flow = (bn.nansum(n_boundary_flow) - bn.nansum(s_boundary_flow)) * y_boundary_len
        boundary_vol = (x_boundary_flow + y_boundary_flow)
        return boundary_vol

    def copy_arrays_values_for_next_timestep(self):
        """Copy values from calculated arrays to input arrays
        """
        self.arr_qe[:] = self.arr_qe_new
        self.arr_qs[:] = self.arr_qs_new
        return self

    def get_output_arrays(self, out_names):
        """Takes a dict of map names
        return a dict of arrays
        """
        out_arrays = {}
        if out_names['out_h'] != None:
            out_arrays['out_h'] = self.arr_h
        if out_names['out_wse'] != None:
            out_arrays['out_wse'] = self.arr_h + self.arr_z
        if out_names['out_vx'] != None:
            pass
        if out_names['out_vy'] != None:
            pass
        if out_names['out_qx'] != None:
            pass
        if out_names['out_qy'] != None:
            pass
        return out_arrays


class Boundary(object):
    """
    A boundary of the computation domain
    Privilegied access is through get_boundary_flow()
    """
    def __init__(self, cell_width, cell_length, boundary_pos):
        self.pos = boundary_pos
        self.cw = cell_width
        self.cl = cell_length
        if self.pos in ('W', 'N'):
            self.postype = 'upstream'
        elif self.pos in ('E', 'S'):
            self.postype = 'downstream'
        else:
            assert False, "Unknown boundary position: {}".format(self.pos)

    def get_boundary_flow(self, qin, qboundary, hflow, n, z, depth,
                            bctype, bcvalue):
        """Take 1D numpy arrays as input
        Return an updated 1D array of flow through the boundary
        Type 2: flow depth (hflow) on the boundary is assumed equal
        to the water depth (depth). i.e. the water depth and terrain
        elevation equal on both sides of the boundary
        Type 3: flow depth is therefore equal to user-defined wse - z
        """
        # check sanity of input arrays
        assert qin.ndim == 1
        assert (qin.shape == qboundary.shape == hflow.shape == n.shape ==
                z.shape == depth.shape == bctype.shape == bcvalue.shape)
        # select slices according to boundary types
        slice_closed = np.where(bctype == 1)
        slice_open = np.where(bctype == 2)
        slice_wse = np.where(bctype == 3)
        # Boundary type 1 (closed)
        qboundary[slice_closed] = 0
        # Boundary type 2 (open)
        qboundary[slice_open] = self.get_flow_open_boundary(qin[slice_open],
                                hflow[slice_open], depth[slice_open])
        # Boundary type 3 (user-defined wse)
        slope = self.get_slope(depth[slice_wse],
                            z[slice_wse], bcvalue[slice_wse])
        hf_boundary = bcvalue[slice_wse] - z[slice_wse]
        qboundary[slice_wse] = self.get_flow_wse_boundary(n[slice_wse],
                                hf_boundary, slope)
        return self

    def get_flow_open_boundary(self, qin, hf, hf_boundary):
        """Velocity at the boundary equal to velocity inside domain
        """
        result = np.zeros_like(qin)
        slice_over = np.where(hf > 0)
        result[slice_over] = qin[slice_over] / hf[slice_over] * hf_boundary[slice_over]
        return result

    def get_slope(self, h, z, user_wse):
        """Return the slope between two water surface elevation
        """
        slope = (user_wse - (h + z)) / self.cl
        max_slope = 0.5
        return np.minimum(np.fabs(slope), max_slope)

    def get_flow_wse_boundary(self, n, hf, slope):
        """
        Gauckler-Manning-Strickler flow equation
        invert the results if a downstream boundary
        flow in m2/s
        """
        v = (1./n) * np.power(hf, 2./3.) * np.power(slope, 1./2.)
        if self.postype == 'upstream':
            return v * hf
        elif self.postype == 'downstream':
            return - v * hf
        else:
            assert False, "Unknown postype {}".format(self.postype)
