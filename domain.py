#! /usr/bin/python
# coding=utf8
"""
Copyright (C) 2015  Laurent Courty

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
#~ import pyximport
import math
import numpy as np
import numexpr as ne
#~ pyximport.install(setup_args={'include_dirs': np.get_include()})
import flow

class SurfaceDomain(object):
    """Represents a staggered grid where flow is simulated
    Accessed through step() and get_output_arrays() methods
    By convention the flow is:
     - calculated at the East and South faces of each cell
     - positive from West to East and from North to South
    """

    def __init__(self, dx, dy, arr_def, arr_h,
                sim_clock=0,
                dtmax=5,
                a=0.5,         # CFL constant
                g=9.80665,     # Standard gravity
                theta=0.9,     # default proposed by Almeida et al.(2012)
                hf_min=0.0001,
                slope_threshold=0.05,
                v_routing=0.1):  # simple routing velocity m/s):
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
        self.arr_h_old = arr_h
        # Set internal arrays to a provided default
        # Input arrays are set externally with set_input_arrays()
        self.arr_h_new = np.copy(arr_def)
        # flow depth
        self.arr_hfe = np.copy(arr_def)
        self.arr_hfs = np.copy(arr_def)
        # wse slope without boundary value
        self.arr_wse_sle = np.copy(arr_def[self.s_i_0])
        self.arr_wse_sls = np.copy(arr_def[self.s_j_0])
        # terrain slope without boundary value
        self.arr_dem_sle = np.copy(arr_def[self.s_i_0])
        self.arr_dem_sls = np.copy(arr_def[self.s_j_0])
        # flows in m2/s
        self.arr_qe, self.arrp_qe = self.pad_array(np.copy(arr_def))
        self.arr_qs, self.arrp_qs = self.pad_array(np.copy(arr_def))
        self.arr_qe_new, self.arrp_qe_new = self.pad_array(np.copy(arr_def))
        self.arr_qs_new, self.arrp_qs_new = self.pad_array(np.copy(arr_def))
        # arrays of flow vector norm
        self.arr_qe_norm = (np.copy(arr_def))
        self.arr_qs_norm = (np.copy(arr_def))
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

    def update_dem_slope(self):
        '''Calculate the terrain slope at cell interface over the domain
        '''
        dem_i0 = self.arr_z[self.s_i_0]
        dem_i1 = self.arr_z[self.s_i_1]
        dem_j0 = self.arr_z[self.s_j_0]
        dem_j1 = self.arr_z[self.s_j_1]

        self.arr_dem_sle[:] = (dem_i1 - dem_i0) / self.dx
        self.arr_dem_sls[:] = (dem_j1 - dem_j0) / self.dy
        return self

    def update_surface_slope(self):
        '''Calculate the WSE slope at cell interface over the domain
        '''
        wse_i0 = self.arr_z[self.s_i_0] + self.arr_h_old[self.s_i_0]
        wse_i1 = self.arr_z[self.s_i_1] + self.arr_h_old[self.s_i_1]
        wse_j0 = self.arr_z[self.s_j_0] + self.arr_h_old[self.s_j_0]
        wse_j1 = self.arr_z[self.s_j_1] + self.arr_h_old[self.s_j_1]

        self.arr_wse_sle[:] = (wse_i1 - wse_i0) / self.dx
        self.arr_wse_sls[:] = (wse_j1 - wse_j0) / self.dy
        return self

    def step(self, next_ts, massbal):
        """Run a full simulation time-step
        """
        self.set_dt(next_ts)
        self.solve_q()
        #~ self.solve_routing_flow()
        boundary_vol = self.apply_boundary_conditions()
        massbal.add_value('boundary_vol', boundary_vol)
        self.solve_h()
        massbal.add_value('old_dom_vol', self.old_domain_volume())
        self.update_surface_slope()
        self.copy_arrays_values_for_next_timestep()
        massbal.add_value('new_dom_vol', self.old_domain_volume())
        return self

    def set_dt(self, next_ts):
        """Calculate the adaptative time-step
        The formula #15 in almeida et al (2012) has been modified to
        accomodate non-square cells
        The time-step is limited by the maximum time-step dtmax.
        """
        hmax = np.amax(self.arr_h_old)  # max depth in domain
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

    def old_domain_volume(self):
        '''return domain volume from old h
        '''
        return np.sum(self.arr_h_old) * self.cell_surf

    def solve_h(self):
        """Calculate new water depth
        """
        # flows converted from m2/s to m3/s
        flow_west = self.arrp_qe_new[self.ss, self.su]
        flow_east = self.arr_qe_new
        flow_north = self.arrp_qs_new[self.su, self.ss]
        flow_south = self.arr_qs_new
        assert flow_west.shape == flow_east.shape == flow_north.shape == flow_south.shape
        arr_Q_sum = ((flow_west - flow_east) * self.dy
                    + (flow_north - flow_south) * self.dx)

        # arr_ext converted from m/s to m, Q from m3/s to m
        self.arr_h_new[:] = (self.arr_h_old +
                            self.arr_ext * self.dt +
                            (arr_Q_sum / self.cell_surf) * self.dt)
        # set to zero if negative
        #~ if np.any(self.arr_h_new < 0):
        self.arr_h_new[:] = np.maximum(0, self.arr_h_new)
        assert not np.any(self.arr_h_new == np.nan)
        return self

    def solve_hflow(self, wse_i_d, wse_i, z_i_d, z_i,
                        wse_j_d, wse_j, z_j_d, z_j):
        """calculate the difference between
        the highest water surface elevation
        and the highest terrain elevevation of the two adjacent cells
        This is used as an approximation of the hydraulic radius
        the last row/col (i.e. on boundary) is not calculated
        """
        self.arr_hfe[:, self.sc] = (np.maximum(wse_i_d, wse_i)
                                     - np.maximum(z_i_d, z_i))
        self.arr_hfs[self.sc, :] = (np.maximum(wse_j_d, wse_j)
                                       - np.maximum(z_j_d, z_j))
        return self

    def almeida2013(self, length, width, wse0, wsem1, hf, q0, qnorm, qm1, qp1, n):
        '''Flow formula from Almeida & Bates 2013
        return a flow in m2/s'''
        if hf <= self.hf_min:
            return 0
        else:
            # flow formula (formula #41 in almeida et al 2012)
            term_1 = (self.theta * q0 + ((1 - self.theta) / 2) * (qm1 + qp1))
            term_2 = (self.g * hf * (self.dt / length) * (wse0 - wsem1))
            term_3 = (1 + self.g * self.dt * (n*n) * qnorm / pow(hf, 7./3.))
            q0_new = (term_1 - term_2) / term_3
            return q0_new

    def solve_qnorm(self):
        """Calculate the flow vector norm to be used in the flow equation
        This method uses values in i-1 and j-1, which seems more logical
        than the version given in Almeida and Bates (2013)
        """
        # values in the Y dim, used to calculate an average of Y flows
        arr_qs_i_j = self.arr_qs
        arr_qs_i_ju = self.arrp_qs[self.su, self.ss]
        arr_qs_id_j = self.arrp_qs[self.ss, self.sd]
        arr_qs_id_ju = self.arrp_qs[self.su, self.sd]
        # values in the X dim, used to calculate an average of X flows
        arr_qe_i_j = self.arr_qe
        arr_qe_iu_j = self.arrp_qe[self.ss, self.su]
        arr_qe_i_jd = self.arrp_qe[self.sd, self.ss]
        arr_qe_iu_jd = self.arrp_qe[self.sd, self.su]

        # average values of flows in relevant dimension
        arr_qs_av = (arr_qs_i_j + arr_qs_i_ju + arr_qs_id_j + arr_qs_id_ju) * .25
        arr_qe_av = (arr_qe_i_j + arr_qe_iu_j + arr_qe_i_jd + arr_qe_iu_jd) * .25

        # norm for one dim. uses the average of flows in the other dim.
        self.arr_qe_norm[:] = np.sqrt(np.square(arr_qs_av) + np.square(arr_qe_i_j))
        self.arr_qs_norm[:] = np.sqrt(np.square(arr_qe_av) + np.square(arr_qs_i_j))
        return self

    def add_arrays(self, arr1, arr2):
        return arr1 + arr2

    def div_arrays(self, arr1, arr2):
        return arr1 / arr2

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

        h_i0 = self.arr_h_old[self.s_i_0]
        h_i1 = self.arr_h_old[self.s_i_1]
        h_j0 = self.arr_h_old[self.s_j_0]
        h_j1 = self.arr_h_old[self.s_j_1]
        assert h_i0.shape == h_i1.shape
        assert h_j0.shape == h_j1.shape

        n_i0 = self.arr_n[self.s_i_0]
        n_i1 = self.arr_n[self.s_i_1]
        n_j0 = self.arr_n[self.s_j_0]
        n_j1 = self.arr_n[self.s_j_1]
        assert n_i0.shape == n_i1.shape
        assert n_j0.shape == n_j1.shape

        # flows
        self.solve_qnorm()
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
        flow.solve_q(
            arr_z0=z_i0, arr_z1=z_i1,
            arr_n0=n_i0, arr_n1=n_i1,
            arr_h0=h_i0, arr_h1=h_i1,
            arr_q0=q_i0, arr_q1=q_i1, arr_qm1=q_im1,
            arr_qnorm=q_vect_i, arr_q0_new=q_i0_new,
            dt=self.dt, cell_len=self.dx, g=self.g,
            theta=self.theta, hf_min=self.hf_min)
        # flow in y direction
        assert z_j0.shape == z_j1.shape == n_j0.shape == n_j1.shape
        assert n_j0.shape == h_j0.shape == h_j1.shape == q_j0.shape
        assert q_j0.shape == q_j1.shape == q_jm1.shape == q_vect_j.shape
        assert q_vect_j.shape == q_j0_new.shape
        flow.solve_q(
            arr_z0=z_j0, arr_z1=z_j1,
            arr_n0=n_j0, arr_n1=n_j1,
            arr_h0=h_j0, arr_h1=h_j1,
            arr_q0=q_j0, arr_q1=q_j1, arr_qm1=q_jm1,
            arr_qnorm=q_vect_j, arr_q0_new=q_j0_new,
            dt=self.dt, cell_len=self.dy, g=self.g,
            theta=self.theta, hf_min=self.hf_min)

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
                                    depth=self.arr_h_old[:, 0],
                                    bctype=self.arr_bctype[:, 0],
                                    bcvalue=self.arr_bcval[:, 0],
                                    qboundary=w_boundary_flow)
        e_boundary_flow = self.arr_qe_new[:, -1]
        self.e_boundary.get_boundary_flow(qin=self.arr_qe_new[:, -2],
                                    hflow=self.arr_hfe[:, -2],
                                    n=self.arr_n[:, -1],
                                    z=self.arr_z[:, -1],
                                    depth=self.arr_h_old[:, -1],
                                    bctype=self.arr_bctype[:, -1],
                                    bcvalue=self.arr_bcval[:, -1],
                                    qboundary=e_boundary_flow)
        n_boundary_flow = self.arrp_qs_new[0, 1:-1]
        self.n_boundary.get_boundary_flow(qin=self.arr_qs_new[0],
                                    hflow=self.arr_hfs[0],
                                    n=self.arr_n[0],
                                    z=self.arr_z[0],
                                    depth=self.arr_h_old[0],
                                    bctype=self.arr_bctype[0],
                                    bcvalue=self.arr_bcval[0],
                                    qboundary=n_boundary_flow)
        s_boundary_flow = self.arr_qs_new[-1]
        self.s_boundary.get_boundary_flow(qin=self.arr_qs_new[-2],
                                    hflow=self.arr_hfs[-2],
                                    n=self.arr_n[-1],
                                    z=self.arr_z[-1],
                                    depth=self.arr_h_old[-1],
                                    bctype=self.arr_bctype[-1],
                                    bcvalue=self.arr_bcval[-1],
                                    qboundary=s_boundary_flow)
        # calculate volume entering through boundaries
        x_boundary_len = (w_boundary_flow.shape[0] + e_boundary_flow.shape[0]) * self.dy
        y_boundary_len = (n_boundary_flow.shape[0] + s_boundary_flow.shape[0]) * self.dx
        x_boundary_flow = (np.sum(w_boundary_flow) - np.sum(e_boundary_flow)) * x_boundary_len
        y_boundary_flow = (np.sum(n_boundary_flow) - np.sum(s_boundary_flow)) * y_boundary_len
        boundary_vol = (x_boundary_flow + y_boundary_flow)
        return boundary_vol

    def copy_arrays_values_for_next_timestep(self):
        """Copy values from calculated arrays to input arrays
        """
        self.arr_qe[:] = self.arr_qe_new
        self.arr_qs[:] = self.arr_qs_new
        self.arr_h_old[:] = self.arr_h_new
        return self

    def get_output_arrays(self, out_names):
        """Takes a dict of map names
        return a dict of arrays
        """
        out_arrays = {}
        if out_names['out_h'] != None:
            out_arrays['out_h'] = self.arr_h_new
        if out_names['out_wse'] != None:
            out_arrays['out_wse'] = self.arr_h_new + self.arr_z
        if out_names['out_vx'] != None:
            pass
        if out_names['out_vy'] != None:
            pass
        if out_names['out_qx'] != None:
            pass
        if out_names['out_qy'] != None:
            pass
        return out_arrays

    def set_routing_flow(self, h0, h1, z0, z1):
        '''Return a routing flow in m3/s
        '''
        # fraction of the depth to be routed
        dh = (z0 + h0) - (z1 + h1)
        # if WSE of neighbour is below the dem of the current cell, set to h0
        dh[:] = np.minimum(dh, h0)
        # don't allow reverse flow
        dh[:] = np.maximum(dh, 0.)
        # fraction of the flow to be routed during the time-step
        flow_fraction = self.v_routing / self.dx  # !! should be the cell length
        # prevent over-drainage of the cell in case of long time-step
        if flow_fraction * self.dt > 1:
            flow_fraction = 1 / self.dt
        route_q = dh * flow_fraction * self.cell_surf
        return route_q

    def solve_routing_flow(self):
        '''Select the cells where the dem slope is above the threshold
        where slope is positive, assign inverse of routing_flow() to arr_q
        where slope is negative, assign routing_flow() to arr_q
        '''
        # boolean arrays where slope is above threshold
        b_slope_sup_i = (np.fabs(self.arr_dem_sle) > self.sl_thresh)
        b_slope_sup_j = (np.fabs(self.arr_dem_sls) > self.sl_thresh)
        # boolean arrays where slopes are above threshold and positive
        b_pos_i = np.logical_and(b_slope_sup_i,
            np.logical_and(self.arr_wse_sle > 0., self.arr_dem_sle > 0.))
        b_pos_j = np.logical_and(b_slope_sup_j,
            np.logical_and(self.arr_wse_sls > 0., self.arr_dem_sls > 0.))
        # boolean arrays where slopes are above threshold and negative
        b_neg_i =np.logical_and(b_slope_sup_i,
            np.logical_and(self.arr_wse_sle < 0., self.arr_dem_sle < 0.))
        b_neg_j =np.logical_and(b_slope_sup_j,
            np.logical_and(self.arr_wse_sls < 0., self.arr_dem_sls < 0.))

        # values to be used for flow calculation
        z_i0 = self.arr_z[self.s_i_0]
        z_i1 = self.arr_z[self.s_i_1]
        z_j0 = self.arr_z[self.s_j_0]
        z_j1 = self.arr_z[self.s_j_1]
        h_i0 = self.arr_h_old[self.s_i_0]
        h_i1 = self.arr_h_old[self.s_i_1]
        h_j0 = self.arr_h_old[self.s_j_0]
        h_j1 = self.arr_h_old[self.s_j_1]

        # assign flows for positive slopes
        self.arr_qe_new[self.s_i_0] = np.where(b_pos_i,
            - self.set_routing_flow(h0=h_i1, h1=h_i0, z0=z_i1, z1=z_i0),
            self.arr_qe_new[self.s_i_0])
        self.arr_qs_new[self.s_j_0] = np.where(b_pos_j,
            -  self.set_routing_flow(h0=h_j1, h1=h_j0, z0=z_j1, z1=z_j0),
            self.arr_qs_new[self.s_j_0])
        # assign flows for negative slopes
        self.arr_qe_new[self.s_i_0] = np.where(b_neg_i,
            self.set_routing_flow(h0=h_i0, h1=h_i1, z0=z_i0, z1=z_i1),
            self.arr_qe_new[self.s_i_0])
        self.arr_qs_new[self.s_j_0] = np.where(b_neg_j,
            self.set_routing_flow(h0=h_j0, h1=h_j1, z0=z_j0, z1=z_j1),
                                self.arr_qs_new[self.s_j_0])
        return self


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


class old_code():

    def flow_dir(self):
        '''
        Return a "slope" array representing the flow direction
        the resulting array is used for simple routing
        '''
        # define differences in Z
        Z = self.arrp_z[1:-1, 1:-1]
        N = self.arrp_z[0:-2, 1:-1]
        S = self.arrp_z[2:, 1:-1]
        E = self.arrp_z[1:-1, 2:]
        W = self.arrp_z[1:-1, 0:-2]
        dN = Z - N
        dE = Z - E
        dS = Z - S
        dW = Z - W

        # return minimum neighbour
        arr_min = np.maximum(np.maximum(dN, dS), np.maximum(dE, dW))

        # create a slope array
        self.arr_slope = np.zeros(shape = Z.shape, dtype = np.uint8)
        # affect values to keep a non-string array
        # (to be compatible with cython)
        W = 1
        E = 2
        S = 3
        N = 4

        for c, min_hdiff in np.ndenumerate(arr_min):
            if min_hdiff <= 0:
                self.arr_slope[c] = 0
            elif min_hdiff == dN[c]:
                self.arr_slope[c] = N
            elif min_hdiff == dS[c]:
                self.arr_slope[c] = S
            elif min_hdiff == dE[c]:
                self.arr_slope[c] = E
            elif min_hdiff == dW[c]:
                self.arr_slope[c] = W
        return self

    def simple_routing(self):
        '''calculate a flow with a given velocity in a given direction
        Application of the automated routing scheme proposed by
        Sampson et al (2013)
        For flows at W and S borders, the calculations are centered
        on the current cells.
        For flows at E and S borders, are considered
        those of the neighbouring cells at W and S borders, respectively
        '''
        # water surface elevation
        arrp_wse_np1 = self.arrp_z + self.arrp_h_np1

        # water depth
        arrp_h_np1_e = self.arrp_h_np1[1:-1, 2:]
        arrp_h_np1_n = self.arrp_h_np1[:-2, 1:-1]

        # arrays of wse
        arr_wse_w = arrp_wse_np1[1:-1, :-2]
        arr_wse_e = arrp_wse_np1[1:-1, 2:]
        arr_wse_s = arrp_wse_np1[2:, 1:-1]
        arr_wse_n = arrp_wse_np1[:-2, 1:-1]
        arr_wse = arrp_wse_np1[1:-1, 1:-1]

        # Identify cells with adequate values
        idx_rout = np.where(np.logical_and(
                        self.arr_h_np1 < self.hmin,
                        self.arr_h_np1 > 0))

        # max routed depth
        arr_h_w = np.minimum(self.arr_h_np1[idx_rout],
                    np.maximum(0, arr_wse[idx_rout] - arr_wse_w[idx_rout]))
        arr_h_s = np.minimum(self.arr_h_np1[idx_rout],
                    np.maximum(0, arr_wse[idx_rout] - arr_wse_s[idx_rout]))
        arr_h_e = np.minimum(self.arr_h_np1[idx_rout],
                    np.maximum(0, arr_wse[idx_rout] - arr_wse_e[idx_rout]))
        arr_h_n = np.minimum(self.arr_h_np1[idx_rout],
                    np.maximum(0, arr_wse[idx_rout] - arr_wse_n[idx_rout]))

        # arrays of flows
        arr_q_w = self.arr_q[idx_rout]['W']
        arr_q_s = self.arr_q[idx_rout]['S']
        # flows of the neighbouring cells for E and N
        arr_q_e = self.arrp_q[1:-1, 2:]['W']
        arr_q_n = self.arrp_q[:-2, 1:-1]['S']
        arr_q_e = arr_q_e[idx_rout]
        arr_q_n = arr_q_n[idx_rout]

        # arrays of calculated routing flow
        arr_sflow_w = - self.dx * arr_h_w * self.v_routing
        arr_sflow_s = - self.dy * arr_h_s * self.v_routing
        arr_sflow_e = self.dx * arr_h_e * self.v_routing
        arr_sflow_n = self.dy * arr_h_n * self.v_routing

        # can route
        idx_route_s = np.where(self.arr_slope[idx_rout] == 'S')
        idx_route_w = np.where(self.arr_slope[idx_rout] == 'W')
        idx_route_e = np.where(self.arr_slope[idx_rout] == 'E')
        idx_route_n = np.where(self.arr_slope[idx_rout] == 'N')

        # affect calculated flow to the flow grid
        arr_q_s[idx_route_s] = arr_sflow_s[idx_route_s]
        arr_q_w[idx_route_w] = arr_sflow_w[idx_route_w]
        arr_q_e[idx_route_e] = arr_sflow_e[idx_route_e]
        arr_q_n[idx_route_n] = arr_sflow_n[idx_route_n]

        return self

    def solve_q(self):
        # Bates 2010
        arr_qw_new = self.arr_qw_new[s_i_self]
        arr_qn_new = self.arr_qn_new[s_j_self]
        # select cells above and under the threshold
        slice_under_w = (hfw <= self.hf_min)
        slice_over_w = (hfw > self.hf_min)
        slice_under_n = (hfn <= self.hf_min)
        slice_over_n = (hfn > self.hf_min)

        # calculate flow
        arr_qw_new[slice_under_w] = 0.
        arr_qw_new[slice_over_w] = self.bates2010_ne(self.dx, self.dy,
                wse_i[slice_over_w], wse_i_up[slice_over_w],
                hfw[slice_over_w], qw[slice_over_w], n_i[slice_over_w])
        arr_qn_new[slice_under_n] = 0.
        arr_qn_new[slice_over_n] = self.bates2010_ne(self.dy, self.dx,
                wse_j[slice_over_n], wse_j_up[slice_over_n],
                hfn[slice_over_n], qn[slice_over_n], n_j[slice_over_n])

        # Almeida 2012
        #~ get_q = np.vectorize(self.almeida2012, otypes=[self.arr_qw.dtype])
        #~ self.arr_qw_new[s_i_self] = get_q(self.dx, self.dy, wse_i, wse_i_up,
                                         #~ hfw, qw, qwm1, qwp1, n_i)
        #~ self.arr_qn_new[s_j_self] = get_q(self.dy, self.dx, wse_j, wse_j_up,
                                         #~ hfn, qn, qnm1, qnp1, n_j)
        return self

    def bates2010(self, length, width, wse_0, wse_up, hf, q0, n):
        '''flow formula from
        Bates, P. D., Horritt, M. S., & Fewtrell, T. J. (2010).
        A simple inertial formulation of the shallow water equations for
        efficient two-dimensional flood inundation modelling.
        Journal of Hydrology, 387(1), 33–45.
        http://doi.org/10.1016/j.jhydrol.2010.03.027
        '''
        slope = (wse_0 - wse_up) / length
        # from article
        num = q0 - self.g * hf * self.dt * slope
        den = 1 + self.g * hf * self.dt * n*n * abs(q0) / np.power(hf, 10./3.)
        # from lisflood code
        #~ den = (1 + self.dt * self.g * n*n * abs(q0) / (pow(hf, 4./3) * hf))
        # similar to Almeida and Bates 2013
        #~ den = 1 + self.g * self.dt * n*n * abs(q0) / np.power(hf, 7./3.)
        return (num / den) * width

    def bates2010_ne(self, length, width, wse_0, wse_up, hf, q0, n):
        '''flow formula from
        Bates, P. D., Horritt, M. S., & Fewtrell, T. J. (2010).
        A simple inertial formulation of the shallow water equations for
        efficient two-dimensional flood inundation modelling.
        Journal of Hydrology, 387(1), 33–45.
        http://doi.org/10.1016/j.jhydrol.2010.03.027
        '''
        g = self.g
        dt = self.dt
        return ne.evaluate("((q0 - g * hf * dt * ((wse_0 - wse_up) / length)) / (1 + g * hf * dt * n*n * abs(q0) / (hf**(10./3.)))) * width")

    def almeida2012(self, length, width, wse0, wsem1, hf, q0, qm1, qp1, n):
        '''Flow formula from Almeida et al. 2012
        Without vector norm which is from Almeida and Bates 2013
        '''
        if hf <= self.hf_min:
            return 0
        else:
            # flow formula (formula #41 in almeida et al 2012)
            term_1 = (self.theta * q0 + ((1 - self.theta) / 2) * (qm1 + qp1))
            term_2 = (self.g * hf * (self.dt / length) * (wse0 - wsem1))
            term_3 = (1 + self.g * self.dt * (n*n) * abs(q0) / pow(hf, 7./3.))
            q0_new = (term_1 - term_2) / term_3
            return q0_new * width
