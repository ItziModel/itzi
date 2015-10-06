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
import math
import numpy as np
from boundary import Boundary

class SurfaceDomain(object):
    """Represents a staggered grid where flow is simulated
    Accessed using the step(), set_input_arrays() 
    and get_output_arrays() methods
    """

    def __init__(self, dx, dy, arr_def, arr_h,
                sim_clock=0,
                dtmax=10,
                a=0.7,         # CFL constant
                g=9.80665,     # Standard gravity
                theta=0.9,     # default proposed by Almeida et al.(2012)
                hf_min=0.0001,
                hmin=0.01,
                v_routing=0.1):  # simple routing velocity m/s):
        self.sim_clock = sim_clock
        self.dtmax = dtmax
        self.cfl = a
        self.g = g
        self.theta = theta
        self.hf_min = hf_min
        self.dx = dx
        self.dy = dy
        self.cell_surf = dx * dy
        # Slices for upstream and downstream cells on a padded array
        self.su = slice(None, 2)
        self.sd = slice(2, None)
        # uniform crop
        self.ss = slice(1, -1)
        # slice to crop first row or column
        # to not conflict with boundary condition
        self.sc = slice(1, None)

        # Input water depth
        self.arr_h_old = arr_h
        # Set internal arrays to a provided default
        # Input arrays are set externally with set_input_arrays()
        self.arr_h_new = np.copy(arr_def)
        self.arr_hfw = np.copy(arr_def)
        self.arr_hfn = np.copy(arr_def)
        #~ self.arr_qw_new = np.copy(arr_def)
        #~ self.arr_qn_new = np.copy(arr_def)
        self.arr_qw, self.arrp_qw = self.pad_array(np.copy(arr_def))
        self.arr_qn, self.arrp_qn = self.pad_array(np.copy(arr_def))
        self.arr_qw_new, self.arrp_qw_new = self.pad_array(np.copy(arr_def))
        self.arr_qn_new, self.arrp_qn_new = self.pad_array(np.copy(arr_def))
        del arr_def

    @staticmethod
    def pad_array(arr):
        """Return the original input array
        as a slice of a larger padded array with one cell
        """
        arr_p = np.pad(arr, 1, 'edge')
        arr = arr_p[1:-1,1:-1]
        return arr, arr_p

    def step(self, next_ts):
        """Run a full simulation time-step
        Input arrays should be set beforehand using set_input_arrays()
        """
        q_old_x_axis = self.arrp_qw[3, 1:]
        q_new_x_axis = self.arr_qw_new[2, :]
        h_old_x_axis = self.arr_h_old[2, :]
        h_new_x_axis = self.arr_h_new[2, :]
        self.set_dt(next_ts)
        self.apply_boundary_conditions()
        self.solve_q()
        self.solve_h()
        self.copy_arrays_values_for_next_timestep()
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

    def solve_h(self):
        """Calculate new water depth
        """
        arr_q_sum = (self.arr_qw_new - self.arrp_qw_new[self.ss, self.sd]
                    + self.arr_qn_new - self.arrp_qn_new[self.sd, self.ss])

        self.arr_h_new[:] = ((self.arr_h_old +
                            (self.arr_ext * self.dt)) +
                            arr_q_sum / self.cell_surf * self.dt)
        # set to zero if negative depth
        self.arr_h_new[:] = np.where(self.arr_h_new < 0, 0,
                                    self.arr_h_new[:])
        return self

    def solve_hflow(self, wse_i_up, wse_i, z_i_up, z_i,
                        wse_j_up, wse_j, z_j_up, z_j):
        """calculate the difference between
        the highest water free surface
        and the highest terrain elevevation of the two adjacent cells
        This acts as an approximation of the hydraulic radius
        """
        self.arr_hfw[:, self.sc] = (np.maximum(wse_i_up, wse_i)
                                     - np.maximum(z_i_up, z_i))
        self.arr_hfn[self.sc, :] = (np.maximum(wse_j_up, wse_j)
                                       - np.maximum(z_j_up, z_j))
        return self

    def bates2010(self, length, width, wse_0, wse_up, hf, q0, n):
        '''flow formula from
        Bates, P. D., Horritt, M. S., & Fewtrell, T. J. (2010).
        A simple inertial formulation of the shallow water equations for
        efficient two-dimensional flood inundation modelling.
        Journal of Hydrology, 387(1), 33–45.
        http://doi.org/10.1016/j.jhydrol.2010.03.027
        '''
        if hf <= self.hf_min:
            return 0
        else:
            slope = (wse_0 - wse_up) / length
            num = (q0 - self.g * hf * self.dt * slope)
            den = (1 + self.dt * self.g * n*n * abs(q0) / (pow(hf, 4/3) * hf))
            return (num / den) * width

    def solve_q(self):
        '''Calculate flow across the whole domain, appart from boundaries
        '''
        # definitions of slices on non-padded arrays
        # don't compute first row or col: solved by boundary conditions
        s_i_self = (slice(None), self.sc)
        s_i_up = (slice(None), slice(None, -1))
        s_j_self = (self.sc, slice(None))
        s_j_up = (slice(None, -1), slice(None))

        z_i = self.arr_z[s_i_self]
        z_i_up = self.arr_z[s_i_up]
        z_j = self.arr_z[s_j_self]
        z_j_up = self.arr_z[s_j_up]
        wse_i = self.arr_z[s_i_self] + self.arr_h_old[s_i_self]
        wse_i_up = self.arr_z[s_i_up] + self.arr_h_old[s_i_up]
        wse_j = self.arr_z[s_j_self] + self.arr_h_old[s_j_self]
        wse_j_up = self.arr_z[s_j_up] + self.arr_h_old[s_j_up]

        # Solve hflow
        self.solve_hflow(wse_i_up, wse_i, z_i_up, z_i,
                        wse_j_up, wse_j, z_j_up, z_j)

        hfw = self.arr_hfw[s_i_self]
        hfn = self.arr_hfn[s_j_self]
        qw = self.arr_qw[s_i_self] / self.dy
        qn = self.arr_qn[s_j_self] / self.dx
        n_i = self.arr_n[s_i_self]
        n_j = self.arr_n[s_j_self]

        get_q = np.vectorize(self.bates2010, otypes=[self.arr_qw.dtype])
        self.arr_qw_new[s_i_self] = get_q(self.dx, self.dy,
                                        wse_i, wse_i_up, hfw, qw, n_i)
        self.arr_qn_new[s_j_self] = get_q(self.dy, self.dx,
                                        wse_j, wse_j_up, hfn, qn, n_j)
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
        w_boundary = Boundary(self.dy, self.dx, boundary_pos='W')
        e_boundary = Boundary(self.dy, self.dx, boundary_pos='E')
        n_boundary = Boundary(self.dx, self.dy, boundary_pos='N')
        s_boundary = Boundary(self.dx, self.dy, boundary_pos='S')

        w_boundary.get_boundary_flow(qin=self.arr_qw[:, 1],
                                    hflow=self.arr_hfw[:, 1],
                                    n=self.arr_n[:, 0],
                                    z=self.arr_z[:, 0],
                                    depth=self.arr_h_old[:, 0],
                                    bctype=self.arr_bctype[:, 0],
                                    bcvalue=self.arr_bcval[:, 0],
                                    qboundary=self.arr_qw_new[:, 0])
        e_boundary.get_boundary_flow(qin=self.arr_qw[:, -1],
                                    hflow=self.arr_hfw[:, -1],
                                    n=self.arr_n[:, -1],
                                    z=self.arr_z[:, -1],
                                    depth=self.arr_h_old[:, -1],
                                    bctype=self.arr_bctype[:, -1],
                                    bcvalue=self.arr_bcval[:, -1],
                                    qboundary=self.arrp_qw_new[1:-1, -1])
        n_boundary.get_boundary_flow(qin=self.arr_qn[1],
                                    hflow=self.arr_hfn[1],
                                    n=self.arr_n[0],
                                    z=self.arr_z[0],
                                    depth=self.arr_h_old[0],
                                    bctype=self.arr_bctype[0],
                                    bcvalue=self.arr_bcval[0],
                                    qboundary=self.arr_qn_new[0])
        s_boundary.get_boundary_flow(qin=self.arr_qn[-1, :],
                                    hflow=self.arr_hfn[-1],
                                    n=self.arr_n[-1],
                                    z=self.arr_z[-1],
                                    depth=self.arr_h_old[-1],
                                    bctype=self.arr_bctype[-1],
                                    bcvalue=self.arr_bcval[-1],
                                    qboundary=self.arrp_qn_new[-1, 1:-1])
        return self

    def copy_arrays_values_for_next_timestep(self):
        """Copy values from calculated arrays to input arrays
        """
        self.arr_qw[:] = self.arr_qw_new
        self.arr_qn[:] = self.arr_qn_new
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

    def set_forced_timestep(self, next_record):
        '''Defines the value of the next forced time step
        Could be the next recording of data or the end of simulation
        '''
        self.forced_timestep = min(self.end_time, next_record)
        return self

    def solve_q_vecnorm(self):
        """Calculate the q vector norm to be used in the flow equation
        This function uses the values in i+1 and j+1,
        as originally explained in Almeida and Bates (2013)
        """
        arr_q_i_jm12 = self.arrp_q['S'][1:-1, 1:-1]
        arr_q_i_jp12 = self.arrp_q['S'][:-2, 1:-1]
        arr_q_ip1_jm12 = self.arrp_q['S'][1:-1, 2:]
        arr_q_ip1_jp12 = self.arrp_q['S'][:-2, 2:]
        arr_q_im12_j = self.arrp_q['W'][1:-1, 1:-1]
        arr_q_ip12_j = self.arrp_q['W'][1:-1, 2:]
        arr_q_im12_jp1 = self.arrp_q['W'][:-2, 1:-1]
        arr_q_ip12_jp1 = self.arrp_q['W'][:-2, 2:]

        arr_q_ip12_j_y = (arr_q_i_jm12 + arr_q_i_jp12 +
                          arr_q_ip1_jm12 + arr_q_ip1_jp12) / 4
        arr_q_i_jp12_x = (arr_q_im12_j + arr_q_ip12_j +
                          arr_q_im12_jp1 + arr_q_ip12_jp1) / 4

        self.arr_q_vecnorm = np.sqrt(
                            np.square(arr_q_ip12_j_y) +
                            np.square(arr_q_i_jp12_x))
        return self

    def solve_q_vecnorm2(self):
        """Calculate the q vector norm to be used in the flow equation
        This function uses values in i-1 and j-1, which seems more logical
        than the version given in Almeida and Bates (2013)
        """
        arr_q_i_jm12 = self.arrp_q['S'][1:-1, 1:-1]
        arr_q_i_jp12 = self.arrp_q['S'][:-2, 1:-1]
        arr_q_im1_jm12 = self.arrp_q['S'][1:-1, :-2]
        arr_q_im1_jp12 = self.arrp_q['S'][:-2, :-2]
        arr_q_im12_j = self.arrp_q['W'][1:-1, 1:-1]
        arr_q_ip12_j = self.arrp_q['W'][1:-1, 2:]
        arr_q_im12_jm1 = self.arrp_q['W'][2:, 1:-1]
        arr_q_ip12_jm1 = self.arrp_q['W'][2:, 2:]

        self.arr_q_im12_j_y[:] = (arr_q_i_jm12 + arr_q_i_jp12 +
                          arr_q_im1_jm12 + arr_q_im1_jp12) / 4
        self.arr_q_i_jm12_x[:] = (arr_q_im12_j + arr_q_ip12_j +
                          arr_q_im12_jm1 + arr_q_ip12_jm1) / 4
        self.arr_q_vecnorm[:] = np.sqrt(
                                    np.square(self.arr_q_im12_j_y) +
                                    np.square(self.arr_q_i_jm12_x))
        return self
