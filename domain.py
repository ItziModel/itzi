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
    """Provides the step() and set_input_arrays() methods
    """

    def __init__(self, dx=None, dy=None, sim_clock=0,
                dtmax=10,
                a=0.7,         # CFL constant
                g=9.80665,     # Standard gravity
                theta=0.9,     # default proposed by Almeida et al.(2012)
                hf_min=0.0001,
                hmin=0.01,
                v_routing=0.1):  # simple routing velocity m/s):
        self.sim_clock = sim_clock
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
        Input arrays are should be set beforehand using set_input_arrays()
        """
        self.set_dt(next_ts)
        self.apply_boundary_conditions()
        self.solve_h()
        self.solve_hflow()
        self.solve_q()
        return self

    def set_input_arrays(self, arrays):
        """Set the input arrays. To be called before each step()
        arrays: a dict of arrays
        """
        # input arrays
        self.arr_z = arrays['z']
        self.arr_n = arrays['n']
        self.arr_ext = arrays['ext']
        self.arr_h_old = arrays['h_hold']
        self.arr_h_new = arrays['h_new']
        self.arr_bctype = arrays['bctype']
        self.arr_bcval = arrays['bcval']
        # 'calculated' arrays
        self.arr_hfw = arrays['hfw']
        self.arr_hfn = arrays['hfn']
        self.arr_qw_new = arrays['qw_new']
        self.arr_qn_new = arrays['qn_new']
        self.arr_qw, self.arrp_qw = self.pad_array(arrays['qw_old'])
        self.arr_qn, self.arrp_qn = self.pad_array(arrays['qn_old'])
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
        arr_q_sum = (self.arr_qw - self.arrp_qw[self.ss, self.sd]
                    + self.arr_qn - self.arrp_qn[self.sd, self.ss])

        self.arr_h_new[:] = ((self.arr_h_old +
                            (self.arr_ext * self.dt)) +
                            arr_q_sum / self.cell_surf * self.dt)
        # set to zero if negative depth
        self.arr_h_new[:] = np.where(self.arr_h_new < 0, 0,
                                    self.arr_h_new[:])
        return self

    def solve_hflow(self):
        """calculate the difference between
        the highest water free surface
        and the highest terrain elevevation of the two adjacent cells
        This acts as an approximation of the hydraulic radius
        """
        # definitions of slices on non-padded arrays
        # don't compute first row or col: solved by boundary conditions
        s_i_self = (slice(None), slice(1, None))
        s_i_up = (slice(None), slice(None, -1))
        s_j_self = (slice(1, None), slice(None))
        s_j_up = (slice(None, -1), slice(None))

        z_i = self.arr_z[s_i_self]
        z_i_up = self.arr_z[s_i_up]
        z_j = self.arr_z[s_j_self]
        z_j_up = self.arr_z[s_j_up]
        wse_i = self.arr_z[s_i_self] + self.arr_h_new[s_i_self]
        wse_i_up = self.arr_z[s_i_up] + self.arr_h_new[s_i_up]
        wse_j = self.arr_z[s_j_self] + self.arr_h_new[s_j_self]
        wse_j_up = self.arr_z[s_j_up] + self.arr_h_new[s_j_up]

        self.arr_hfw[:, self.sc] = (np.maximum(wse_i_up, wse_i)
                                     - np.maximum(z_i_up, z_i))
        self.arr_hfn[self.sc, :] = (np.maximum(wse_j_up, wse_j)
                                       - np.maximum(z_j_up, z_j))
        return self

    def solve_slope(self, wse_0, wse_up, length):
        return (wse_0 - wse_up) / length

    def bates2010(self, slope, width, hf, q0, n):
        '''flow formula from
        Bates, P. D., Horritt, M. S., & Fewtrell, T. J. (2010).
        A simple inertial formulation of the shallow water equations for
        efficient two-dimensional flood inundation modelling.
        Journal of Hydrology, 387(1), 33–45.
        http://doi.org/10.1016/j.jhydrol.2010.03.027
        '''
        num = (q0 - self.g * hf * self.dt * slope)
        den = (1 + self.dt * self.g * n*n * abs(q0) / (pow(hf, 4/3) * hf))
        return (num / den) * width

    def solve_q(self):
        '''
        '''

        get_q = np.vectorize(bates2010)

        return self

    def apply_boundary_conditions(self):
        '''Select relevant 1D slices and apply boundary conditions.
        1D arrays passed to the boundary method include cells bordering
        the boundary on the inside of the domain.
        For the values applying at cells interface (flow depth and flow):
        'qboundary' is the flow at the very boundary
        'hflow' and 'qin' are the next value inside the domain
        Therefore, only 'qboundary' should necessitate a padded array.
        '''
        w_boundary = Boundary(self.dy, self.dx, boundary_pos='W')
        e_boundary = Boundary(self.dy, self.dx, boundary_pos='E')
        n_boundary = Boundary(self.dx, self.dy, boundary_pos='N')
        s_boundary = Boundary(self.dx, self.dy, boundary_pos='S')

        w_boundary.get_boundary_flow(qin=self.arr_qw[:, 1],
                                    qboundary=self.arr_qw[:, 0],
                                    hflow=self.arr_hfw[:, 1],
                                    n=self.arr_n[:, 0],
                                    z=self.arr_z[:, 0],
                                    depth=self.arr_h[:, 0],
                                    bctype=self.arr_bctype[:, 0],
                                    bcvalue=self.arr_bcval[:, 0])
        e_boundary.get_boundary_flow(qin=self.arr_qw[:, -1],
                                    qboundary=self.arrp_qw[1:-1, -1],
                                    hflow=self.arr_hfw[:, -1],
                                    n=self.arr_n[:, -1],
                                    z=self.arr_z[:, -1],
                                    depth=self.arr_h[:, -1],
                                    bctype=self.arr_bctype[:, -1],
                                    bcvalue=self.arr_bcval[:, -1])
        n_boundary.get_boundary_flow(qin=self.arr_qn[1],
                                    qboundary=self.arr_qn[0],
                                    hflow=self.arr_hfn[1],
                                    n=self.arr_n[0],
                                    z=self.arr_z[0],
                                    depth=self.arr_h[0],
                                    bctype=self.arr_bctype[0],
                                    bcvalue=self.arr_bcval[0])
        s_boundary.get_boundary_flow(qin=self.arr_qn[-1, :],
                                    qboundary=self.arrp_qn[-1, 1:-1],
                                    hflow=self.arr_hfn[-1],
                                    n=self.arr_n[-1],
                                    z=self.arr_z[-1],
                                    depth=self.arr_h[-1],
                                    bctype=self.arr_bctype[-1],
                                    bcvalue=self.arr_bcval[-1])
        return self
