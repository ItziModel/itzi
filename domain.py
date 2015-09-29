#! /usr/bin/python
# coding=utf8

"""
COPYRIGHT:    (C) 2015 by Laurent Courty

              This program is free software under the GNU General Public
              License (version 2). Read the LICENSE file for details.
"""

import math
import numpy as np

class SurfaceDomain(object):
    """
    """

    def __init__(self,
                dx = None,
                dy = None,
                dtmax = 10,
                a = 0.7,         # CFL constant
                g = 9.80665,     # Standard gravity
                theta = 0.9,     # default proposed by Almeida et al.(2012)
                hf_min = 0.0001,
                hmin = 0.01,
                v_routing=0.1):  # simple routing velocity m/s):

        self.sim_clock = 0
        self.cfl = a
        self.g = g
        self.theta = theta
        self.hf_min = hf_min

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
        """
        """
        self.set_dt(next_ts)

        return self.sim_clock

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

    def gms(self, slope, hf, n, width):
        '''Gauckler-Manning-Strickler flow formula
        '''
        V = (1 / n) * pow(hf, 2/3) * pow(slope, 1/2)
        return V * hf * width

    def solve_q(self):
        '''
        '''

        get_q = np.vectorize(bates2010)

        return self
