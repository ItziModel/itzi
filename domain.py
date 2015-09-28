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
                arr_z = None,
                arr_n = None,
                arr_bc = None,
                arr_h = None,
                dx = None,
                dy = None,
                start_time = 0,
                end_time = None,
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
        #~ self.arr_h_old, self.arrp_h_old = self.pad_array(arr_h)

    @staticmethod
    def pad_array(arr):
        """Return the original input array
        as a slice of a larger padded array with one cell
        """
        arr_p = np.pad(arr, 1, 'edge')
        arr = arr_p[1:-1,1:-1]
        return arr, arr_p

    def step(self, forced_ts):
        """
        """
        self.forced_timestep = min(self.end_time, next_ts)
        self.set_dt()

        return self

    def set_dt(self):
        """Calculate the adaptative time-step
        The formula #15 in almeida et al (2012) has been modified to
        accomodate non-square cells
        The time-step is limited by the maximum time-step dtmax.
        """
        hmax = np.amax(self.arr_h_old)
        if hmax > 0:
            self.dt = min(self.dtmax, self.cfl * (min(self.dx, self.dy) /
                                     (math.sqrt(self.g * hmax))))
        else:
            self.dt = self.dtmax

        # set sim_clock and check if timestep is within forced_timestep
        if self.sim_clock + self.dt > self.forced_timestep:
            self.dt = self.forced_timestep - self.sim_clock
            self.sim_clock += self.dt
        else:
            self.sim_clock += self.dt
        return self



class Domain(SurfaceDomain):
    """
    """
    def __init__(self,
                arr_rain = None,
                arr_evap = None,
                arr_inf = None,
                arr_user = None,):
        #~ self.arr_ext = arr_user + arr_rain - arr_evap - arr_inf
        #~ del arr_user, arr_rain, arr_evap, arr_inf
        #~ dtype = np.float32
        #~ self.arr_qw = np.zeros(shape = (self.yr, self.xr),
                                #~ dtype = dtype)
        # Slices for upstream and downstream cells on a padded array
        self.su = slice(None, 2)
        self.sd = slice(2, None)
        # uniform crop
        self.ss = slice(1, -1)
        # slice to crop first row or column
        # to not conflict with boundary condition
        self.sc = slice(1, None)

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
        # whole domain
        wse_i_up = (self.arrp_z[self.ss, self.su] +
                        self.arrp_h_new[self.ss, self.su])
        z_i_up = self.arrp_z[self.ss, self.su]
        wse_j_up = (self.arrp_z[self.su, self.ss] +
                    self.arrp_h_new[self.su, self.ss])
        z_j_up = self.arrp_z[self.su, self.ss]
        wse = self.arr_z + self.arr_h_new
        # don't compute first row or col: solved by boundary conditions
        z_i = self.arr_z[:, self.sc]
        z_i_up = z_i_up[:, self.sc]
        z_j = self.arr_z[self.sc, :]
        z_j_up = z_j_up[self.sc, :]
        wse_i = wse[:, self.sc]
        wse_i_up = wse_i_up[:, self.sc]
        wse_j = wse[self.sc, :]
        wse_j_up = wse_j_up[self.sc, :]

        self.arr_hfw[:, self.sc] = (np.maximum(wse_i_up, wse_i)
                                     - np.maximum(z_i_up, z_i))
        self.arr_hfn[self.sc, :] = (np.maximum(wse_j_up, wse_j)
                                       - np.maximum(z_j_up, z_j))
        return self

    def bates2010(self):
        slope = (wse_0 - wse_m1) / cell_length
        num = (q0 - self.g * hf * self.dt * slope)
        den = (1 + self.dt * self.g * n*n * abs(q0) / (pow(hf, 4/3) * hf))

        return (num / den) * cell_width
