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
try:
    import bottleneck as bn
except ImportError:
    bn = np

import flow
import time
from itzi_error import NullError


class SuperficialSimulation(object):
    """Represents a staggered grid where flow is simulated
    Accessed through step() and get_output_arrays() methods
    By convention the flow is:
     - calculated at the East and South faces of each cell
     - positive from West to East and from North to South
    """

    def __init__(self, domain, param,
                 sim_clock=0,
                 g=9.80665):     # Standard gravity
        self.dom = domain
        self.sim_clock = sim_clock
        self.dtmax = param['dtmax']
        self.cfl = param['cfl']
        self.g = g
        self.theta = param['theta']
        self.hf_min = param['hmin']
        self.sl_thresh = param['slmax']
        self.v_routing = param['vrouting']
        self.dx = domain.dx
        self.dy = domain.dy
        self.cell_surf = self.dx * self.dy

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
        self.w_boundary = Boundary(self.dy, self.dx, boundary_pos='W')
        self.e_boundary = Boundary(self.dy, self.dx, boundary_pos='E')
        self.n_boundary = Boundary(self.dx, self.dy, boundary_pos='N')
        self.s_boundary = Boundary(self.dx, self.dy, boundary_pos='S')

    def update_flow_dir(self):
        ''' Return arrays of flow directions used for rain routing
        each cell is assigned a direction in which it will drain
        0: the flow is going dowstream, index-wise
        1: the flow is going upstream, index-wise
        -1: no routing happening on that face
        '''
        # get a padded array
        arrp_z = self.dom.get_padded('z')
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
        flow.flow_dir(arr_max_dz[self.s_j_0], dN, dS, self.dom.get('dirs'))
        # x direction
        flow.flow_dir(arr_max_dz[self.s_i_0], dW, dE, self.dom.get('dire'))
        return self

    def step(self, next_ts, massbal):
        """Run a full simulation time-step
        """
        start_clock = time.time()
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
        self.arr_err = np.isnan(self.dom.get('h'))
        if np.any(self.arr_err):
            raise NullError

        self.swap_flow_arrays()
        end_clock = time.time()
        if massbal:
            massbal.add_value('step_duration', end_clock - start_clock)
        return self

    def set_dt(self, next_ts):
        """Calculate the adaptative time-step
        The formula #15 in almeida et al (2012) has been modified to
        accomodate non-square cells
        The time-step is limited by the maximum time-step dtmax.
        """
        maxh = self.dom.amax('h')  # max depth in domain
        min_dim = min(self.dx, self.dy)
        if maxh > 0:
            self.dt = min(self.dtmax,
                          self.cfl * (min_dim /
                                      (math.sqrt(self.g * maxh))))
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
        return bn.nansum(self.dom.get('h')) * self.cell_surf

    def update_h(self):
        """Calculate new water depth
        """
        flow_west = self.dom.get_padded('qe_new')[self.ss, self.su]
        flow_east = self.dom.get('qe_new')
        flow_north = self.dom.get_padded('qs_new')[self.su, self.ss]
        flow_south = self.dom.get('qs_new')
        assert (flow_west.shape == flow_east.shape ==
                flow_north.shape == flow_south.shape)

        self.hfix_vol = 0.
        flow.solve_h(arr_ext=self.dom.get('ext'),
                     arr_qe=flow_east, arr_qw=flow_west,
                     arr_qn=flow_north, arr_qs=flow_south,
                     arr_bct=self.dom.get('bct'), arr_bcv=self.dom.get('bcv'),
                     arr_h=self.dom.get('h'), arr_hmax=self.dom.get('hmax'),
                     dx=self.dx, dy=self.dy, dt=self.dt,
                     hfix_vol=self.hfix_vol)
        assert not np.any(self.dom.get('h') < 0)
        return self

    def solve_q(self):
        '''Solve flow inside the domain using C/Cython function
        '''
        flow.solve_q(arr_dire=self.dom.get('dire'), arr_dirs=self.dom.get('dirs'),
                     arr_z=self.dom.get('z'), arr_n=self.dom.get('n'),
                     arr_h=self.dom.get('h'),
                     arrp_qe=self.dom.get_padded('qe'), arrp_qs=self.dom.get_padded('qs'),
                     arr_qe_new=self.dom.get('qe_new'), arr_qs_new=self.dom.get('qs_new'),
                     arr_hfe=self.dom.get('hfe'), arr_hfs=self.dom.get('hfs'),
                     arr_v=self.dom.get('v'), arr_vdir=self.dom.get('vdir'),
                     arr_vmax=self.dom.get('vmax'),
                     dt=self.dt, dx=self.dx, dy=self.dy, g=self.g,
                     theta=self.theta, hf_min=self.hf_min,
                     v_rout=self.v_routing, sl_thres=self.sl_thresh)
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

        w_boundary_flow = self.dom.get_padded('qe_new')[1:-1, 0]
        self.w_boundary.get_boundary_flow(qin=self.dom.get('qe_new')[:, 0],
                                          hflow=self.dom.get('hfe')[:, 0],
                                          n=self.dom.get('n')[:, 0],
                                          z=self.dom.get('z')[:, 0],
                                          depth=self.dom.get('h')[:, 0],
                                          bctype=self.dom.get('bct')[:, 0],
                                          bcvalue=self.dom.get('bcv')[:, 0],
                                          qboundary=w_boundary_flow)
        e_boundary_flow = self.dom.get('qe_new')[:, -1]
        self.e_boundary.get_boundary_flow(qin=self.dom.get('qe_new')[:, -2],
                                          hflow=self.dom.get('hfe')[:, -2],
                                          n=self.dom.get('n')[:, -1],
                                          z=self.dom.get('z')[:, -1],
                                          depth=self.dom.get('h')[:, -1],
                                          bctype=self.dom.get('bct')[:, -1],
                                          bcvalue=self.dom.get('bcv')[:, -1],
                                          qboundary=e_boundary_flow)
        n_boundary_flow = self.dom.get_padded('qs_new')[0, 1:-1]
        self.n_boundary.get_boundary_flow(qin=self.dom.get('qs_new')[0],
                                          hflow=self.dom.get('hfs')[0],
                                          n=self.dom.get('n')[0],
                                          z=self.dom.get('z')[0],
                                          depth=self.dom.get('h')[0],
                                          bctype=self.dom.get('bct')[0],
                                          bcvalue=self.dom.get('bcv')[0],
                                          qboundary=n_boundary_flow)
        s_boundary_flow = self.dom.get('qs_new')[-1]
        self.s_boundary.get_boundary_flow(qin=self.dom.get('qs_new')[-2],
                                          hflow=self.dom.get('hfs')[-2],
                                          n=self.dom.get('n')[-1],
                                          z=self.dom.get('z')[-1],
                                          depth=self.dom.get('h')[-1],
                                          bctype=self.dom.get('bct')[-1],
                                          bcvalue=self.dom.get('bcv')[-1],
                                          qboundary=s_boundary_flow)
        # calculate volume entering through boundaries
        x_boundary_len = (w_boundary_flow.shape[0] +
                          e_boundary_flow.shape[0]) * self.dy
        y_boundary_len = (n_boundary_flow.shape[0] +
                          s_boundary_flow.shape[0]) * self.dx
        x_boundary_flow = (bn.nansum(w_boundary_flow) -
                           bn.nansum(e_boundary_flow)) * x_boundary_len
        y_boundary_flow = (bn.nansum(n_boundary_flow) -
                           bn.nansum(s_boundary_flow)) * y_boundary_len
        boundary_vol = (x_boundary_flow + y_boundary_flow)
        return boundary_vol

    def swap_flow_arrays(self):
        """Swap flow arrays from calculated to input
        """
        self.dom.swap_arrays('qe', 'qe_new')
        self.dom.swap_arrays('qs', 'qs_new')
        return self

    def get_output_arrays(self, out_names):
        """Takes a dict of map names
        return a dict of unmasked arrays
        """
        out_arrays = {}
        if out_names['out_h'] is not None:
            out_arrays['out_h'] = self.dom.get_unmasked('h')
        if out_names['out_wse']  is not None:
            out_arrays['out_wse'] = self.dom.get_unmasked('h') + self.dom.get('z')
        if out_names['out_v']  is not None:
            out_arrays['out_v'] = self.dom.get_unmasked('v')
        if out_names['out_vdir']  is not None:
            out_arrays['out_vdir'] = self.dom.get_unmasked('vdir')
        if out_names['out_qx']  is not None:
            out_arrays['out_qx'] = self.dom.get_unmasked('qe_new') * self.dy
        if out_names['out_qy']  is not None:
            out_arrays['out_qy'] = self.dom.get_unmasked('qs_new') * self.dx
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
        slice_closed = np.where((bctype == 0) | (bctype == 1))
        slice_open = np.where(bctype == 2)
        slice_wse = np.where(bctype == 3)
        # Boundary type 1 (closed)
        qboundary[slice_closed] = 0
        # Boundary type 2 (open)
        qboundary[slice_open] = self.get_flow_open_boundary(qin[slice_open],
                                                            hflow[slice_open],
                                                            depth[slice_open])
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
        result[slice_over] = (qin[slice_over] / hf[slice_over] *
                              hf_boundary[slice_over])
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
