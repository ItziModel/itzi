#! /usr/bin/python
# coding=utf8

"""
COPYRIGHT:    (C) 2015 by Laurent Courty

               This program is free software under the GNU General Public
               License (v3). Read the LICENCE file for details.
"""

import math
import numpy as np
import hydro_cython

import hydro_cython
import utils

#~ from grass.pygrass.gis.region import Region

class RasterDomain(object):
    """A raster computational domain made of:
    Various Numpy arrays:
    arr_z: terrain elevation in m
    arr_n: Manning's friction coefficient
    arr_ext: external flows (rain, inflitration, user etc.) in m/s
    arr_bc: structured array: 't': boundary type, 'v' boundary value
    arr_h: water depth

    region: an instance of pygrass Region
    dtmax: maximum time step in seconds
    a: constant for CFL defined by de Almeida et al.(2012) (default: 0.7)
        control the calculated time-step length
    g: standard gravity. m.s^-2
    theta: weighting factor. Control the amount of diffusion in flow eq.
    hf_min: minimum flow depth - used as a "float zero"
    hmin: minimum water depth at cell for wetting / drying process
    """


    def __init__(
                self,
                arr_z = None,
                arr_n = None,
                arr_rain = None,
                arr_evap = None,
                arr_inf = None,
                arr_user = None,
                arr_bc = None,
                arr_h = None,
                region = None,
                dtmax = 5,
                a = 0.7,        # CFL constant
                g = 9.80665,
                theta = 0.7,  # 0.9: default proposed by Almeida et al.(2012)
                hf_min = 0.001,
                hmin = 0.01,
                v_routing=0.1):  # simple routing velocity m/s

        #####################
        # Process arguments #
        #####################

        # general variables
        self.dtmax = dtmax
        self.a = a
        self.g = g
        self.theta = theta
        self.hf_min = hf_min
        self.hmin = hmin
        self.v_routing = v_routing

        # region dimensions
        self.xr = region.cols
        self.yr = region.rows
    
        # cell size in m
        self.dx = region.ewres
        self.dy = region.nsres

        # cell surface
        self.cell_surf = region.ewres * region.nsres

        # Input arrays
        self.set_arr_z(arr_z)
        self.set_arr_n(arr_n)
        self.set_arr_ext(arr_rain, arr_evap, arr_inf, arr_user)
        self.set_arr_bc(arr_bc)
        self.set_arr_h(arr_h)

        ##########################
        # Do some data crunching #
        ##########################

        # create calculated arrays
        self.type_faces = np.dtype([('W', np.float32), ('S', np.float32)])
        # flow at time n
        self.arr_q = np.zeros(shape = (self.yr, self.xr), dtype = self.type_faces)
        # flow depth
        self.arr_hf = np.zeros(shape = (self.yr, self.xr), dtype = self.type_faces)
        # depth at time n+1
        self.arr_h_np1 = np.copy(self.arr_h)
        # flow at time n+1
        self.arr_q_np1 = np.copy(self.arr_q)

        # value needed for flow calculation
        self.arr_q_vecnorm = np.zeros(shape = (self.yr,self.xr), dtype = np.float64)
        self.arr_q_im12_j_y = np.zeros(shape = (self.yr,self.xr), dtype = np.float64)
        self.arr_q_i_jm12_x = np.zeros(shape = (self.yr,self.xr), dtype = np.float64)

        # pad arrays
        #~ self.arr_h, self.arrp_h = utils.pad_array(self.arr_h)
        self.arr_q, self.arrp_q = utils.pad_array(self.arr_q)
        self.arr_hf, self.arrp_hf = utils.pad_array(self.arr_hf)
        self.arr_h_np1, self.arrp_h_np1 = utils.pad_array(self.arr_h_np1)
        self.arr_q_np1, self.arrp_q_np1 = utils.pad_array(self.arr_q_np1)

        # generate flow direction map
        self.flow_dir()

        ##########################
        # Some general variables #
        ##########################

        self.grid_volume = 0.

    def set_arr_z(self, arr_z):
        """Set the DEM array and pad it
        """
        self.arr_z = arr_z
        self.arr_z, self.arrp_z = utils.pad_array(self.arr_z)        
        return self


    def set_arr_n(self, arr_n):
        """Set the Manning's n value array and pad it
        """
        self.arr_n = arr_n
        self.arr_n, self.arrp_n = utils.pad_array(self.arr_n)
        return self


    def set_arr_h(self, arr_h):
        """Set the water depth array and pad it
        """
        if arr_h == None:
            self.arr_h = np.zeros(shape = (self.yr,self.xr), dtype = np.float32)
        else:
            self.arr_h = arr_h
        self.arr_h, self.arrp_h = utils.pad_array(self.arr_h)
        return self


    def set_arr_ext(self, arr_rain, arr_evap, arr_inf, arr_user):
        """Set the external flow array
        arr_user: Numpy array in m/s
        arr_rain, arr_evap, arr_inf: Numpy arrays in mm/h

        arr_ext: Numpy array in m/s
        """
        # set arrays to zero if not given
        if arr_rain == None:
            arr_rain = np.zeros(shape = (self.yr,self.xr), dtype = np.float16)
        if arr_evap == None:
            arr_evap = np.zeros(shape = (self.yr,self.xr), dtype = np.float16)
        if arr_inf == None:
            arr_inf = np.zeros(shape = (self.yr,self.xr), dtype = np.float16)
        if arr_user == None:
            arr_user = np.zeros(shape = (self.yr,self.xr), dtype = np.float32)

        # calculate the resulting array
        self.arr_ext = arr_user + (arr_rain - arr_evap - arr_inf) / 1000 / 3600
        return self


    def set_arr_bc(self, arr_bc):
        """Set boundary condition map and process the boundaries
        input: a structured Numpy array
        """
        self.arr_bc = arr_bc
        self.process_boundaries()
        return self


    def solve_dt(self):
        """Calculate the adaptative time-step according
        to the formula #15 in almeida et al (2012)
        dtmax: maximum time-step
        """
        # calculate the maximum water depth in the domain
        hmax = np.amax(self.arr_h)
        
        if hmax > 0:
            # formula #15 in almeida et al (2012)
            self.dt = min(self.a
                            * (min(self.dx, self.dy)
                            / (math.sqrt(self.g * hmax))), self.dtmax)
        else:
            self.dt = self.dtmax

        return self
        

    def process_boundaries(self):
        """keep only 1D array of each boudary, inside the domain
        create dict_bc, a dictionnary that hold the useful values
        """
        W_BCi = np.copy(self.arr_bc[:,0])
        E_BCi = np.copy(self.arr_bc[:,-1])
        N_BCi = np.copy(self.arr_bc[0,:])
        S_BCi = np.copy(self.arr_bc[-1,:])
        # delete the original array
        del self.arr_bc
        # put boundaries in a dict
        self.dict_bc =  {'W': W_BCi, 'E': E_BCi, 'N': N_BCi, 'S': S_BCi}
        return self


    def solve_h(self):
        """Calculate new water depth
        """
        arr_flow_sum = (self.arr_q['W'] - self.arrp_q[1:-1, 2:]['W']
                    + self.arr_q['S'] - self.arrp_q[:-2, 1:-1]['S'])

        self.arr_h_np1[:] = (self.arr_h + (self.arr_ext * self.dt)) + arr_flow_sum / self.cell_surf * self.dt
        # set to zero if negative depth
        self.arr_h_np1[:] = np.where(self.arr_h_np1 < 0, 0, self.arr_h_np1[:])

        return self


    def solve_hflow(self):
        """calculate the difference between
        the highest water free surface
        and the highest terrain elevevation
        of the two adjacent cells
        This acts as an approximation of the hydraulic radius
        """
        wse_im1 = self.arrp_z[1:-1, :-1] + self.arrp_h[1:-1, :-1]
        wse_i = self.arrp_z[1:-1, 1:] + self.arrp_h[1:-1, 1:]
        z_im1 = self.arrp_z[1:-1, :-1]
        z_i = self.arrp_z[1:-1, 1:]

        wse_jm1 = self.arrp_z[1:, 1:-1] + self.arrp_h[1:, 1:-1]
        wse_j = self.arrp_z[:-1, 1:-1] + self.arrp_h[:-1, 1:-1]
        z_jm1 = self.arrp_z[1:, 1:-1]
        z_j = self.arrp_z[:-1, 1:-1]

        self.arrp_hf[1:-1, 1:]['W'] = (np.maximum(wse_im1, wse_i)
                                     - np.maximum(z_im1, z_i))
        self.arrp_hf[0:-1, 1:-1]['S'] = (np.maximum(wse_jm1, wse_j)
                                       - np.maximum(z_jm1, z_j))
        return self


    def solve_q_vecnorm(self):
        """Calculate the q vector norm to be used in the flow equation
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

        self.arr_q_vecnorm = np.sqrt(np.square(arr_q_ip12_j_y) + np.square(arr_q_i_jp12_x))

        return self


    def solve_q_vecnorm2(self):
        """Calculate the q vector norm to be used in the flow equation
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

        self.arr_q_vecnorm[:] = np.sqrt(np.square(self.arr_q_im12_j_y) + np.square(self.arr_q_i_jm12_x))

        return self


    def solve_q(self):
        '''Solve the general flow in the domain
        '''
        # solve vector norm for all the domain
        self.solve_q_vecnorm2()
        # call a cython function for flow calculation
        self.arr_q_np1['W'], self.arr_q_np1['S'] = hydro_cython.get_flow(
            self.arrp_z,
            self.arrp_n,
            self.arr_h,
            self.arr_hf['W'],
            self.arr_hf['S'],
            self.arrp_q['W'],
            self.arrp_q['S'],
            self.arrp_h_np1,
            self.arr_q_np1['W'],
            self.arr_q_np1['S'],
            self.arr_q_vecnorm,
            self.hmin,
            self.hf_min,
            self.dt, self.dx, self.dy,
            self.g, self.theta)
        return self


    def update_input_values(self):
        """update the arrays of flows and depth
        to be used as entry at next time-step
        """
        self.arr_q[:] = self.arr_q_np1
        self.arr_h[:] = self.arr_h_np1
        return self


    def solve_gridvolume(self):
        """calculatre the total grid volume
        """
        self.grid_volume = np.sum(self.arr_h) * self.cell_surf
        return self


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
        self.arr_slope = np.zeros(shape = Z.shape, dtype = np.string_)

        for c, min_hdiff in np.ndenumerate(arr_min):

            if min_hdiff <= 0:
                self.arr_slope[c] = '0'
            elif min_hdiff == dN[c]:
                self.arr_slope[c] = 'N'
            elif min_hdiff == dS[c]:
                self.arr_slope[c] = 'S'
            elif min_hdiff == dE[c]:
                self.arr_slope[c] = 'E'
            elif min_hdiff == dW[c]:
                self.arr_slope[c] = 'W'

        return self


    def simple_routing(self):
        '''calculate a flow with a given velocity'''
        # water surface elevation
        arrp_wse_np1 = self.arrp_z + self.arrp_h_np1

        # arrays of wse
        arr_wse_w = arrp_wse_np1[1:-1, :-2]
        arr_wse_s = arrp_wse_np1[2:, 1:-1]
        arr_wse = arrp_wse_np1[1:-1, 1:-1]

        # max routed depth
        arr_h_w = np.minimum(self.arr_h_np1, np.maximum(0, arr_wse - arr_wse_w))
        arr_h_s = np.minimum(self.arr_h_np1, np.maximum(0, arr_wse - arr_wse_s))

        arr_q_w = self.arrp_q[1:-1, 1:-1]['W']
        arr_q_s = self.arrp_q[1:-1, 1:-1]['S']

        # arrays of routing flow
        arr_sflow_w = self.dy * arr_h_w * self.v_routing
        arr_sflow_s = self.dy * arr_h_s * self.v_routing

        # can route
        idx_can_route_s = np.where(np.logical_and(
                                    self.arr_h_np1 < self.hmin,
                                    self.arr_slope == 'S'))
        idx_can_route_w = np.where(np.logical_and(
                                    self.arr_h_np1 < self.hmin,
                                    self.arr_slope == 'W'))

        # S
        arr_q_s[idx_can_route_s] = - arr_sflow_s[idx_can_route_s]
        # W
        arr_q_w[idx_can_route_w] = - arr_sflow_w[idx_can_route_w]

        return self


    def drying(self):
        '''Check low-level cells and apply drying (Bradbrook 2006)
        '''
        arr_flow_sum = (self.arr_q['W'] - self.arrp_q[1:-1, 2:]['W']
                      + self.arr_q['S'] - self.arrp_q[:-2, 1:-1]['S'])

        #~ for coor, h, in self.arr_h_np1

        # select negative depth cell
        drying_cells = np.where(np.logical_and(self.arr_h_np1 < 0, arr_flow_sum < 0))
        #~ drying_cells = np.where(self.arr_h_np1 < 0)
        # drying parameter
        arr_dp = - (self.cell_surf * (self.arr_h[drying_cells])) / (self.dt * (arr_flow_sum[drying_cells] + self.arr_ext[drying_cells]))
        #~ print 'arr_dp', arr_dp
        # Apply drying parameter to drying cells
        self.arr_q['W'][drying_cells] *= arr_dp
        self.arrp_q[1:-1, 2:]['W'][drying_cells] *= arr_dp
        self.arr_q['S'][drying_cells] *= arr_dp
        self.arrp_q[:-2, 1:-1]['S'][drying_cells] *= arr_dp

        return 0
