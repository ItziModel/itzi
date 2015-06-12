#! /usr/bin/python
# coding=utf8

"""
COPYRIGHT:    (C) 2015 by Laurent Courty

               This program is free software under the GNU General Public
               License (v3). Read the LICENCE file for details.
"""

import math
import numpy as np

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
                dtmax = 10,
                a = 0.7,
                g = 9.80665,
                theta = 0.9,  # default proposed by Almeida et al.(2012)
                hf_min = 0.001,
                hmin = 0.01):  

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

        # pad arrays
        #~ self.arr_h, self.arrp_h = utils.pad_array(self.arr_h)
        self.arr_q, self.arrp_q = utils.pad_array(self.arr_q)
        self.arr_hf, self.arrp_hf = utils.pad_array(self.arr_hf)
        self.arr_h_np1, self.arrp_h_np1 = utils.pad_array(self.arr_h_np1)
        self.arr_q_np1, self.arrp_q_np1 = utils.pad_array(self.arr_q_np1)

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
        """
        Calculate new water depth
        """
        flow_sum = (self.arr_q['W'] - self.arrp_q[1:-1, 2:]['W']
                    + self.arr_q['S'] - self.arrp_q[:-2, 1:-1]['S'])

        self.arr_h_np1[:] = (self.arr_h + (self.arr_ext * self.dt)) + flow_sum / self.cell_surf * self.dt

        return self


    def solve_hflow(self):
        """calculate the difference between
        the highest water free surface and the highest terrain elevevation
        of the two adjacent cells
        This act as an approximation of the hydraulic radius
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
