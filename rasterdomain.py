# coding=utf8
"""
Copyright (C) 2016  Laurent Courty

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
from datetime import datetime, timedelta
import numpy as np
try:
    import bottleneck as bn
except ImportError:
    bn = np

import gis
import flow


class TimedArray(object):
    """A container for np.ndarray with time informations.
    Update the array value according to the simulation time.
    array is accessed via get_array()
    """
    def __init__(self, mkey, igis, f_arr_def):
        assert isinstance(mkey, basestring), "not a string!"
        assert hasattr(f_arr_def, '__call__'), "not a function!"
        self.mkey = mkey
        self.igis = igis  # GIS interface
        # A function to generate a default array
        self.f_arr_def = f_arr_def
        # default values for start and end
        # intended to trigger update when is_valid() is first called
        self.a_start = datetime(1, 1, 2)
        self.a_end = datetime(1, 1, 1)

    def get(self, sim_time):
        """Return a numpy array valid for the given time
        If the array stored is not valid, update the values of the object
        """
        assert isinstance(sim_time, datetime), "not a datetime object!"
        if not self.is_valid(sim_time):
            self.update_values_from_gis(sim_time)
        return self.arr

    def is_valid(self, sim_time):
        """input being a time in timedelta
        If the current stored array is within the range of the map,
        return True
        If not return False
        """
        if self.a_start <= sim_time <= self.a_end:
            return True
        else:
            return False

    def update_values_from_gis(self, sim_time):
        """Update array, start_time and end_time from GIS
        if GIS return None, set array to default value
        """
        # Retrieve values
        arr, arr_start, arr_end = self.igis.get_array(self.mkey, sim_time)
        # set to default if no array retrieved
        if not isinstance(arr, np.ndarray):
            arr = self.f_arr_def()
        # check retrieved values
        assert isinstance(arr_start, datetime), "not a datetime object!"
        assert isinstance(arr_end, datetime), "not a datetime object!"
        assert arr_start <= sim_time <= arr_end, "wrong time retrieved!"
        # update object values
        self.a_start = arr_start
        self.a_end = arr_end
        self.arr = arr
        return self


class RasterDomain(object):
    """Group all rasters for the raster domain.
    Store them as np.ndarray with validity information (TimedArray)
    Include tools to update arrays from and write results to GIS,
    including management of the masking and unmasking of arrays.
    """
    def __init__(self, dtype, igis, input_maps, output_maps):
        self.dtype = dtype
        # geographical data
        self.gis = igis
        self.shape = (self.gis.yr, self.gis.xr)
        self.dx = self.gis.dx
        self.dy = self.gis.dy

        # slice for a simple padding (allow stencil calculation on boundary)
        self.simple_pad = (slice(1,-1), slice(1,-1))

        # input and output map names (GIS names)
        self.in_map_names = input_maps
        self.out_map_names = output_maps
        # correspondance between input map names and the arrays
        self.in_k_corresp = {'z': 'in_z', 'n': 'in_n', 'h': 'in_h',
                             'y': 'in_y',
                             'por': 'in_eff_por', 'pres': 'in_cap_pressure',
                             'con': 'in_hyd_conduct', 'in_inf': 'in_inf',
                             'rain': 'in_rain', 'in_q': 'in_q',
                             'bcv': 'in_bcval', 'bct': 'in_bctype'}
        # all keys that will be used for the arrays
        self.k_input = self.in_k_corresp.keys()
        self.k_internal = ['inf', 'hmax', 'ext', 'y', 'hfe', 'hfs', 'qe',
                           'qs', 'qe_new', 'qs_new', 've', 'vs', 've_max',
                           'vs_max', 'bvol', 'q_drain', 'dire', 'dirs']
        self.k_all = self.k_input + self.k_internal
        # maps used to calculate external value
        self.k_ext = ['rain', 'inf', 'q_drain', 'in_q']

        # dictionnary of unmasked input tarrays
        self.tarr = dict.fromkeys(self.k_input)

        # boolean dict that indicate if an array has been updated
        self.isnew = dict.fromkeys(self.k_all, True)

        # create an array mask
        self.mask = np.full(shape=self.shape,
                            fill_value=False, dtype=np.bool_)

        # Instantiate arrays and padded arrays filled with zeros
        self.arr = dict.fromkeys(self.k_all)
        self.arrp = dict.fromkeys(self.k_all)
        self.create_arrays()

        # Instantiate TimedArrays
        self.create_timed_arrays()

    def zeros_array(self):
        """return a np array of the domain dimension, filled with zeros
        dtype is set to object's dtype.
        Intended to be used as default for most of the input model maps
        """
        return np.zeros(shape=self.shape, dtype=self.dtype)

    def pad_array(self, arr):
        """Return the original input array
        as a slice of a larger padded array with one cell
        """
        arr_p = np.pad(arr, 1, 'edge')
        arr = arr_p[self.simple_pad]
        return arr, arr_p

    #~ def ones_array(self):
        #~ """return a np array of the domain dimension, filled with ones
        #~ dtype is set to unsigned integer.
        #~ Intended to be used as default for bctype map
        #~ """
        #~ return np.ones(shape=self.shape, dtype=self.dtype)

    def create_timed_arrays(self):
        """Create TimedArray objects and store them in the input dict
        """
        for k, arr in self.tarr.iteritems():
            self.tarr[k] = TimedArray(self.in_k_corresp[k],
                                      self.gis, self.zeros_array)
        return self

    def create_arrays(self):
        """Instantiate masked arrays and padded arrays
        the unpadded arrays are a slice of the padded ones
        """
        for k, arr in self.arr.iteritems():
            self.arr[k], self.arrp[k] = self.pad_array(self.zeros_array())
        return self

    def update_mask(self, arr):
        '''Create a mask array by marking NULL values from arr as True.
        '''
        self.mask[:] = np.isnan(arr)
        return self

    def mask_array(self, arr, default_value):
        '''Replace NULL values in the input array by the default_value
        '''
        mask = np.logical_or(np.isnan(arr), self.mask)
        arr[mask] = default_value
        assert not np.any(np.isnan(arr))
        return arr

    def unmask_array(self, arr):
        '''Replace values in the input array by NULL values from mask
        '''
        unmasked_array = np.copy(arr)
        unmasked_array[self.mask] = np.nan
        return unmasked_array

    def update_input_arrays(self, sim_time):
        """get new array using TimedArray
        Replace the NULL values (mask)
        """
        for k, ta in self.tarr.iteritems():
            if not ta.is_valid(sim_time):
                self.arr[k][:] = ta.get(sim_time)
                self.isnew[k] = True
                if k == 'z':
                    self.update_mask(self.arr[k])
                    default_value = np.finfo(self.dtype).max
                    # must run update_flow_dir() in SuperficialSimulation
                elif k == 'n':
                    default_value = 1
                else:
                    default_value = 0
                # mask arrays
                self.arr[k][:] = self.mask_array(self.arr[k], default_value)
            else:
                self.isnew[k] = False
        return self

    def update_ext_array(self):
        """If one of the 
        Combine rain, infiltration etc. into a unique array 'ext' in m/s
        rainfall and infiltration are considered in mm/h,
        in_q and q_drain in m/s
        send relevant values to MassBal
        """
        mmh_to_ms = 1000. * 3600.
        #~ # mass balance in m3
        #~ if self.massbal:
            #~ surf_dt = self.dx * self.dy * self.dt
            #~ rain_vol = (bn.nansum(in_rain[np.logical_not(self.mask)]) /
                        #~ mmh_to_ms * surf_dt)
            #~ inf_vol = (bn.nansum(in_inf[np.logical_not(self.mask)]) /
                       #~ mmh_to_ms * surf_dt)
            #~ inflow_vol = bn.nansum(in_q[np.logical_not(self.mask)]) * surf_dt
            #~ self.massbal.add_value('rain_vol', rain_vol)
            #~ self.massbal.add_value('inf_vol', inf_vol)
            #~ self.massbal.add_value('inflow_vol', inflow_vol)
        
        if any([self.isnew[k] for k in self.k_ext]):
            flow.set_ext_array(self.arr['in_q'], self.arr['rain'],
                               self.arr['inf'], self.arr['q_drain'],
                               self.arr['ext'], mmh_to_ms)
            self.isnew['ext'] = True
        else:
            self.isnew['ext'] = False
        return self

    def swap_arrays(self, k1, k2):
        """swap values of two arrays
        """
        self.arr[k1], self.arr[k2] = self.arr[k2], self.arr[k1]
        self.arrp[k1], self.arrp[k2] = self.arrp[k2], self.arrp[k1]
        return self

    def get(self, k):
        """return the unpadded, masked array of key 'k'
        """
        return self.arr[k]

    def get_padded(self, k):
        """return the padded, masked array of key 'k'
        """
        return self.arrp[k]

    def get_unmasked(self, k):
        """return unpadded array with NaN
        """
        return self.unmask_array(self.arr[k])

    def get_output_arrays(self):
        """Takes a dict of map names
        return a dict of unmasked arrays
        """
        out_arrays = {}
        if self.out_map_names['out_h'] != None:
            out_arrays['out_h'] = self.get_unmasked('h')
        if self.out_map_names['out_wse'] != None:
            out_arrays['out_wse'] = self.get_unmasked('h') + self.get('z')
        if self.out_map_names['out_vx'] != None:
            arr_vx = np.empty_like(self.get('h'))
            flow.solve_v(self.get_unmasked('qe_new'), self.get('hfe'), arr_vx)
            assert not np.any(np.isnan(arr_vx))
            out_arrays['out_vx'] = arr_vx
        if self.out_map_names['out_vy'] != None:
            arr_vy = np.empty_like(self.get('h'))
            flow.solve_v(self.get_unmasked('qs_new'), self.get('hfs'), arr_vy)
            assert not np.any(np.isnan(arr_vy))
            out_arrays['out_vy'] = arr_vy
        if self.out_map_names['out_qx'] != None:
            out_arrays['out_qx'] = self.get_unmasked('qe_new') * self.dy
        if self.out_map_names['out_qy'] != None:
            out_arrays['out_qy'] = self.get_unmasked('qs_new') * self.dx
        return out_arrays
