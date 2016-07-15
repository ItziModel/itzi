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
import csv
import copy

import gis
import flow


class TimedArray(object):
    """A container for np.ndarray with time informations.
    Update the array value according to the simulation time.
    array is accessed via get()
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
        # data type
        self.dtype = dtype
        # geographical data
        self.gis = igis
        self.shape = (self.gis.yr, self.gis.xr)
        self.dx = self.gis.dx
        self.dy = self.gis.dy
        self.cell_surf = self.dx * self.dy

        # conversion factor between mm/h and m/s
        self.mmh_to_ms = 1000. * 3600.

        # number of cells in a row must be a multiple of that number
        byte_num = 256 / 8  # AVX2
        itemsize = np.dtype(self.dtype).itemsize
        self.row_mul = int(byte_num / itemsize)

        # slice for a simple padding (allow stencil calculation on boundary)
        self.simple_pad = (slice(1, -1), slice(1, -1))

        # input and output map names (GIS names)
        self.in_map_names = input_maps
        self.out_map_names = output_maps
        # correspondance between input map names and the arrays
        self.in_k_corresp = {'z': 'dem', 'n': 'friction', 'h': 'start_h',
                             'y': 'start_y',
                             'por': 'effective_pororosity',
                             'pres': 'capillary_pressure',
                             'con': 'hydraulic_conductivity',
                             'in_inf': 'infiltration',
                             'rain': 'rain', 'in_q': 'inflow',
                             'bcv': 'bcval', 'bct': 'bctype'}
        # all keys that will be used for the arrays
        self.k_input = self.in_k_corresp.keys()
        self.k_internal = ['inf', 'hmax', 'ext', 'y', 'hfe', 'hfs',
                           'qe', 'qs', 'qe_new', 'qs_new',
                           'ue', 'us', 'v', 'vdir', 'vmax',
                           'q_drain', 'dire', 'dirs']
        # arrays gathering the cumulated water depth from corresponding array
        self.k_stats = ['st_bound', 'st_inf', 'st_rain', 'st_inflow',
                        'st_drain']
        self.stats_corresp = {'inf': 'st_inf', 'rain': 'st_rain',
                              'in_q': 'st_inflow', 'q_drain': 'st_drain'}
        self.k_all = self.k_input + self.k_internal + self.k_stats
        # maps used to calculate external value
        self.k_ext = ['rain', 'inf', 'q_drain', 'in_q']
        # last update of statistical map entry
        self.stats_update_time = dict.fromkeys(self.k_stats)

        # dictionnary of unmasked input tarrays
        self.tarr = dict.fromkeys(self.k_input)

        # boolean dict that indicate if an array has been updated
        self.isnew = dict.fromkeys(self.k_all, True)
        self.isnew['q_drain'] = False

        # create an array mask
        self.mask = np.full(shape=self.shape,
                            fill_value=False, dtype=np.bool_)

        # Instantiate arrays and padded arrays filled with zeros
        self.arr = dict.fromkeys(self.k_all)
        self.arrp = dict.fromkeys(self.k_all)
        self.create_arrays()

        # Instantiate TimedArrays
        self.create_timed_arrays()

    def water_volume(self):
        """get current water volume in the domain"""
        return self.asum('h') * self.cell_surf

    def inf_vol(self, sim_time):
        self.populate_stat_array('inf', sim_time)
        return self.asum('st_inf') * self.cell_surf

    def rain_vol(self, sim_time):
        self.populate_stat_array('rain', sim_time)
        return self.asum('st_rain') * self.cell_surf

    def inflow_vol(self, sim_time):
        self.populate_stat_array('in_q', sim_time)
        return self.asum('st_inflow') * self.cell_surf

    def drain_vol(self, sim_time):
        self.populate_stat_array('drain', sim_time)
        return self.asum('st_drain') * self.cell_surf

    def boundary_vol(self):
        return self.asum('st_bound') * self.cell_surf

    def zeros_array(self):
        """return a np array of the domain dimension, filled with zeros.
        dtype is set to object's dtype.
        Intended to be used as default for the input model maps.
        """
        return np.zeros(shape=self.shape, dtype=self.dtype)

    def pad_array(self, arr):
        """Return the original input array
        as a slice of a larger padded array with one cell
        """
        arr_p = np.pad(arr, 1, 'edge')
        arr = arr_p[self.simple_pad]
        return arr, arr_p

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
        return self

    def unmask_array(self, arr):
        '''Replace values in the input array by NULL values from mask
        '''
        unmasked_array = np.copy(arr)
        unmasked_array[self.mask] = np.nan
        return unmasked_array

    def update_input_arrays(self, sim_time):
        """Get new array using TimedArray
        First update the DEM and mask if needed
        Replace the NULL values (mask)
        """
        # make sure DEM is treated first
        if not self.tarr['z'].is_valid(sim_time):
            self.arr['z'][:] = self.tarr['z'].get(sim_time)
            self.isnew['z'] = True
            # note: must run update_flow_dir() in SuperficialSimulation
            self.update_mask(self.arr['z'])
            self.mask_array(self.arr['z'], np.finfo(self.dtype).max)

        # loop through the arrays
        for k, ta in self.tarr.iteritems():
            if not ta.is_valid(sim_time):
                # z is done before
                if k == 'z':
                    continue
                # calculate statistics before updating array
                if k in ['in_q', 'rain']:
                    self.populate_stat_array(k, sim_time)
                # update array
                self.gis.msgr.verbose(u"{}: update input array <{}>".format(sim_time, k))
                self.arr[k][:] = ta.get(sim_time)
                self.isnew[k] = True
                if k == 'n':
                    fill_value = 1
                else:
                    fill_value = 0
                # mask arrays
                self.mask_array(self.arr[k], fill_value)
            else:
                self.isnew[k] = False
        # calculate water volume at the beginning of the simulation
        if self.isnew['h']:
            self.start_volume = self.asum('h')
        return self

    def populate_stat_array(self, k, sim_time):
        """given an external input array key,
        populate the corresponding statistic array.
        If it's the first update, only check in the time.
        Should be called before updating the array
        """
        sk = self.stats_corresp[k]
        update_time = self.stats_update_time[sk]
        # make sure everything is in m/s
        if k in ['rain', 'inf']:
            conv_factor = 1 / self.mmh_to_ms
        else:
            conv_factor = 1.

        if self.stats_update_time[sk] is None:
                self.stats_update_time[sk] = sim_time
        else:
            self.gis.msgr.verbose(u"{}: Populating array <{}>".format(sim_time, sk))
            time_diff = (sim_time - update_time).total_seconds()
            flow.populate_stat_array(self.arr[k], self.arr[sk],
                                     conv_factor, time_diff)
            self.stats_update_time[sk] = sim_time
        return None

    def update_ext_array(self):
        """If one of the input array has been updated,
        combine rain, infiltration etc. into a unique array 'ext' in m/s.
        Rainfall and infiltration are considered in mm/h,
        in_q and q_drain in m/s
        """
        if any([self.isnew[k] for k in self.k_ext]):
            flow.set_ext_array(self.arr['in_q'], self.arr['rain'],
                               self.arr['inf'], self.arr['q_drain'],
                               self.arr['ext'], self.mmh_to_ms)
            self.isnew['ext'] = True
        else:
            self.isnew['ext'] = False
        return self

    def get_output_arrays(self, interval_s, sim_time):
        """Returns a dict of unmasked arrays to be written to the disk
        """
        out_arrays = {}
        if self.out_map_names['h'] is not None:
            out_arrays['h'] = self.get_unmasked('h')
        if self.out_map_names['wse'] is not None:
            out_arrays['wse'] = self.get_unmasked('h') + self.get('z')
        if self.out_map_names['v'] is not None:
            out_arrays['v'] = self.get_unmasked('v')
        if self.out_map_names['vdir'] is not None:
            out_arrays['vdir'] = self.get_unmasked('vdir')
        if self.out_map_names['qx'] is not None:
            out_arrays['qx'] = self.get_unmasked('qe_new') * self.dy
        if self.out_map_names['qy'] is not None:
            out_arrays['qy'] = self.get_unmasked('qs_new') * self.dx
        # statistics (average of last interval)
        if self.out_map_names['boundaries'] is not None:
            out_arrays['boundaries'] = self.get_unmasked('st_bound') / interval_s
        if self.out_map_names['inflow'] is not None:
            self.populate_stat_array('in_q', sim_time)
            out_arrays['inflow'] = self.get_unmasked('st_inflow') / interval_s
        if self.out_map_names['infiltration'] is not None:
            self.populate_stat_array('inf', sim_time)
            out_arrays['infiltration'] = (self.get_unmasked('st_inf') /
                                          interval_s) * self.mmh_to_ms
        if self.out_map_names['rainfall'] is not None:
            self.populate_stat_array('rain', sim_time)
            out_arrays['rainfall'] = (self.get_unmasked('st_rain') /
                                      interval_s) * self.mmh_to_ms
        return out_arrays

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

    def amax(self, k):
        """return maximum value of an unpadded array
        """
        return np.amax(self.arr[k])

    def asum(self, k):
        """return the sum of an unpadded array
        values outside the proper domain are the defaults values
        """
        return flow.arr_sum(self.arr[k])

    def reset_stats(self, sim_time):
        """Set stats arrays to zeros and the update time to current time
        """
        for k in ['st_bound', 'st_inf', 'st_rain',
                  'st_inflow', 'st_drain']:
            self.arr[k][:] = 0.
            self.stats_update_time[k] = sim_time
        return self
