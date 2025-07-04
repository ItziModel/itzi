# coding=utf8
"""
Copyright (C) 2016-2025 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

from datetime import datetime
import numpy as np

import itzi.flow as flow
import itzi.messenger as msgr
from itzi import rastermetrics


class TimedArray:
    """A container for np.ndarray with time informations.
    Update the array value according to the simulation time.
    array is accessed via get()
    """

    def __init__(self, mkey, igis, f_arr_def):
        assert isinstance(mkey, str), "not a string!"
        assert hasattr(f_arr_def, "__call__"), "not a function!"
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
        """input being a time in datetime
        If the current stored array is within the range of the map,
        return True
        If not return False
        """
        return bool(self.a_start <= sim_time <= self.a_end)

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


class RasterDomain:
    """Group all rasters for the raster domain.
    Store them as np.ndarray with validity information (TimedArray)
    Include management of the masking and unmasking of arrays.
    """

    def __init__(self, dtype, arr_mask, cell_shape):
        # data type
        self.dtype = dtype
        # geographical data
        self.shape = arr_mask.shape
        self.dx, self.dy = cell_shape
        self.cell_surf = self.dx * self.dy
        self.mask = arr_mask

        # number of cells in a row must be a multiple of that number
        byte_num = 256 / 8  # AVX2
        itemsize = np.dtype(self.dtype).itemsize
        self.row_mul = int(byte_num / itemsize)

        # slice for a simple padding (allow stencil calculation on boundary)
        self.simple_pad = (slice(1, -1), slice(1, -1))

        # all keys that will be used for the arrays
        self.k_input = [
            "dem",
            "friction",
            "h",
            "y",
            "effective_porosity",
            "capillary_pressure",
            "hydraulic_conductivity",
            "soil_water_content",
            "in_inf",
            "losses",
            "rain",
            "inflow",
            "bcval",
            "bctype",
        ]
        self.k_internal = [
            "inf",
            "hmax",
            "ext",
            "y",
            "hfe",
            "hfs",
            "qe",
            "qs",
            "qe_new",
            "qs_new",
            "etp",
            "eff_precip",
            "ue",
            "us",
            "v",
            "vdir",
            "vmax",
            "fr",
            "n_drain",
            "capped_losses",
            "dire",
            "dirs",
        ]
        # arrays gathering the cumulated water depth from corresponding array
        self.k_stats = [
            "st_bound",
            "st_inf",
            "st_rain",
            "st_etp",
            "st_inflow",
            "st_losses",
            "st_ndrain",
            "st_herr",
        ]
        self.k_all = self.k_input + self.k_internal + self.k_stats

        self.start_volume = None

        # Instantiate arrays and padded arrays filled with zeros
        self.arr = dict.fromkeys(self.k_all)
        self.arrp = dict.fromkeys(self.k_all)
        self.create_arrays()

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
        arr_p = np.pad(arr, 1, "edge")
        arr = arr_p[self.simple_pad]
        return arr, arr_p

    def create_arrays(self):
        """Instantiate masked arrays and padded arrays
        the unpadded arrays are a slice of the padded ones
        """
        for k in self.arr.keys():
            self.arr[k], self.arrp[k] = self.pad_array(self.zeros_array())
        return self

    def update_mask(self, arr):
        """Create a mask array by marking NULL values from arr as True."""
        pass
        # self.mask[:] = np.isnan(arr)
        return self

    def mask_array(self, arr, default_value):
        """Replace NULL values in the input array by the default_value"""
        mask = np.logical_or(np.isnan(arr), self.mask)
        arr[mask] = default_value
        assert not np.any(np.isnan(arr))
        return self

    def unmask_array(self, arr):
        """Replace values in the input array by NULL values from mask"""
        unmasked_array = np.copy(arr)
        unmasked_array[self.mask] = np.nan
        return unmasked_array

    def update_ext_array(self):
        """If one of the external input array has been updated,
        combine them into a unique array 'ext' in m/s.
        in_q and n_drain in m/s.
        This applies for inputs that are needed to be taken into account,
         at every timestep, like inflows from user or drainage.
        """
        flow.set_ext_array(
            self.arr["inflow"],
            self.arr["n_drain"],
            self.arr["eff_precip"],
            self.arr["ext"],
        )
        return self

    def swap_arrays(self, k1, k2):
        """swap values of two arrays"""
        self.arr[k1], self.arr[k2] = self.arr[k2], self.arr[k1]
        self.arrp[k1], self.arrp[k2] = self.arrp[k2], self.arrp[k1]
        return self

    def update_array(self, k, arr):
        """Update the values of an array with those of a given array."""
        if arr.shape != self.shape:
            return ValueError
        if k == "dem":
            self.update_mask(arr)
            fill_value = np.finfo(self.dtype).max
        elif k == "h":
            if self.start_volume is None:
                self.start_volume = rastermetrics.calculate_total_volume(
                    arr, self.cell_surf
                )
            fill_value = 0
        elif k == "friction":
            fill_value = 1
        else:
            fill_value = 0
        self.mask_array(arr, fill_value)
        self.arr[k][:], self.arrp[k][:] = self.pad_array(arr)
        return self

    def get_array(self, k):
        """return the unpadded, masked array of key 'k'"""
        return self.arr[k]

    def get_padded(self, k):
        """return the padded, masked array of key 'k'"""
        return self.arrp[k]

    def get_unmasked(self, k):
        """return unpadded array with NaN"""
        return self.unmask_array(self.arr[k])

    def reset_stats(self):
        """Set stats arrays to zeros"""
        for k in self.k_stats:
            self.arr[k][:] = 0.0
        return self
