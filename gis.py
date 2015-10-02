#!/usr/bin/env python
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
import numpy as np
from datetime import datetime, timedelta

import grass.script as grass
import grass.temporal as tgis
from grass.pygrass import raster
from grass.pygrass.gis.region import Region
from grass.pygrass.messages import Messenger

class Igis(object):
    """
    A class providing an access to GRASS GIS Python interfaces:
    scripting, pygrass, temporal GIS
    The interface of this class relies on numpy arrays for raster values.
    Everything related to GRASS maps or stds stays in that class.
    """

    # a unit convertion table relative to seconds
    t_unit_conv = {'seconds': 1, 'minutes': 60, 'hours': 3600, 'days': 86400}
    # datatype conversion between GRASS and numpy
    dtype_conv = {'FCELL': ('float16', 'float32'),
                'DCELL': ('float_', 'float64', 'float128'),
                'CELL': ('bool_', 'int_', 'intc', 'intp',
                        'int8', 'int16', 'int32', 'int64',
                        'uint8', 'uint16', 'uint32', 'uint64')}


    def __init__(self, start_time, end_time):
        assert isinstance(start_time, datetime), \
            "start_time not a datetime object!"
        assert isinstance(end_time, datetime), \
            "end_time not a datetime object!"
        assert start_time <= end_time, "start_time > end_time!"

        self.start_time = start_time
        self.end_time = end_time
        tgis.init()
        msgr = Messenger()
        region = Region()
        self.xr = region.cols
        self.ry = region.rows
        self.dx = region.ewres
        self.dy = region.nsres
        self.overwrite = grass.overwrite()
        self.mapset = grass.read_command('g.mapset', flags='p').strip()

        self.arrays = {'z': None, 'n': None, 'h_old': None,
            'rain': None, 'inf':None, 'bcval': None, 'bctype': None}

    def grass_dtype(self, dtype):
        if dtype in self.dtype_conv['DCELL']:
            mtype = 'DCELL'
        elif dtype in self.dtype_conv['CELL']:
            mtype = 'CELL'
        elif dtype in self.dtype_conv['FCELL']:
            mtype = 'FCELL'
        else:
            assert False, "datatype incompatible with GRASS!"
        return mtype

    def to_s(self, unit, time):
        """Change an input time into seconds
        """
        assert isinstance(unit, basestring), "{} Not a string".format(unit)
        return self.t_unit_conv[unit] * time

    def from_s(self, unit, time):
        """Change an input time from seconds to another unit
        """
        assert isinstance(unit, basestring), "{} Not a string".format(unit)
        return time / self.t_unit_conv[unit]

    def raster_list_from_strds(self, strds_name, sim_end_time):
        """Return a list of maps (as dict) from a given strds
        """
        # !!! only for relative strds
        assert isinstance(strds_name, basestring), \
                "{} not a string".format(strds_name)
        assert isinstance(sim_end_time, (int, float)), \
                "{} not a real number".format(sim_end_time)

        strds = tgis.open_stds.open_old_stds(strds_name, 'strds')
        cols = ['id','name','start_time','end_time',
                'west','east','south','north']
        end_time_in_stds_unit = self.from_s(
                                    strds.get_relative_time_unit(),
                                    sim_end_time)
        where_statement = 'start_time <= {}'.format(
                                str(end_time_in_stds_unit))
        maplist = strds.get_registered_maps(columns=','.join(cols),
                                            where=where_statement,
                                            order='start_time')
        return [dict(zip(cols, i)) for i in maplist]

    def read_raster_map(self, rast_name, dtype):
        """Read a GRASS raster and return a numpy array
        """
        with raster.RasterRow(rast_name, mode='r') as rast:
            array = np.array(rast, dtype=dtype)
        return array

    def write_raster_map(self, arr, rast_name):
        """Take a numpy array and write it to GRASS DB
        """
        if can_ovr == True and raster.RasterRow(rast_name).exist() == True:
            utils.remove(rast_name, 'raster')
            msgr.verbose(_("Removing raster map {}".format(rast_name)))
        mtype = grass_dtype(arr.dtype)
        with raster.RasterRow(rast_name, mode='w', mtype=mtype) as newraster:
            newrow = raster.Buffer((arr.shape[1],))
            for row in arr:
                newrow[:] = row[:]
                newraster.put_row(newrow)
        return self

    def get_input_arrays(self, k_list, sim_time):
        """returns a dict of arrays valid for the current time.
        k_list: a list of requested arrays as dict keys
        """
        assert isinstance(sim_time, datetime), \
            "sim_time not a datetime object!"
        valid_keys = [k for k in self.arrays.keys()]
        for k in k_list:
            assert k in valid_keys
            input_arrays[k] = self.arrays[k]

        return input_arrays
