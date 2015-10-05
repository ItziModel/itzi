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
from collections import namedtuple

import grass.script as grass
import grass.temporal as tgis
from grass.pygrass import raster
from grass.pygrass.gis.region import Region
from grass.pygrass.messages import Messenger
import grass.pygrass.utils as gutils
from grass.exceptions import FatalError

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


    def __init__(self, start_time, end_time, dtype, mkeys):
        assert isinstance(start_time, datetime), \
            "start_time not a datetime object!"
        assert isinstance(end_time, datetime), \
            "end_time not a datetime object!"
        assert start_time <= end_time, "start_time > end_time!"

        self.start_time = start_time
        self.end_time = end_time
        self.dtype = dtype
        tgis.init(raise_fatal_error=True)
        self.msgr = Messenger(raise_on_error=True)
        region = Region()
        self.xr = region.cols
        self.ry = region.rows
        self.dx = region.ewres
        self.dy = region.nsres
        self.overwrite = grass.overwrite()
        self.mapset = gutils.getenv('MAPSET')
        self.maps = dict.fromkeys(mkeys)
        # define MapData namedtuple and cols to retrieve from STRDS
        self.cols = ['id','start_time','end_time']
                #~ 'name','west','east','south','north']
        self.MapData = namedtuple('MapData', self.cols)

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
        return self.t_unit_conv[unit] * int(time)

    def from_s(self, unit, time):
        """Change an input time from seconds to another unit
        """
        assert isinstance(unit, basestring), "{} Not a string".format(unit)
        return int(time) / self.t_unit_conv[unit]

    def to_datetime(self, unit, time):
        """Take a number and a unit as entry
        return a datetime object relative to start_time
        usefull for assigning start_time and end_time
        to maps from relative stds
        """
        return self.start_time + timedelta(seconds=self.to_s(unit, time))

    def format_id(self, name):
        """Take a map or stds name as input
        and return a fully qualified name, i.e. including mapset
        """
        if '@' in name:
            return name
        else:
            return '@'.join((name, self.mapset))

    def read(self, map_names):
        """Read all requested maps from GIS
        take as input map_names, a dictionary of maps/STDS names
        for each entry in map_names:
            if a strds, load all maps in the instance's time extend,
                store them as a list
            if a single map, set the start and end time to fit simulation.
                store it in a list for consistency
        store result in instance's dictionary
        """
        # retrieve a list of the existing datasets
        #~ get_dataset_list(type, temporal_type, columns=None, where=None,
             #~ order=None)
        for k,map_name in map_names.iteritems():
            if not map_name:
                map_list = None
                continue
            try:
                map_list = self.raster_list_from_strds(self.format_id(map_name))
            except FatalError:
                map_list = [self.MapData(id=self.format_id(map_name),
                    start_time=self.start_time, end_time=self.end_time)]
            self.maps[k] = map_list
        return self

    def raster_list_from_strds(self, strds_name):
        """Return a list of maps (as dict) from a given strds
        for all the simulation duration
        """
        assert isinstance(strds_name, basestring), "expect a string"

        strds = tgis.open_stds.open_old_stds(strds_name, 'strds')
        if strds.get_temporal_type() == 'relative':
            # get start time and end time in seconds
            rel_end_time = (self.end_time - self.start_time).total_seconds()
            rel_unit = strds.get_relative_time_unit().encode('ascii','ignore')
            start_time_in_stds_unit = 0
            end_time_in_stds_unit = self.from_s(rel_unit, rel_end_time)
        elif strds.get_temporal_type() == 'absolute':
            start_time_in_stds_unit = self.start_time
            end_time_in_stds_unit = self.end_time
        else:
            assert False, "unknown temporal type"

        # retrieve data from DB
        where = 'start_time <= {e} AND end_time >= {s}'.format(
            e=str(end_time_in_stds_unit), s=str(start_time_in_stds_unit))
        maplist = strds.get_registered_maps(columns=','.join(self.cols),
                                            where=where,
                                            order='start_time')
        # change time data to datetime format
        if strds.get_temporal_type() == 'relative':
            maplist = [(i[0], self.to_datetime(rel_unit, i[1]),
                self.to_datetime(rel_unit, i[2])) for i in maplist]
        return [self.MapData(*i) for i in maplist]

    def read_raster_map(self, rast_name):
        """Read a GRASS raster and return a numpy array
        """
        with raster.RasterRow(rast_name, mode='r') as rast:
            array = np.array(rast, dtype=self.dtype)
        return array

    def write_raster_map(self, arr, rast_name):
        """Take a numpy array and write it to GRASS DB
        """
        assert isinstance(arr, np.ndarray), "arr not a np array!"
        assert isinstance(rast_name, basestring), "rast_name not a string!"
        if self.overwrite == True and raster.RasterRow(rast_name).exist() == True:
            gutils.remove(rast_name, 'raster')
            self.msgr.verbose(_("Removing raster map {}".format(rast_name)))
        mtype = self.grass_dtype(arr.dtype)
        with raster.RasterRow(rast_name, mode='w', mtype=mtype) as newraster:
            newrow = raster.Buffer((arr.shape[1],))
            for row in arr:
                newrow[:] = row[:]
                newraster.put_row(newrow)
        return self

    def get_input_arrays(self, arrays, sim_time):
        """take a list of map names as input
        add valid array to each relevant record
        """
        assert isinstance(sim_time, datetime), \
            "sim_time not a datetime object!"

        input_arrays = {}
        for k,l in self.maps.iteritems():
            if l:
                for m in l:
                    if m.start_time <= sim_time <= m.end_time:
                        input_arrays[k] = self.read_raster_map(m.id)
            else:
                input_arrays[k] = None
        return input_arrays

    def get_array(self, mkey, sim_time):
        """take a given map key and simulation time
        return a numpy array associated with its start and end time
        if no map is found, return None instead of an array
        and the start_time and end_time of the simulation
        """
        assert isinstance(mkey, basestring), "not a string!"
        assert isinstance(sim_time, datetime), "not a datetime object!"
        assert mkey in self.maps.keys(), "unknown map key!"
        if self.maps[mkey] == None:
            return None, self.start_time, self.end_time
        else:
            for m in self.maps[mkey]:
                if m.start_time <= sim_time <= m.end_time:
                    arr = self.read_raster_map(m.id)
            return arr, m.start_time, m.end_time

class old_code():
    def create_stds(mapset, stds_h_name, stds_wse_name, sim_start_time, can_ovr):
        """create wse and water depth STRDS
        """

        # set ids, name and decription of result data sets
        stds_h_id = rw.format_opt_map(stds_h_name, mapset)
        stds_wse_id = rw.format_opt_map(stds_wse_name, mapset)
        stds_h_title = "water depth"
        stds_wse_title = "water surface elevation"
        stds_h_desc = "water depth generated on " + sim_start_time.isoformat()
        stds_wse_desc = "water surface elevation generated on " + sim_start_time.isoformat()
        # data type of stds
        stds_dtype = "strds"
        # Temporal type of stds
        temporal_type = "relative"

        # database connection
        dbif = tgis.SQLDatabaseInterfaceConnection()
        dbif.connect()

        # water depth
        if stds_h_name:
            stds_h = tgis.open_new_stds(stds_h_name, stds_dtype,
                            temporal_type, stds_h_title, stds_h_desc,
                            "mean", dbif=dbif, overwrite=can_ovr)
        # water surface elevation
        if stds_wse_name:
            stds_wse = tgis.open_new_stds(stds_wse_name, stds_dtype,
                            temporal_type, stds_wse_title, stds_wse_desc,
                            "mean", dbif=dbif, overwrite=can_ovr)
        return stds_h_id, stds_wse_id

    def register_in_stds():
        # depth
        if options['out_h']:
            list_h = ','.join(list_h) # transform map list into a string
            kwargs = {'maps': list_h,
                    'start': 0,
                    'unit':'seconds',
                    'increment':int(record_t)}
            tgis.register.register_maps_in_space_time_dataset('rast',
                                                    stds_h_id, **kwargs)
