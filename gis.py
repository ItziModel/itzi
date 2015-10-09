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
from grass.exceptions import FatalError, CalledModuleError

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
    # conversion from month number to accepted grass month notation
    # datetime object strftime can't be used because it depend on the locale
    month_conv = {1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr',
                5: 'may', 6: 'june', 7: 'july', 8: 'aug',
                9: 'Sept', 10: 'oct', 11: 'Nov', 12: 'dec'}

    def __init__(self, start_time, end_time, dtype, mkeys):
        assert isinstance(start_time, datetime), \
            "start_time not a datetime object!"
        assert isinstance(end_time, datetime), \
            "end_time not a datetime object!"
        assert start_time <= end_time, "start_time > end_time!"

        self.start_time = start_time
        self.end_time = end_time
        self.dtype = dtype
        self.msgr = Messenger(raise_on_error=True)
        region = Region()
        self.xr = region.cols
        self.ry = region.rows
        self.dx = region.ewres
        self.dy = region.nsres
        self.overwrite = grass.overwrite()
        self.mapset = gutils.getenv('MAPSET')
        self.maps = dict.fromkeys(mkeys)
        # init temporal module
        tgis.init(raise_fatal_error=True)
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
        return self.t_unit_conv[unit] * time

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

    def name_is_stds(self, name):
        """return True if the name given as input is a registered strds
        False if not
        """
        if not hasattr(self, 'stds_list'):
            stds_dict = tgis.get_dataset_list('strds', '', columns='id')
            self.stds_list = [v[0] for l in stds_dict.values() for v in l]

        if self.format_id(name) in self.stds_list:
            return True
        else:
            return False

    def name_is_map(self, name):
        """return True if the given name is a map in the grass database
        False if not
        """
        try:
            grass.read_command('r.info', map=self.format_id(name),flags='r')
        except CalledModuleError:
            return False
        else:
            return True

    def read(self, map_names):
        """Read maps names from GIS
        take as input map_names, a dictionary of maps/STDS names
        for each entry in map_names:
            if the name is empty or None, store None
            if a strds, load all maps in the instance's time extend,
                store them as a list
            if a single map, set the start and end time to fit simulation.
                store it in a list for consistency
        each map is stored as a MapData namedtuple
        store result in instance's dictionary
        """
        for k,map_name in map_names.iteritems():
            if not map_name:
                map_list = None
                continue
            elif self.name_is_stds(map_name):
                map_list = self.raster_list_from_strds(self.format_id(map_name))
            elif self.name_is_map(map_name):
                map_list = [self.MapData(id=self.format_id(map_name),
                    start_time=self.start_time, end_time=self.end_time)]
            else:
                self.msgr.fatal(_("{} not found!".format(map_name)))
            self.maps[k] = map_list
        return self

    def raster_list_from_strds(self, strds_name):
        """Return a list of maps from a given strds
        for all the simulation duration
        Each map data is stored as a MapData namedtuple
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

    def write_raster_map(self, arr, rast_name, map_time, temporal_type):
        """Take a numpy array and write it to GRASS DB
        """
        assert isinstance(arr, np.ndarray), "arr not a np array!"
        assert isinstance(rast_name, basestring), "not a string!"
        assert isinstance(temporal_type, basestring), "not a string!"
        assert isinstance(map_time, datetime), "not a datetime object!"
        if self.overwrite == True and raster.RasterRow(rast_name).exist() == True:
            gutils.remove(rast_name, 'raster')
        mtype = self.grass_dtype(arr.dtype)
        with raster.RasterRow(rast_name, mode='w', mtype=mtype) as newraster:
            newrow = raster.Buffer((arr.shape[1],), mtype=mtype)
            for row in arr:
                newrow[:] = row[:]
                newraster.put_row(newrow)
        # write timestamp
        self.write_raster_timestamp(rast_name, map_time, temporal_type)
        return self

    def write_raster_timestamp(self, rast_name, map_time, temporal_type):
        '''rast_name: name of the map in grass
        map_time: a datetime object
        format the command line and run grass module r.timestamp
        '''
        assert isinstance(map_time, datetime), "not a datetime object!"
        if temporal_type == 'relative':
            rel_time = map_time - self.start_time  # create a timedelta
            if rel_time.days < 1:
                rast_time = '{s} seconds'.format(s=rel_time.seconds)
            else:
                rast_time = '{d} days {s} seconds'.format(
                d=rel_time.days, s=rel_time.seconds)
        elif temporal_type == 'absolute':
            rast_time = '{d} {m} {y} {h}:{min}:{s}'.format(y=map_time.year,
                m=self.month_conv[map_time.month], d=map_time.day,
                h=map_time.hour, min=map_time.minute, s=map_time.second)
        else:
            assert False, "unknown temporal type!"
        # write the timestamp in grass
        grass.run_command('r.timestamp', map=rast_name, date=rast_time)
        return self

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

    def register_maps_in_strds(self, mkey, strds_name, map_list, t_type):
        '''Register given maps
        '''
        # create strds
        strds_id = self.format_id(strds_name)
        strds_title = mkey
        strds_desc = ""
        strds = tgis.open_new_stds(strds_id, 'strds', t_type,
                strds_title, strds_desc, "mean", overwrite=self.overwrite)
        # register maps
        raster_dts_lst = []
        for map_name in map_list:
            map_id = self.format_id(map_name)
            raster_dts = tgis.RasterDataset(map_id)
            raster_dts.read_timestamp_from_grass()
            # load spatial data from map
            raster_dts.load()
            if raster_dts.map_exists():
                raster_dts.update()
            else:
                raster_dts.insert()
            raster_dts_lst.append(raster_dts)
        tgis.register.register_map_object_list('raster', raster_dts_lst,
                strds, delete_empty=True, unit='seconds')
        return self


class old_code():
    def create_strds(mapset, stds_h_name, stds_wse_name, sim_start_time, can_ovr):
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
