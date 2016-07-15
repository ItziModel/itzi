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
import numpy as np
from datetime import datetime, timedelta
from collections import namedtuple
import os

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
                  'DCELL': ('float_', 'float64'),
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
        self.msgr = Messenger()
        region = Region()
        self.xr = region.cols
        self.yr = region.rows
        self.dx = region.ewres
        self.dy = region.nsres
        self.overwrite = grass.overwrite()
        self.mapset = gutils.getenv('MAPSET')
        self.maps = dict.fromkeys(mkeys)
        # init temporal module
        tgis.init(raise_fatal_error=True)
        # define MapData namedtuple and cols to retrieve from STRDS
        self.cols = ['id', 'start_time', 'end_time']
        self.MapData = namedtuple('MapData', self.cols)

        # color tables files
        _ROOT = os.path.dirname(__file__)
        _DIR = os.path.join(_ROOT, 'data', 'colortable')
        _H = 'depth.txt'
        _V = 'velocity.txt'
        _DEF = 'default.txt'
        self.rules_h = os.path.join(_DIR, _H)
        self.rules_v = os.path.join(_DIR, _V)
        self.rules_def = os.path.join(_DIR, _DEF)
        assert os.path.isfile(self.rules_h)
        assert os.path.isfile(self.rules_v)
        assert os.path.isfile(self.rules_def)

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

    @staticmethod
    def format_id(name):
        """Take a map or stds name as input
        and return a fully qualified name, i.e. including mapset
        """
        if '@' in name:
            return name
        else:
            return '@'.join((name, gutils.getenv('MAPSET')))

    @staticmethod
    def name_is_stds(name):
        """return True if the name given as input is a registered strds
        False if not
        """
        # make sure temporal module is initialized
        tgis.init(raise_fatal_error=True)
        if tgis.SpaceTimeRasterDataset(name).is_in_db():
            return True
        else:
            return False

    @staticmethod
    def name_is_map(map_id):
        """return True if the given name is a map in the grass database
        False if not
        """
        try:
            grass.read_command('g.findfile', file=map_id, element='cell')
        except CalledModuleError:
            return False
        else:
            return True

    def get_sim_extend_in_stds_unit(self, strds):
        """Take a strds object as input
        Return the simulation start_time and end_time, expressed in
        the unit of the input strds
        """
        if strds.get_temporal_type() == 'relative':
            # get start time and end time in seconds
            rel_end_time = (self.end_time - self.start_time).total_seconds()
            rel_unit = strds.get_relative_time_unit().encode('ascii', 'ignore')
            start_time_in_stds_unit = 0
            end_time_in_stds_unit = self.from_s(rel_unit, rel_end_time)
        elif strds.get_temporal_type() == 'absolute':
            start_time_in_stds_unit = self.start_time
            end_time_in_stds_unit = self.end_time
        else:
            assert False, "unknown temporal type"
        return start_time_in_stds_unit, end_time_in_stds_unit

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
        for k, map_name in map_names.iteritems():
            if not map_name:
                map_list = None
                continue
            elif self.name_is_stds(self.format_id(map_name)):
                strds_id = self.format_id(map_name)
                if not self.stds_temporal_sanity(strds_id):
                    self.msgr.fatal(_
                        (u"{}: inadequate temporal format".format(map_name)))
                map_list = self.raster_list_from_strds(strds_id)
            elif self.name_is_map(self.format_id(map_name)):
                map_list = [self.MapData(id=self.format_id(map_name),
                                         start_time=self.start_time,
                                         end_time=self.end_time)]
            else:
                self.msgr.fatal(_(u"{} not found!".format(map_name)))
            self.maps[k] = map_list
        return self

    def stds_temporal_sanity(self, stds_id):
        """Make the following check on the given stds:
        - Topology is valid
        - No gap
        - Cover all simulation time
        return True if all the above is True, False otherwise
        """
        out = True
        stds = tgis.open_stds.open_old_stds(stds_id, 'strds')
        # valid topology
        if not stds.check_temporal_topology():
            out = False
            self.msgr.warning(_(u"{}: invalid topology".format(stds_id)))
        # no gap
        if stds.count_gaps() != 0:
            out = False
            self.msgr.warning(_(u"{}: gaps found".format(stds_id)))
        # cover all simulation time
        sim_start, sim_end = self.get_sim_extend_in_stds_unit(stds)
        stds_start, stds_end = stds.get_temporal_extent_as_tuple()
        if stds_start > sim_start:
            out = False
            self.msgr.warning(_(u"{}: starts after simulation".format(stds_id)))
        if stds_end < sim_end:
            out = False
            self.msgr.warning(_(u"{}: ends before simulation".format(stds_id)))
        return out

    def raster_list_from_strds(self, strds_name):
        """Return a list of maps from a given strds
        for all the simulation duration
        Each map data is stored as a MapData namedtuple
        """
        assert isinstance(strds_name, basestring), "expect a string"

        # transform simulation start and end time in strds unit
        strds = tgis.open_stds.open_old_stds(strds_name, 'strds')
        sim_start, sim_end = self.get_sim_extend_in_stds_unit(strds)

        # retrieve data from DB
        where = "start_time <= '{e}' AND end_time >= '{s}'".format(
            e=str(sim_end), s=str(sim_start))
        maplist = strds.get_registered_maps(columns=','.join(self.cols),
                                            where=where,
                                            order='start_time')
        # check if every map exist
        maps_not_found = [m[0] for m in maplist if not self.name_is_map(m[0])]
        if any(maps_not_found):
            err_msg = u"STRDS <{}>: Can't find following maps: {}"
            str_lst = ','.join(maps_not_found)
            self.msgr.fatal(err_msg.format(strds_name, str_lst))
        # change time data to datetime format
        if strds.get_temporal_type() == 'relative':
            rel_unit = strds.get_relative_time_unit().encode('ascii', 'ignore')
            maplist = [(i[0], self.to_datetime(rel_unit, i[1]),
                       self.to_datetime(rel_unit, i[2])) for i in maplist]
        return [self.MapData(*i) for i in maplist]

    def read_raster_map(self, rast_name):
        """Read a GRASS raster and return a numpy array
        """
        with raster.RasterRow(rast_name, mode='r') as rast:
            array = np.array(rast, dtype=self.dtype)
        return array

    def write_raster_map(self, arr, rast_name, mkey):
        """Take a numpy array and write it to GRASS DB
        """
        assert isinstance(arr, np.ndarray), "arr not a np array!"
        assert isinstance(rast_name, basestring), "not a string!"
        mtype = self.grass_dtype(arr.dtype)
        with raster.RasterRow(rast_name, mode='w', mtype=mtype,
                              overwrite=self.overwrite) as newraster:
            newrow = raster.Buffer((arr.shape[1],), mtype=mtype)
            for row in arr:
                newrow[:] = row[:]
                newraster.put_row(newrow)
        # apply color table
        self.apply_color_table(rast_name, mkey)
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
        if self.maps[mkey] is None:
            return None, self.start_time, self.end_time
        else:
            for m in self.maps[mkey]:
                if m.start_time <= sim_time <= m.end_time:
                    arr = self.read_raster_map(m.id)
                    return arr, m.start_time, m.end_time
            else:
                assert None, "No map found for {k} at time {t}".format(
                                            k=mkey, t=sim_time)

    def register_maps_in_strds(self, mkey, strds_name, map_list, t_type):
        '''Register given maps
        '''
        assert isinstance(mkey, basestring), "not a string!"
        assert isinstance(strds_name, basestring), "not a string!"
        assert isinstance(t_type, basestring), "not a string!"
        # create strds
        strds_id = self.format_id(strds_name)
        strds_title = mkey
        strds_desc = ""
        strds = tgis.open_new_stds(strds_id, 'strds', t_type,
                                   strds_title, strds_desc, "mean",
                                   overwrite=self.overwrite)

        # create RasterDataset objects list
        raster_dts_lst = []
        for map_name, map_time in map_list:
            # create RasterDataset
            map_id = self.format_id(map_name)
            raster_dts = tgis.RasterDataset(map_id)
            # load spatial data from map
            raster_dts.load()
            # set time
            if t_type == 'relative':
                # create timedelta
                rel_time = map_time - self.start_time
                rast_time = rel_time.total_seconds()
                raster_dts.set_relative_time(rast_time, None, 'seconds')
            elif t_type == 'absolute':
                raster_dts.set_absolute_time(start_time=map_time)
            else:
                assert False, "unknown temporal type!"
            # populate the list
            raster_dts_lst.append(raster_dts)
        # Finaly register the maps
        if t_type == 'relative':
            r_unit = 'seconds'
        elif t_type == 'absolute':
            r_unit = ''
        else:
            assert False, "unknown temporal type!"
        tgis.register.register_map_object_list('raster', raster_dts_lst,
                                               strds, delete_empty=True,
                                               unit=r_unit)
        return self

    def apply_color_table(self, map_name, mkey):
        '''apply a color table determined by mkey to the given map
        '''
        if mkey == 'h':
            colors_rules = self.rules_h
        elif mkey == 'v':
            colors_rules = self.rules_v
        elif mkey == 'vdir':
            colors_rules = 'aspectcolr'
        else:
            colors_rules = self.rules_def
        grass.run_command('r.colors', quiet=True,
                          rules=colors_rules, map=map_name)
        return self
