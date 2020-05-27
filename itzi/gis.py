# coding=utf8
"""
Copyright (C) 2015-2020 Laurent Courty

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
from __future__ import absolute_import
import os
import atexit
from collections import namedtuple
from datetime import datetime, timedelta
# from multiprocessing import Process, JoinableQueue
from threading import Thread, Lock
from queue import Queue
import copy

import numpy as np

import itzi.messenger as msgr

import grass.script as gscript
import grass.temporal as tgis
import grass.pygrass.utils as gutils
from grass.pygrass.gis.region import Region
from grass.pygrass import raster
from grass.pygrass.vector import VectorTopo
from grass.pygrass.vector.geometry import Point, Line
from grass.pygrass.vector.basic import Cats
from grass.pygrass.vector.table import Link

# color rules
_ROOT = os.path.dirname(__file__)
_DIR = os.path.join(_ROOT, 'data', 'colortable')
RULE_H = os.path.join(_DIR, 'depth.txt')
RULE_V = os.path.join(_DIR, 'velocity.txt')
RULE_VDIR = os.path.join(_DIR, 'vdir.txt')
RULE_FR = os.path.join(_DIR, 'froude.txt')
RULE_DEF = os.path.join(_DIR, 'default.txt')
colors_rules_dict = {'h': RULE_H, 'v': RULE_V, 'vdir': RULE_VDIR,
                     'fr': RULE_FR}
# Check if color rule paths are OK
for f in colors_rules_dict.values():
    assert os.path.isfile(f)


def file_exists(name):
    """Return True if name is an existing map or stds, False otherwise
    """
    if not name:
        return False
    else:
        _id = Igis.format_id(name)
        return Igis.name_is_map(_id) or Igis.name_is_stds(_id)


def check_output_files(file_list):
    """Check if the output files exist
    """
    for map_name in file_list:
        if file_exists(map_name) and not gscript.overwrite():
            msgr.fatal(u"File {} exists and will not be overwritten".format(map_name))


def apply_color_table(map_name, mkey):
    '''Apply a color table determined by mkey to the given map
    '''
    try:
        colors_rules = colors_rules_dict[mkey]
    except KeyError:
        # in case no specific color table is given, use GRASS default.
        pass
    else:
        gscript.run_command('r.colors', quiet=True,
                            rules=colors_rules, map=map_name)
    return None


def raster_writer(q, lock):
    """Write a raster map in GRASS
    """
    while True:
        # Get values from the queue
        next_object = q.get()
        if next_object is None:
            break
        arr, rast_name, mtype, mkey, overwrite = next_object
        # Write raster
        lock.acquire()
        with raster.RasterRow(rast_name, mode='w', mtype=mtype,
                              overwrite=overwrite) as newraster:
            newrow = raster.Buffer((arr.shape[1],), mtype=mtype)
            for row in arr:
                newrow[:] = row[:]
                newraster.put_row(newrow)
        # Apply colour table
        apply_color_table(rast_name, mkey)
        lock.release()
        # Signal end of task
        q.task_done()


class Igis():
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

    # define used namedtuples
    strds_cols = ['id', 'start_time', 'end_time']
    MapData = namedtuple('MapData', strds_cols)
    LinkDescr = namedtuple('LinkDescr', ['layer', 'table'])

    def __init__(self, start_time, end_time, dtype, mkeys, region_id, raster_mask_id):
        assert isinstance(start_time, datetime), \
            "start_time not a datetime object!"
        assert isinstance(end_time, datetime), \
            "end_time not a datetime object!"
        assert start_time <= end_time, "start_time > end_time!"

        self.region_id = region_id
        self.raster_mask_id = raster_mask_id
        self.start_time = start_time
        self.end_time = end_time
        self.dtype = dtype

        self.old_mask_name = None

        # LatLon is not supported
        if gscript.locn_is_latlong():
            msgr.fatal(u"latlong location is not supported. "
                       u"Please use a projected location")
        # Set region
        if self.region_id:
            gscript.use_temp_region()
            gscript.run_command("g.region", region=region_id)
        self.region = Region()
        self.xr = self.region.cols
        self.yr = self.region.rows
        # Check if region is at least 3x3
        if self.xr < 3 or self.yr < 3:
            msgr.fatal(u"GRASS Region should be at least 3 cells by 3 cells")
        self.dx = self.region.ewres
        self.dy = self.region.nsres
        self.reg_bbox = {'e': self.region.east, 'w': self.region.west,
                         'n': self.region.north, 's': self.region.south}
        # Set temporary mask
        if self.raster_mask_id:
            self.set_temp_mask()
        self.overwrite = gscript.overwrite()
        self.mapset = gutils.getenv('MAPSET')
        self.maps = dict.fromkeys(mkeys)
        # init temporal module
        tgis.init()
        # Create thread and queue for writing raster maps
        self.raster_lock = Lock()
        self.raster_writer_queue = Queue(maxsize=15)
        worker_args = (self.raster_writer_queue, self.raster_lock)
        self.raster_writer_thread = Thread(name="RasterWriter",
                                           target=raster_writer,
                                           args=worker_args)
        self.raster_writer_thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finalize()
        self.cleanup()

    def finalize(self):
        """Make sure that all maps are written.
        """
        msgr.debug("Writing last maps...")
        self.raster_writer_queue.join()

    def cleanup(self):
        """Remove temporary region and mask.
        """
        msgr.debug("Reset mask and region")
        if self.raster_mask_id:
            msgr.debug("Remove temp MASK...")
            self.del_temp_mask()
        if self.region_id:
            msgr.debug("Remove temp region...")
            gscript.del_temp_region()
        self.raster_writer_queue.put(None)
        self.raster_writer_thread.join()

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
        assert isinstance(unit, str), u"{} Not a string".format(unit)
        return self.t_unit_conv[unit] * time

    def from_s(self, unit, time):
        """Change an input time from seconds to another unit
        """
        assert isinstance(unit, str), u"{} Not a string".format(unit)
        return int(time) / self.t_unit_conv[unit]

    def to_datetime(self, unit, time):
        """Take a number and a unit as entry
        return a datetime object relative to start_time
        usefull for assigning start_time and end_time
        to maps from relative stds
        """
        return self.start_time + timedelta(seconds=self.to_s(unit, time))

    def has_mask(self):
        """Return True if the mapset has a mask, False otherwise.
        """
        return bool(gscript.read_command("g.list", type="raster", pattern="MASK"))

    def get_npmask(self):
        """Return a boolean numpy ndarray where True is outside the domain.
        """
        if self.has_mask():
            grass_mask = self.read_raster_map('MASK')
            return ~np.isclose(grass_mask, 1.)
        else:
            return np.full(shape=(self.yr, self.xr), fill_value=False, dtype=np.bool_)

    def set_temp_mask(self):
        """If a mask is already set, keep it for later.
        Set a new mask.
        """
        has_old_mask = self.has_mask()
        if has_old_mask:
            # Save the current MASK under a temp name
            self.old_mask_name = "itzi_old_MASK_{}".format(os.getpid())
            gscript.run_command("g.rename", quiet=True, overwrite=True,
                                raster=f"MASK,{self.old_mask_name}")
        gscript.run_command("r.mask", quiet=True, raster=self.raster_mask_id)
        assert self.has_mask()
        return self

    def del_temp_mask(self):
        """Reset the old mask, remove if there was not.
        """
        if self.old_mask_name is not None:
            gscript.run_command("g.rename", quiet=True, overwrite=True,
                                raster=f"{self.old_mask_name},MASK")
        else:
            gscript.run_command("r.mask", quiet=True, flags='r')
        return self

    def coor2pixel(self, coor):
        """convert coordinates easting and northing to pixel row and column
        """
        return gutils.coor2pixel(coor, self.region)

    def is_in_region(self, x, y):
        """For a given coordinate pair(x, y),
        return True is inside raster region, False otherwise.
        """
        bool_x = (self.reg_bbox['w'] < x < self.reg_bbox['e'])
        bool_y = (self.reg_bbox['s'] < y < self.reg_bbox['n'])
        return bool(bool_x and bool_y)

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
        tgis.init()
        return bool(tgis.SpaceTimeRasterDataset(name).is_in_db())

    @staticmethod
    def name_is_map(map_id):
        """return True if the given name is a map in the grass database
        False if not
        """
        return bool(gscript.find_file(name=map_id, element='cell').get('file'))

    def get_sim_extend_in_stds_unit(self, strds):
        """Take a strds object as input
        Return the simulation start_time and end_time, expressed in
        the unit of the input strds
        """
        if strds.get_temporal_type() == 'relative':
            # get start time and end time in seconds
            rel_end_time = (self.end_time - self.start_time).total_seconds()
            rel_unit = strds.get_relative_time_unit()
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
        for k, map_name in map_names.items():
            if not map_name:
                map_list = None
                continue
            elif self.name_is_stds(self.format_id(map_name)):
                strds_id = self.format_id(map_name)
                if not self.stds_temporal_sanity(strds_id):
                    msgr.fatal(u"{}: inadequate temporal format"
                               u"".format(map_name))
                map_list = self.raster_list_from_strds(strds_id)
            elif self.name_is_map(self.format_id(map_name)):
                map_list = [self.MapData(id=self.format_id(map_name),
                                         start_time=self.start_time,
                                         end_time=self.end_time)]
            else:
                msgr.fatal(u"{} not found!".format(map_name))
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
            msgr.warning(u"{}: invalid topology".format(stds_id))
        # no gap
        if stds.count_gaps() != 0:
            out = False
            msgr.warning(u"{}: gaps found".format(stds_id))
        # cover all simulation time
        sim_start, sim_end = self.get_sim_extend_in_stds_unit(stds)
        stds_start, stds_end = stds.get_temporal_extent_as_tuple()
        if stds_start > sim_start:
            out = False
            msgr.warning(u"{}: starts after simulation".format(stds_id))
        if stds_end < sim_end:
            out = False
            msgr.warning(u"{}: ends before simulation".format(stds_id))
        return out

    def raster_list_from_strds(self, strds_name):
        """Return a list of maps from a given strds
        for all the simulation duration
        Each map data is stored as a MapData namedtuple
        """
        assert isinstance(strds_name, str), u"expect a string"

        # transform simulation start and end time in strds unit
        strds = tgis.open_stds.open_old_stds(strds_name, 'strds')
        sim_start, sim_end = self.get_sim_extend_in_stds_unit(strds)

        # retrieve data from DB
        where = "start_time <= '{e}' AND end_time >= '{s}'".format(
            e=str(sim_end), s=str(sim_start))
        maplist = strds.get_registered_maps(columns=','.join(self.strds_cols),
                                            where=where,
                                            order='start_time')
        # check if every map exist
        maps_not_found = [m[0] for m in maplist if not self.name_is_map(m[0])]
        if any(maps_not_found):
            err_msg = u"STRDS <{}>: Can't find following maps: {}"
            str_lst = ','.join(maps_not_found)
            msgr.fatal(err_msg.format(strds_name, str_lst))
        # change time data to datetime format
        if strds.get_temporal_type() == 'relative':
            rel_unit = strds.get_relative_time_unit()
            maplist = [(i[0], self.to_datetime(rel_unit, i[1]),
                        self.to_datetime(rel_unit, i[2])) for i in maplist]
        return [self.MapData(*i) for i in maplist]

    def read_raster_map(self, rast_name):
        """Read a GRASS raster and return a numpy array
        """
        self.raster_lock.acquire()
        with raster.RasterRow(rast_name, mode='r') as rast:
            array = np.array(rast, dtype=self.dtype)
        self.raster_lock.release()
        return array

    def write_raster_map(self, arr, rast_name, mkey):
        """Take a numpy array and write it to GRASS DB
        """
        assert isinstance(arr, np.ndarray), u"arr not a np array!"
        assert isinstance(rast_name, str), u"not a string!"
        assert isinstance(mkey, str), u"not a string!"
        # self.write_raster_map_blocking(arr, rast_name, mkey)
        self.write_raster_map_nonblocking(arr, rast_name, mkey)
        return self

    def write_raster_map_nonblocking(self, arr, rast_name, mkey):
        mtype = self.grass_dtype(arr.dtype)
        assert isinstance(mtype, str), u"not a string!"
        q_obj = (arr.copy(), copy.deepcopy(rast_name), copy.deepcopy(mtype),
                 copy.deepcopy(mkey), self.overwrite)
        self.raster_writer_queue.put(q_obj)
        return self

    def write_raster_map_blocking(self, arr, rast_name, mkey):
        mtype = self.grass_dtype(arr.dtype)
        assert isinstance(mtype, str), u"not a string!"
        with raster.RasterRow(rast_name, mode='w', mtype=mtype,
                              overwrite=self.overwrite) as newraster:
            newrow = raster.Buffer((arr.shape[1],), mtype=mtype)
            for row in arr:
                newrow[:] = row[:]
                newraster.put_row(newrow)
        # apply color table
        apply_color_table(rast_name, mkey)
        return self

    def create_db_links(self, vect_map, linking_elem):
        """vect_map an open vector map
        """
        dblinks = {}
        for layer_name, layer_dscr in linking_elem.items():
            # Create DB links
            dblink = Link(layer=layer_dscr.layer_number, name=layer_name,
                          table=vect_map.name + layer_dscr.table_suffix, key='cat')
            # add link to vector map
            if dblink not in vect_map.dblinks:
                vect_map.dblinks.add(dblink)
            # create table
            dbtable = dblink.table()
            dbtable.create(layer_dscr.cols, overwrite=True)
            dblinks[layer_name] = self.LinkDescr(dblink.layer, dbtable)
        return dblinks

    def write_vector_map(self, drainage_network, map_name, linking_elem):
        """Write a vector map to GRASS GIS using
        drainage_network is a networkx object
        """
        with VectorTopo(map_name, mode='w', overwrite=self.overwrite) as vect_map:
            # create db links and tables
            dblinks = self.create_db_links(vect_map, linking_elem)

            # set category manually
            cat_num = 1

            # dict to keep DB infos to write DB after geometries
            db_info = {k: [] for k in linking_elem}

            # Points
            for node in drainage_network.nodes():
                if node.coordinates:
                    point = Point(*node.coordinates)
                    # add values
                    map_layer, dbtable = dblinks['node']
                    self.write_vector_geometry(vect_map, point,
                                               cat_num, map_layer)
                    # Get DB attributes
                    attrs = tuple([cat_num] + node.get_attrs())
                    db_info['node'].append(attrs)
                    # bump cat
                    cat_num += 1

            # Lines
            for in_node, out_node, edge_data in drainage_network.edges_iter(data=True):
                link = edge_data['object']
                # assemble geometry
                in_node_coor = in_node.coordinates
                out_node_coor = out_node.coordinates
                if in_node_coor and out_node_coor:
                    line_object = Line([in_node_coor]
                                       + link.vertices
                                       + [out_node_coor])
                    # set category and layer link
                    map_layer, dbtable = dblinks['link']
                    self.write_vector_geometry(vect_map, line_object,
                                               cat_num, map_layer)
                    # keep DB info
                    attrs = tuple([cat_num] + link.get_attrs())
                    db_info['link'].append(attrs)
                    # bump cat
                    cat_num += 1

        # write DB
        for geom_type, attrs in db_info.items():
            map_layer, dbtable = dblinks[geom_type]
            for attr in attrs:
                dbtable.insert(attr)
            dbtable.conn.commit()
        return self

    def write_vector_geometry(self, vector_map, geom, cat_num, map_layer):
        """Write geometry in the adequate layer
        """
        cats = Cats(geom.c_cats)
        cats.reset()
        cats.set(cat_num, map_layer)
        # write geometry
        vector_map.write(geom)

    def get_array(self, mkey, sim_time):
        """take a given map key and simulation time
        return a numpy array associated with its start and end time
        if no map is found, return None instead of an array
        and the start_time and end_time of the simulation
        """
        assert isinstance(mkey, str), u"not a string!"
        assert isinstance(sim_time, datetime), u"not a datetime object!"
        assert mkey in self.maps.keys(), u"unknown map key!"
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

    def register_maps_in_stds(self, stds_title, stds_name, map_list, stds_type, t_type):
        """Create a STDS, create one mapdataset for each map and
        register them in the temporal database
        """
        assert isinstance(stds_title, str), u"not a string!"
        assert isinstance(stds_name, str), u"not a string!"
        assert isinstance(t_type, str), u"not a string!"
        # Print message in case of decreased GRASS verbosity
        if msgr.verbosity() <= 2:
            msgr.message(u"Registering maps in temporal framework...")
        # create stds
        stds_id = self.format_id(stds_name)
        stds_desc = ""
        stds = tgis.open_new_stds(stds_id, stds_type, t_type,
                                  stds_title, stds_desc, "mean",
                                  overwrite=self.overwrite)

        # create MapDataset objects list
        map_dts_lst = []
        for map_name, map_time in map_list:
            # create MapDataset
            map_id = self.format_id(map_name)
            map_dts_type = {'strds': tgis.RasterDataset,
                            'stvds': tgis.VectorDataset}
            map_dts = map_dts_type[stds_type](map_id)
            # load spatial data from map
            map_dts.load()
            # set time
            assert isinstance(map_time, datetime)
            if t_type == 'relative':
                rel_time = (map_time - self.start_time).total_seconds()
                map_dts.set_relative_time(rel_time, None, 'seconds')
            elif t_type == 'absolute':
                map_dts.set_absolute_time(start_time=map_time)
            else:
                assert False, "unknown temporal type!"
            # populate the list
            map_dts_lst.append(map_dts)
        # Finally register the maps
        t_unit = {'relative': 'seconds', 'absolute': ''}
        stds_corresp = {'strds': 'raster', 'stvds': 'vector'}
        del_empty = {'strds': True, 'stvds': False}
        tgis.register.register_map_object_list(stds_corresp[stds_type],
                                               map_dts_lst, stds,
                                               delete_empty=del_empty[stds_type],
                                               unit=t_unit[t_type])
        return self
