#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
COPYRIGHT:    (C) 2015 by Laurent Courty

               This program is free software under the GNU General Public
               License (v3).  Read the LICENCE file for details.
"""

import grass.temporal as tgis
from grass.pygrass.messages import Messenger
import numpy as np

import utils
import rw


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


def to_s(unit, time):
    """Change an input time into second
    """
    if unit in ['second', 'seconds']:
        return time
    elif unit in ['minutes', 'minute']:
        return time * 60
    elif unit in ['hours', 'hour']:
        return time * 3600
    elif unit in ["day", "days"]:
        return time * 86400


def from_s(unit, time):
    """Change an input time from seconds to another unit
    """
    if unit in ["second", "seconds"]:
        return time
    elif unit in ['minutes', 'minute']:
        return time / 60
    elif unit in ['hours', 'hour']:
        return time / 3600
    elif unit in ["day", "days"]:
        return time / 86400
    


def update_time_variable_input(strds, sim_clock):
    """Update a TimeArray object according taking account of the
    time variability of the input.
    """
    # select the map that match the simulation time
    sim_clock_map_unit = from_s(strds.get_relative_time_unit(), sim_clock)
    where_statement = 'start_time <= ' + str(sim_clock_map_unit) + ' AND end_time > ' + str(sim_clock_map_unit)
    input_map = strds.get_registered_maps_as_objects(order='start_time', where=where_statement)[0]
    # load the corresponding map
    st = to_s(
                    input_map.get_relative_time_unit(),
                    input_map.relative_time.get_start_time())
    et = to_s(
                    input_map.get_relative_time_unit(),
                    input_map.relative_time.get_end_time())
    ta = TimeArray(
                    start_time = st,
                    end_time = et,
                    arr = input_map.get_np_array())
    return ta


class TimeArray(object):
    """A Numpy array with temporal informations
    start in seconds
    end in seconds
    """

    msgr = Messenger(raise_on_error=True)


    def __init__(self,
                 arr = None,
                 start_time=None,
                 end_time=None):
        self.arr = arr
        self.set_time(start=start_time, end=end_time)


    def is_valid(self, sim_time):
        """Check if the current array is valid for the given time
        """
        if sim_time >= self.start and sim_time < self.end:
            return True
        else:
            return False


    def set_time(self, start = None, end = None):
        """Set the of the array
        Check if start_time < to end_time
        """
        if start == None and end == None:
            self.msgr.fatal('Please provide at least the start or the end time')
        elif start == None:
            self.start = start
        elif end == None:
            self.end = end

        if not start <= end:
            self.msgr.fatal('The end time should be superior or equal to start time')
        else:
            self.start = start
            self.end = end
