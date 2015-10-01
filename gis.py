#!/usr/bin/env python
# coding=utf8
"""
Copyright (C) 2015  Laurent Courty lrntct@gmail.com

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
    t_unit_conv = {'seconds': 1,
                    'minutes': 60,
                    'hours': 3600,
                    'days': 86400}

    def __init__(self):
        tgis.init()

    def to_s(self, unit, time):
        """Change an input time into second
        """
        assert isinstance(unit, basestring), "{} Not a string".format(unit)
        return self.t_unit_conv[unit] * time

    def from_s(self, unit, time):
        """Change an input time from seconds to another unit
        """
        assert isinstance(unit, basestring), "{} Not a string".format(unit)
        return time / self.t_unit_conv[unit]


    def get_map_list_from_strds(self, strds_name, sim_end_time):
        """Return a list of maps (as dict) from a given strds
        """
        assert isinstance(strds_name, basestring), \
                "{} not a string".format(strds_name)
        assert isinstance(sim_end_time, (int, float)), \
                "{} not a real number".format(sim_end_time)
        # !!! only for relative strds
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
