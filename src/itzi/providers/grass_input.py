# coding=utf8
"""
Copyright (C) 2025 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

from typing import Tuple, Mapping, TypedDict, TYPE_CHECKING

import numpy as np

from itzi.providers.base import RasterInputProvider
from itzi.array_definitions import ARRAY_DEFINITIONS, ArrayCategory
import itzi.messenger as msgr
from itzi.providers.domain_data import DomainData

if TYPE_CHECKING:
    from datetime import datetime
    from itzi.providers.grass_interface import GrassInterface


class GrassRasterInputConfig(TypedDict):
    grass_interface: "GrassInterface"
    # A dict of key: input names
    input_map_names: Mapping[str, str]
    default_start_time: "datetime"
    default_end_time: "datetime"


class GrassRasterInputProvider(RasterInputProvider):
    def __init__(self, config: GrassRasterInputConfig) -> None:
        self.grass_interface = config["grass_interface"]
        self.start_time = config["default_start_time"]
        self.end_time = config["default_end_time"]
        self.map_lists = self.get_map_lists(config["input_map_names"])
        self.origin = self.get_origin()

    def get_domain_data(self) -> DomainData:
        """Return a DomainData object"""
        return DomainData(
            north=self.grass_interface.region.north,
            south=self.grass_interface.region.south,
            east=self.grass_interface.region.east,
            west=self.grass_interface.region.west,
            rows=self.grass_interface.region.rows,
            cols=self.grass_interface.region.cols,
            crs_wkt=self.grass_interface.get_crs_wkt(),
        )

    def get_map_lists(
        self, map_names: Mapping[str, str]
    ) -> Mapping[str, list["GrassInterface.MapData"]]:
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
        map_lists = {
            arr_def.key: None
            for arr_def in ARRAY_DEFINITIONS
            if ArrayCategory.INPUT in arr_def.category
        }
        for k, map_name in map_names.items():
            if not map_name:
                map_list = None
                continue
            map_id = self.grass_interface.format_id(map_name)
            if self.grass_interface.name_is_stds(map_id):
                strds_id = map_id
                if not self.grass_interface.stds_temporal_sanity(strds_id):
                    msgr.fatal("{}: inadequate temporal format".format(map_name))
                map_list = self.grass_interface.raster_list_from_strds(strds_id)
            elif self.grass_interface.name_is_map(map_id):
                map_list = [
                    self.grass_interface.MapData(
                        id=map_id,
                        start_time=self.start_time,
                        end_time=self.end_time,
                    )
                ]
            else:
                msgr.fatal("{} not found!".format(map_name))
            map_lists[k] = map_list
        return map_lists

    def get_array(
        self, map_key: str, current_time: "datetime"
    ) -> Tuple[np.ndarray, "datetime", "datetime"]:
        """Take a given map key and current time
        return a numpy array associated with its start and end time
        if no map is found, return None instead of an array
        and a default start_time and end_time."""
        if map_key not in self.map_lists.keys():
            raise ValueError(f"Unknown map key: {map_key}")
        if self.map_lists[map_key] is None:
            return None, self.start_time, self.end_time
        else:
            for map_name in self.map_lists[map_key]:
                if map_name.start_time <= current_time <= map_name.end_time:
                    arr = self.grass_interface.read_raster_map(map_name.id)
                    return arr, map_name.start_time, map_name.end_time
            else:
                # This should not happen unless the maps is removed from the GRASS DB after listing the maps.
                assert None, "No map found for {k} at time {t}".format(k=map_key, t=current_time)
