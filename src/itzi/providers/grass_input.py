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

from typing import Tuple, TypedDict, TYPE_CHECKING

import numpy as np

from itzi.providers.base import RasterInputProvider

if TYPE_CHECKING:
    from datetime import datetime
    from itzi.providers.grass_interface import GrassInterface


class GrassRasterInputConfig(TypedDict):
    grass_interface: "GrassInterface"


class GrassRasterInputProvider(RasterInputProvider):
    def __init__(self, config: GrassRasterInputConfig) -> None:
        self.grass_interface = config["grass_interface"]

    def get_array(
        self, map_key: str, current_time: "datetime"
    ) -> Tuple[np.ndarray, "datetime", "datetime"]:
        """Take a given map key and current time
        return a numpy array associated with its start and end time
        if no map is found, return None instead of an array
        and a default start_time and end_time."""
        arr, arr_start, arr_end = self.grass_interface.get_array(map_key, current_time)
        return arr, arr_start, arr_end
