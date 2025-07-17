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

from datetime import datetime, timedelta
from typing import Dict, Self

import numpy as np

from itzi.providers.base import RasterOutputProvider
from itzi.data_containers import SimulationData


class GrassRasterOutputProvider(RasterOutputProvider):
    """Write simulation outputs to GRASS."""

    def initialize(self, config: Dict) -> Self:
        """Initialize output provider with configuration."""
        self.grass_interface = config["grass_interface"]
        self.out_map_names = config["out_map_names"]
        self.hmin = config["hmin"]

        self.record_counter = 0
        self.output_maplist = {k: [] for k in self.out_map_names.keys()}
        return self

    def write_array(self, array: np.ndarray, map_key: str, sim_time: datetime | timedelta) -> None:
        """Write simulation data for current time step."""
        suffix = str(self.record_counter).zfill(4)
        map_name = "{}_{}".format(self.out_map_names[map_key], suffix)
        # write the raster
        self.grass_interface.write_raster_map(array, map_name, map_key)
        # Set depth values to null under the given threshold
        if map_key == "h":
            self.grass_interface.set_null(map_name, self.hmin)
        # add map name and time to the corresponding list
        self.output_maplist[map_key].append((map_name, sim_time))

    def write_max_array(self, arr_max, map_key):
        map_max_name = f"{self.out_map_names[map_key]}_max"
        self.grass_interface.write_raster_map(arr_max, map_max_name, map_key)

    def finalize(self, final_data: SimulationData) -> None:
        """Finalize outputs and cleanup."""

        # Write the final raster maps
        self.grass_interface.finalize()
        # register in GRASS temporal framework
        for map_key, lst in self.output_maplist.items():
            strds_name = self.out_map_names[map_key]
            if strds_name is None:
                continue
            self.gis.register_maps_in_stds(map_key, strds_name, lst, "strds", self.temporal_type)
        # write maps of maximal values
        if self.out_map_names["h"]:
            self.write_max_array(final_data.raw_arrays["hmax"], "h")
        if self.out_map_names["v"]:
            self.write_max_array(final_data.raw_arrays["vmax"], "v")
        # Cleanup the GIS state
        # ⚠️ possible race condition with the vector provider
        # The object responsible for creating the object should tear it down too
        self.grass_interface.cleanup()
