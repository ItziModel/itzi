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

from itzi.providers.base import RasterOutputProvider, VectorOutputProvider
from itzi.data_containers import SimulationData, DrainageNetworkData


class GrassRasterOutputProvider(RasterOutputProvider):
    """Write simulation outputs to GRASS."""

    def initialize(self, config: Dict) -> Self:
        """Initialize output provider with configuration."""
        self.grass_interface = config["grass_interface"]
        # user-selected map names. Keys are user-facing names
        self.out_map_names = config["out_map_names"]
        self.hmin = config["hmin"]
        self.temporal_type = config["temporal_type"]
        self.record_counter = {k: 0 for k in self.out_map_names.keys()}
        self.output_maplist = {k: [] for k in self.out_map_names.keys()}
        return self

    def write_array(self, array: np.ndarray, map_key: str, sim_time: datetime | timedelta) -> None:
        """Write simulation data for current time step."""
        suffix = str(self.record_counter[map_key]).zfill(4)
        map_name = "{}_{}".format(self.out_map_names[map_key], suffix)
        # write the raster
        self.grass_interface.write_raster_map(array, map_name, map_key, self.hmin)
        # Set depth values to null under the given threshold. Temporarily in gis.py
        # if map_key == "water_depth":
        #     self.grass_interface.set_null(map_name, self.hmin)
        # add map name and time to the corresponding list
        self.output_maplist[map_key].append((map_name, sim_time))
        self.record_counter[map_key] += 1

    def _write_max_array(self, arr_max, map_key):
        map_max_name = f"{self.out_map_names[map_key]}_max"
        self.grass_interface.write_raster_map(arr_max, map_max_name, map_key, hmin=0.0)

    def finalize(self, final_data: SimulationData) -> None:
        """Finalize outputs and cleanup."""

        # Write the final raster maps
        self.grass_interface.finalize()
        # register in GRASS temporal framework
        for map_key, lst in self.output_maplist.items():
            strds_name = self.out_map_names[map_key]
            if strds_name is None:
                continue
            self.grass_interface.register_maps_in_stds(
                map_key, strds_name, lst, "strds", self.temporal_type
            )
        # write maps of maximal values
        if self.out_map_names["water_depth"]:
            self._write_max_array(final_data.raw_arrays["hmax"], "water_depth")
        if self.out_map_names["v"]:
            self._write_max_array(final_data.raw_arrays["vmax"], "v")


class GrassVectorOutputProvider(VectorOutputProvider):
    """Write drainage simulation outputs to GRASS."""

    def initialize(self, config: Dict) -> Self:
        """Initialize output provider with simulation configuration."""
        self.grass_interface = config["grass_interface"]
        self.drainage_map_name = config["drainage_map_name"]
        self.temporal_type = config["temporal_type"]

        self.record_counter = 0
        self.vector_drainage_maplist = []
        return self

    def write_vector(
        self, drainage_data: DrainageNetworkData, sim_time: datetime | timedelta
    ) -> None:
        """Write drainage simulation data for current time step."""
        if self.drainage_map_name and drainage_data:
            # format map name
            suffix = str(self.record_counter).zfill(4)
            map_name = f"{self.drainage_map_name}_{suffix}"
            # write the map
            self.grass_interface.write_vector_map(drainage_data, map_name)
            # add map name and time to the list
            self.vector_drainage_maplist.append((map_name, sim_time))
            self.record_counter += 1

    def finalize(self, drainage_data: DrainageNetworkData) -> None:
        """Finalize outputs and cleanup."""
        if self.drainage_map_name and drainage_data:
            self.grass_interface.register_maps_in_stds(
                stds_title="Itzï drainage results",
                stds_name=self.drainage_map_name,
                map_list=self.vector_drainage_maplist,
                stds_type="stvds",
                t_type=self.temporal_type,
            )
