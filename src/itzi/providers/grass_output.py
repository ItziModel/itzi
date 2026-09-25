"""
Copyright (C) 2025-2026 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from itzi_core.providers import RasterOutputProvider, VectorOutputProvider

if TYPE_CHECKING:
    from datetime import datetime, timedelta

    from itzi_core import TemporalType
    from itzi_core.data_containers import DrainageNetworkAttributes, DrainageNetworkTopology

    from itzi.providers.grass_interface import GrassInterface


class GrassRasterOutputConfig(TypedDict):
    grass_interface: GrassInterface
    out_map_names: Mapping[str, str]
    hmin: float
    temporal_type: TemporalType


class GrassVectorOutputConfig(TypedDict):
    grass_interface: GrassInterface
    drainage_map_name: str
    temporal_type: TemporalType


class GrassRasterOutputProvider(RasterOutputProvider):
    """Write simulation outputs to GRASS."""

    def __init__(self, config: GrassRasterOutputConfig) -> None:
        """Initialize output provider with configuration."""
        self.grass_interface = config["grass_interface"]
        # user-selected map names. Keys are user-facing names
        self.out_map_names = config["out_map_names"]
        self.hmin = config["hmin"]
        self.temporal_type = config["temporal_type"]
        for stds_name in self.out_map_names.values():
            self.grass_interface.validate_output_stds_temporal_type(
                stds_name=stds_name,
                stds_type="strds",
                expected_temporal_type=self.temporal_type,
            )
        self.record_counter = {k: 0 for k in self.out_map_names}
        self.output_maplist = {k: [] for k in self.out_map_names}

    def _write_array(
        self, array: np.ndarray, map_key: str, sim_time: datetime | timedelta
    ) -> None:
        """Write simulation data for current time step."""
        suffix = str(self.record_counter[map_key]).zfill(4)
        map_name = f"{self.out_map_names[map_key]}_{suffix}"
        # write the raster
        self.grass_interface.write_raster_map(array, map_name, map_key, self.hmin)
        # Set depth values to null under the given threshold. Temporarily in gis.py
        # if map_key == "water_depth":
        #     self.grass_interface.set_null(map_name, self.hmin)
        # add map name and time to the corresponding list
        self.output_maplist[map_key].append((map_name, sim_time))
        self.record_counter[map_key] += 1

    def write_arrays(
        self, array_dict: Mapping[str, np.ndarray], sim_time: datetime | timedelta
    ) -> None:
        for arr_key, arr in array_dict.items():
            if isinstance(arr, np.ndarray):
                self._write_array(array=arr, map_key=arr_key, sim_time=sim_time)

    def finalize(self) -> None:
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


class GrassVectorOutputProvider(VectorOutputProvider):
    """Write drainage simulation outputs to GRASS."""

    def __init__(self, config: GrassVectorOutputConfig) -> None:
        """Initialize output provider with simulation configuration."""
        self.grass_interface = config["grass_interface"]
        self.drainage_map_name = config["drainage_map_name"]
        self.temporal_type = config["temporal_type"]
        self.grass_interface.validate_output_stds_temporal_type(
            stds_name=self.drainage_map_name,
            stds_type="stvds",
            expected_temporal_type=self.temporal_type,
        )

        self.record_counter = 0
        self.vector_drainage_maplist: list[tuple[str, datetime | timedelta]] = []
        self.drainage_topology: DrainageNetworkTopology | None = None

    def write_topology(self, topology: DrainageNetworkTopology) -> None:
        """Store the fixed drainage-network geometry for subsequent records."""
        if self.drainage_topology is not None:
            raise RuntimeError("Drainage topology has already been written.")
        self.drainage_topology = topology

    def write_attributes(
        self, attributes: DrainageNetworkAttributes, sim_time: datetime | timedelta
    ) -> None:
        """Write drainage simulation data for current time step."""
        if self.drainage_topology is None:
            raise RuntimeError("Drainage attributes cannot be written before topology.")
        suffix = str(self.record_counter).zfill(4)
        map_name = f"{self.drainage_map_name}_{suffix}"
        self.grass_interface.write_vector_map(self.drainage_topology, attributes, map_name)
        self.vector_drainage_maplist.append((map_name, sim_time))
        self.record_counter += 1

    def finalize(self) -> None:
        """Finalize outputs and cleanup."""
        if self.vector_drainage_maplist:
            self.grass_interface.register_maps_in_stds(
                stds_title="Itzï drainage results",
                stds_name=self.drainage_map_name,
                map_list=self.vector_drainage_maplist,
                stds_type="stvds",
                t_type=self.temporal_type,
            )
