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
from typing import TYPE_CHECKING

import numpy as np
from itzi_core import OUTPUT_ARRAY_KEYS
from itzi_core.providers import RasterOutputProvider, VectorOutputProvider

from itzi.grass.names import derived_record_name

if TYPE_CHECKING:
    from datetime import datetime, timedelta

    from itzi_core import TemporalType
    from itzi_core.data_containers import DrainageNetworkAttributes, DrainageNetworkTopology

    from itzi.grass.interface import GrassInterface


class GrassRasterOutputProvider(RasterOutputProvider):
    """Write simulation outputs to GRASS."""

    def __init__(
        self,
        grass_interface: GrassInterface,
        out_map_names: Mapping[str, str],
        hmin: float,
        temporal_type: TemporalType,
    ) -> None:
        invalid_output_keys = sorted(set(out_map_names) - OUTPUT_ARRAY_KEYS)
        if invalid_output_keys:
            raise ValueError(
                f"out_map_names contains invalid input keys: {', '.join(invalid_output_keys)}"
            )

        self.grass_interface = grass_interface
        # Dataset IDs qualified in the current mapset. Keys are output variables.
        self.out_map_names = out_map_names
        self.hmin = hmin
        self.temporal_type = temporal_type
        for stds_id in self.out_map_names.values():
            self.grass_interface.validate_output_stds_temporal_type(
                stds_id=stds_id,
                stds_type="strds",
                expected_temporal_type=self.temporal_type,
            )
        self.record_counter = {k: 0 for k in self.out_map_names}
        self.output_maplist = {k: [] for k in self.out_map_names}

    def _write_array(
        self, array: np.ndarray, map_key: str, sim_time: datetime | timedelta
    ) -> None:
        map_id = derived_record_name(self.out_map_names[map_key], self.record_counter[map_key])
        # write the raster
        self.grass_interface.write_raster_map(array, map_id, map_key, self.hmin)
        # Set depth values to null under the given threshold. Temporarily in gis.py
        # if map_key == "water_depth":
        #     self.grass_interface.set_null(map_name, self.hmin)
        # add map name and time to the corresponding list
        self.output_maplist[map_key].append((map_id, sim_time))
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
            strds_id = self.out_map_names[map_key]
            self.grass_interface.register_maps_in_stds(
                map_key, strds_id, lst, "strds", self.temporal_type
            )


class GrassVectorOutputProvider(VectorOutputProvider):
    """Write drainage simulation outputs to GRASS."""

    def __init__(
        self,
        grass_interface: GrassInterface,
        drainage_map_id: str,
        temporal_type: TemporalType,
    ) -> None:
        self.grass_interface = grass_interface
        self.drainage_map_id = drainage_map_id
        self.temporal_type = temporal_type
        self.grass_interface.validate_output_stds_temporal_type(
            stds_id=self.drainage_map_id,
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
        map_id = derived_record_name(self.drainage_map_id, self.record_counter)
        self.grass_interface.write_vector_map(self.drainage_topology, attributes, map_id)
        self.vector_drainage_maplist.append((map_id, sim_time))
        self.record_counter += 1

    def finalize(self) -> None:
        """Finalize outputs and cleanup."""
        if self.vector_drainage_maplist:
            self.grass_interface.register_maps_in_stds(
                stds_title="Itzï drainage results",
                stds_id=self.drainage_map_id,
                map_list=self.vector_drainage_maplist,
                stds_type="stvds",
                t_type=self.temporal_type,
            )
