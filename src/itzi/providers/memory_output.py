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
from typing import Mapping, TypedDict
from copy import deepcopy

import numpy as np

from itzi.providers.base import RasterOutputProvider, VectorOutputProvider
from itzi.data_containers import SimulationData, DrainageNetworkData


class MemoryRasterOutputConfig(TypedDict):
    out_map_names: Mapping[str, str]


class MemoryVectorOutputConfig(TypedDict):
    pass


class MemoryRasterOutputProvider(RasterOutputProvider):
    """Save rasters in memory as numpy arrays."""

    def __init__(self, config: MemoryRasterOutputConfig) -> None:
        """Initialize output provider with simulation configuration."""
        # user-selected map names.
        self.out_map_names = config["out_map_names"]
        self.output_maps_dict = {k: [] for k in self.out_map_names.keys()}

    def write_arrays(
        self, array_dict: Mapping[str, np.ndarray], sim_time: datetime | timedelta
    ) -> None:
        for arr_key, arr in array_dict.items():
            if isinstance(arr, np.ndarray):
                self.output_maps_dict[arr_key].append((deepcopy(sim_time), arr.copy()))

    def finalize(self, final_data: SimulationData) -> None:
        """Finalize outputs and cleanup."""
        pass


class MemoryVectorOutputProvider(VectorOutputProvider):
    """Save drainage simulation outputs in memory."""

    def __init__(self, config: MemoryVectorOutputConfig | None = None) -> None:
        """Initialize output provider with simulation configuration."""
        self.drainage_data = []

    def write_vector(
        self, drainage_data: DrainageNetworkData, sim_time: datetime | timedelta
    ) -> None:
        """Save simulation data for current time step."""
        self.drainage_data.append((deepcopy(sim_time), deepcopy(drainage_data)))

    def finalize(self, drainage_data: DrainageNetworkData) -> None:
        """Finalize outputs and cleanup."""
        pass
