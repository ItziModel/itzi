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
from copy import deepcopy

import numpy as np

from itzi.providers.base import RasterOutputProvider, VectorOutputProvider
from itzi.data_containers import SimulationData, DrainageNetworkData


class MemoryRasterOutputProvider(RasterOutputProvider):
    """Save rasters in memory as numpy arrays."""

    def initialize(self, config: Dict) -> Self:
        """Initialize output provider with simulation configuration."""
        # user-selected map names.
        self.out_map_names = config["out_map_names"]
        self.output_maps_dict = {k: [] for k in self.out_map_names.keys()}
        return self

    def write_array(self, array: np.ndarray, map_key: str, sim_time: datetime | timedelta) -> None:
        """Save simulation data for current time step."""
        self.output_maps_dict[map_key].append((deepcopy(sim_time), array.copy()))

    def finalize(self, final_data: SimulationData) -> None:
        """Finalize outputs and cleanup."""
        pass


class MemoryVectorOutputProvider(VectorOutputProvider):
    """Save drainage simulation outputs in memory."""

    def initialize(self, config: Dict | None = None) -> Self:
        """Initialize output provider with simulation configuration."""
        self.drainage_data = []
        return self

    def write_vector(
        self, drainage_data: DrainageNetworkData, sim_time: datetime | timedelta
    ) -> None:
        """Save simulation data for current time step."""
        self.drainage_data.append((deepcopy(sim_time), deepcopy(drainage_data)))

    def finalize(self, drainage_data: DrainageNetworkData) -> None:
        """Finalize outputs and cleanup."""
        pass
