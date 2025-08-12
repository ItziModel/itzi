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

from typing import Dict, Self
from datetime import datetime, timedelta

import numpy as np
import icechunk

from itzi.providers.base import RasterOutputProvider
from itzi.data_containers import SimulationData


class IcechunkRasterOutputProvider(RasterOutputProvider):
    """Save raster results in an Icechunk store."""

    def initialize(self, config: Dict) -> Self:
        """Create a repo in case it does not exists already"""
        self.out_map_names = config["out_map_names"]
        storage = config["icechunk_storage"]
        try:
            self.repo = icechunk.Repository.open(storage)
        except icechunk.IcechunkError as e:
            if "repository doesn't exist" in str(e):
                self.repo = icechunk.Repository.create(storage)
            else:
                raise

    def write_array(self, array: np.ndarray, map_key: str, sim_time: datetime | timedelta) -> None:
        pass

    def write_arrays(
        self, array_dict: Dict[str, np.ndarray], sim_time: datetime | timedelta
    ) -> None:
        pass

    def finalize(self, final_data: SimulationData) -> None:
        """Finalize outputs and cleanup."""
        pass
