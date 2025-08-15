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

from abc import ABC, abstractmethod
from typing import Dict, Self, TYPE_CHECKING, Union, Tuple

if TYPE_CHECKING:
    from datetime import datetime, timedelta
    import numpy as np
    from itzi.data_containers import SimulationData, DrainageNetworkData


class RasterInputProvider(ABC):
    """Abstract base class for handling raster simulation inputs."""

    @abstractmethod
    def __init__(self, config: Dict) -> None:
        pass

    @property
    @abstractmethod
    def origin(self):
        pass

    @abstractmethod
    def get_array(
        self, map_key: str, current_time: "datetime"
    ) -> Tuple["np.ndarray", "datetime", "datetime"]:
        """Take a given map key and current time
        return a numpy array associated with its start and end time
        if no map is found, return None instead of an array
        and a default start_time and end_time."""
        pass


class RasterOutputProvider(ABC):
    """Abstract base class for handling raster simulation outputs."""

    @abstractmethod
    def initialize(self, config: Dict) -> Self:
        """Initialize output provider with simulation configuration."""
        pass

    @abstractmethod
    def write_arrays(
        self, array_dict: Dict[str, "np.ndarray"], sim_time: Union["datetime", "timedelta"]
    ) -> None:
        """Write all arrays for the current time step."""
        pass

    @abstractmethod
    def finalize(self, final_data: "SimulationData") -> None:
        """Finalize outputs and cleanup."""
        pass


class VectorOutputProvider(ABC):
    """Abstract base class for drainage simulation outputs."""

    @abstractmethod
    def initialize(self, config: Dict) -> Self:
        """Initialize output provider with simulation configuration."""
        pass

    @abstractmethod
    def write_vector(
        self, drainage_data: "DrainageNetworkData", sim_time: Union["datetime", "timedelta"]
    ) -> None:
        """Write simulation data for current time step."""
        pass

    @abstractmethod
    def finalize(self, drainage_data: "DrainageNetworkData") -> None:
        """Finalize outputs and cleanup."""
        pass
