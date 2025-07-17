from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional
import numpy as np


class RasterOutputProvider(ABC):
    """Abstract base class for handling raster simulation outputs."""

    @abstractmethod
    def write_array(self, arr: np.ndarray, map_name: str, map_key: str) -> None:
        """Write simulation data for current time step."""
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Finalize outputs and cleanup."""
        pass


class VectorOutputProvider(ABC):
    """Abstract base class for drainage simulation outputs."""

    @abstractmethod
    def write_vector(self, arr: np.ndarray, map_name: str, map_key: str) -> None:
        """Write simulation data for current time step."""
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Finalize outputs and cleanup."""
        pass
