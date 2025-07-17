from itzi.providers.base import OutputProvider
from itzi.gis import Igis

class GrassOutputProvider(OutputProvider):
    """Abstract base class for handling simulation outputs."""

    @abstractmethod
    def initialize(self, simulation_config: Dict) -> None:
        """Initialize output provider with simulation configuration."""
        pass
    
    @abstractmethod
    def write_step(self, simulation_data: SimulationData) -> None:
        """Write simulation data for current time step."""
        pass
    
    @abstractmethod
    def finalize(self, final_data: SimulationData) -> None:
        """Finalize outputs and cleanup."""
        pass
