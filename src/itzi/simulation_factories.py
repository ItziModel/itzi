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

from typing import Dict, TYPE_CHECKING

import numpy as np

import itzi.rasterdomain as rasterdomain
import itzi.messenger as msgr
from itzi.simulation import Simulation
from itzi.simulation_builder import SimulationBuilder

if TYPE_CHECKING:
    from itzi.providers.grass_interface import GrassInterface
    from itzi.data_containers import SimulationConfig


def create_grass_simulation(
    sim_config: "SimulationConfig",
    grass_interface: "GrassInterface",
    dtype=np.float32,
) -> tuple[Simulation, Dict[str, rasterdomain.TimedArray]]:
    """A factory function that returns a Simulation object.
    Legacy wrapper - use SimulationBuilder directly for new code."""

    msgr.verbose("Setting up GRASS simulation...")
    from itzi.providers.grass_input import GrassRasterInputProvider
    from itzi.providers.grass_output import GrassRasterOutputProvider, GrassVectorOutputProvider

    raster_input_provider = GrassRasterInputProvider(
        {
            "grass_interface": grass_interface,
            "input_map_names": sim_config.input_map_names,
            "default_start_time": sim_config.start_time,
            "default_end_time": sim_config.end_time,
        }
    )

    raster_output_provider = GrassRasterOutputProvider(
        {
            "grass_interface": grass_interface,
            "out_map_names": sim_config.output_map_names,
            "hmin": sim_config.surface_flow_parameters.hmin,
            "temporal_type": sim_config.temporal_type,
        }
    )
    vector_output_provider = GrassVectorOutputProvider(
        {
            "grass_interface": grass_interface,
            "temporal_type": sim_config.temporal_type,
            "drainage_map_name": sim_config.drainage_output,
        }
    )
    return (
        SimulationBuilder(sim_config, grass_interface.get_npmask(), dtype)
        .with_input_provider(raster_input_provider)
        .with_raster_output_provider(raster_output_provider)
        .with_vector_output_provider(vector_output_provider)
        .build()
    )
