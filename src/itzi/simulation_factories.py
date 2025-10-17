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
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

import itzi.rasterdomain as rasterdomain
import itzi.messenger as msgr
from itzi.simulation import Simulation
from itzi.simulation_builder import SimulationBuilder

if TYPE_CHECKING:
    from itzi.providers.grass_interface import GrassInterface
    from itzi.data_containers import SimulationConfig
    import icechunk


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


def create_icechunk_simulation(
    sim_config: "SimulationConfig",
    arr_mask: ArrayLike,
    input_icechunk_storage: "icechunk.Storage",
    output_icechunk_storage: "icechunk.Storage",
    input_icechunk_group: str = "main",
    output_dir: Path = Path("."),
    dtype: DTypeLike = np.float32,
) -> tuple[Simulation, Dict[str, rasterdomain.TimedArray]]:
    """A factory function that returns a Simulation object with Icechunk raster backend and Parquet vector backend."""
    msgr.verbose("Setting up models...")
    try:
        import pyproj
        from itzi.providers.icechunk_input import (
            IcechunkRasterInputProvider,
            IcechunkRasterInputConfig,
        )
        from itzi.providers.icechunk_output import IcechunkRasterOutputProvider
        from itzi.providers.geoparquet_output import ParquetVectorOutputProvider
    except ImportError:
        raise ImportError(
            "To use the Icechunk backend, install itzi with: "
            "'uv tool install itzi[cloud]' "
            "or 'pip install itzi[cloud]'"
        )

    # Set up raster input provider
    raster_input_provider_config: IcechunkRasterInputConfig = {
        "icechunk_storage": input_icechunk_storage,
        "icechunk_group": input_icechunk_group,
        "input_map_names": sim_config.input_map_names,
        "simulation_start_time": sim_config.start_time,
        "simulation_end_time": sim_config.end_time,
    }
    raster_input_provider = IcechunkRasterInputProvider(config=raster_input_provider_config)
    domain_data = raster_input_provider.get_domain_data()

    # Set up raster output provider
    # Generate coordinate arrays from domain_data
    coords = domain_data.get_coordinates()
    x_coords = coords["x"]
    y_coords = coords["y"]
    crs = pyproj.CRS.from_wkt(domain_data.crs_wkt)
    raster_output_provider = IcechunkRasterOutputProvider(
        {
            "out_map_names": sim_config.output_map_names,
            "crs": crs,
            "x_coords": x_coords,
            "y_coords": y_coords,
            "icechunk_storage": output_icechunk_storage,
        }
    )

    # Set up vector output provider
    vector_output_provider = ParquetVectorOutputProvider(
        {
            "crs": crs,
            "output_dir": output_dir,
            "drainage_map_name": sim_config.drainage_output,
        }
    )
    return (
        SimulationBuilder(sim_config, arr_mask, dtype)
        .with_input_provider(raster_input_provider)
        .with_raster_output_provider(raster_output_provider)
        .with_vector_output_provider(vector_output_provider)
        .build()
    )
