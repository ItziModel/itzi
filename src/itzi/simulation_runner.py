"""
Copyright (C) 2026 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

from collections.abc import Mapping
from datetime import datetime
from time import monotonic
from typing import TYPE_CHECKING, Literal

import numpy as np
from itzi_core import SimulationBuilder, SimulationConfig
from itzi_core.providers.csv_mass_balance_output import CSVMassBalanceOutputProvider

import itzi.messenger as msgr
from itzi.ensemble_models import EffectiveMask
from itzi.grass_session import GrassParams, GrassSessionManager

if TYPE_CHECKING:
    from itzi_core import Simulation

    from itzi.providers.grass_interface import GrassInterface


class SimulationRunner:
    """Provide the necessary tools to run one simulation.
    Must be instantiated within an active GRASS session."""

    def __init__(
        self,
        sim_config: SimulationConfig,
        grass_params: GrassParams,
        hotstart_path: str | None = None,
        stats_file: str | None = None,
        effective_mask: EffectiveMask | None = None,
        input_kinds: Mapping[str, Literal["raster", "strds"]] | None = None,
        exclusive_stats: bool = False,
    ) -> None:
        self.g_interface: GrassInterface
        self.sim: Simulation

        # display parameters (if verbose)
        msgr.display_sim_param(sim_config)

        # Check GRASS version
        from itzi.providers import grass_interface

        grass_interface.GrassInterface.ensure_min_version()
        GrassSessionManager.ensure_temporal_initialized()
        msgr.debug("GRASS session set")

        # return error if output files exist
        output_names = list(sim_config.output_map_names.values())
        if sim_config.drainage_output is not None:
            output_names.append(sim_config.drainage_output)
        grass_interface.check_output_files(output_names)
        msgr.debug("Output files OK")

        data_type = np.float32
        # Create the grass_interface object
        self.g_interface = grass_interface.GrassInterface(
            start_time=sim_config.start_time,
            end_time=sim_config.end_time,
            dtype=data_type,
            region_id=grass_params.region,
            raster_mask_id=grass_params.mask,
            effective_mask=(effective_mask.mode, effective_mask.source)
            if effective_mask is not None
            else None,
            non_blocking_write=False,
        )
        # Create Simulation with GRASS backend
        msgr.verbose("Setting up GRASS simulation...")
        from itzi.providers.grass_input import GrassRasterInputProvider
        from itzi.providers.grass_output import (
            GrassRasterOutputProvider,
            GrassVectorOutputProvider,
        )

        raster_input_provider = GrassRasterInputProvider(
            grass_interface=self.g_interface,
            input_map_names=sim_config.input_map_names,
            default_start_time=sim_config.start_time,
            default_end_time=sim_config.end_time,
            input_kinds=input_kinds,
        )
        raster_output_provider = GrassRasterOutputProvider(
            grass_interface=self.g_interface,
            out_map_names=sim_config.output_map_names,
            hmin=sim_config.surface_flow_parameters.hmin,
            temporal_type=sim_config.temporal_type,
        )
        sim_builder = (
            SimulationBuilder(sim_config, self.g_interface.get_npmask(), data_type)
            .with_input_provider(raster_input_provider)
            .with_raster_output_provider(raster_output_provider)
        )
        if sim_config.drainage_output is not None:
            vector_output_provider = GrassVectorOutputProvider(
                grass_interface=self.g_interface,
                temporal_type=sim_config.temporal_type,
                drainage_map_name=sim_config.drainage_output,
            )
            sim_builder.with_vector_output_provider(vector_output_provider)
        if stats_file:
            if exclusive_stats:
                from itzi.providers.csv_output import ExclusiveCSVMassBalanceOutputProvider

                stats_provider = ExclusiveCSVMassBalanceOutputProvider(
                    stats_file, overwrite=self.g_interface.overwrite
                )
            else:
                stats_provider = CSVMassBalanceOutputProvider(stats_file)
            sim_builder.with_mass_balance_output_provider(stats_provider)
        if hotstart_path:
            sim_builder.with_hotstart(hotstart_path)
        self.sim: Simulation = sim_builder.build()
        # Initialize the simulation
        self.sim.initialize()

    def run(self):
        """Run a full simulation"""
        sim_start_time = datetime.now()
        last_progress_update = monotonic() - 0.5
        msgr.verbose("Starting time-stepping...")
        while self.sim.sim_time < self.sim.end_time:
            # display advance of simulation
            now = monotonic()
            if now - last_progress_update >= 0.2:
                msgr.percent(
                    self.sim.start_time,
                    self.sim.end_time,
                    self.sim.sim_time,
                    sim_start_time,
                )
                last_progress_update = now
            # step models
            self.step()
        return self

    def finalize(self):
        """Tear down the simulation and return to previous state."""
        self.sim.finalize()
        # Cleanup the grass interface object
        if hasattr(self, "g_interface"):
            self.g_interface.finalize()
            self.g_interface.cleanup()
        return self

    def step(self):
        """Do one simulation step."""
        self.sim.update()
        return self

    @property
    def origin(self):
        return (self.sim.domain_data.north, self.sim.domain_data.west)

    def __del__(self):
        # Cleanup the grass interface object
        if hasattr(self, "g_interface"):
            self.g_interface.finalize()
            self.g_interface.cleanup()
