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
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from itzi_core import SimulationBuilder, SimulationConfig

import itzi.messenger as msgr
from itzi.ensemble_models import EffectiveMask
from itzi.grass_session import GrassParams, GrassSessionManager
from itzi.providers.csv_output import ExclusiveCSVMassBalanceOutputProvider
from itzi.providers.grass_input import GrassRasterInputProvider
from itzi.providers.grass_output import GrassRasterOutputProvider, GrassVectorOutputProvider

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
    ) -> None:
        from itzi.grass.utils import (
            check_output_files,
            ensure_min_version,
            get_current_mapset,
            qualify_output_id,
            resolve_input_identifier,
        )

        self.grass_params = grass_params
        self.hotstart_path = hotstart_path
        self.stats_file = stats_file
        self.effective_mask = effective_mask
        ensure_min_version()
        GrassSessionManager.ensure_temporal_initialized()
        if input_kinds is None:
            resolved_inputs = {
                key: resolve_input_identifier(name)
                for key, name in sim_config.input_map_names.items()
            }
            self.input_kinds = {key: kind for key, (_, kind) in resolved_inputs.items()}
            input_names = {key: map_id for key, (map_id, _) in resolved_inputs.items()}
        else:
            self.input_kinds = input_kinds
            input_names = sim_config.input_map_names
        mapset = get_current_mapset()
        self.sim_config = sim_config.model_copy(
            update={
                "input_map_names": input_names,
                "output_map_names": {
                    key: qualify_output_id(name, mapset)
                    for key, name in sim_config.output_map_names.items()
                },
                "drainage_output": qualify_output_id(sim_config.drainage_output, mapset)
                if sim_config.drainage_output is not None
                else None,
            }
        )
        self.g_interface: GrassInterface
        self.sim: Simulation

        msgr.display_sim_param(self.sim_config)

        output_names = list(self.sim_config.output_map_names.values())
        if self.sim_config.drainage_output is not None:
            output_names.append(self.sim_config.drainage_output)
        check_output_files(output_names)
        msgr.debug("Output files OK")

    def initialize(self) -> Self:
        from itzi.providers.grass_interface import GrassInterface

        data_type = np.float32
        self.g_interface = GrassInterface(
            start_time=self.sim_config.start_time,
            end_time=self.sim_config.end_time,
            dtype=data_type,
            region_id=self.grass_params.region,
            raster_mask_id=self.grass_params.mask,
            effective_mask=(self.effective_mask.mode, self.effective_mask.source)
            if self.effective_mask is not None
            else None,
            non_blocking_write=False,
        )
        msgr.verbose("Setting up GRASS simulation...")

        raster_input_provider = GrassRasterInputProvider(
            grass_interface=self.g_interface,
            input_map_names=self.sim_config.input_map_names,
            default_start_time=self.sim_config.start_time,
            default_end_time=self.sim_config.end_time,
            input_kinds=self.input_kinds,
        )
        raster_output_provider = GrassRasterOutputProvider(
            grass_interface=self.g_interface,
            out_map_names=self.sim_config.output_map_names,
            hmin=self.sim_config.surface_flow_parameters.hmin,
            temporal_type=self.sim_config.temporal_type,
        )
        sim_builder = (
            SimulationBuilder(self.sim_config, self.g_interface.get_npmask(), data_type)
            .with_input_provider(raster_input_provider)
            .with_raster_output_provider(raster_output_provider)
        )
        if self.sim_config.drainage_output is not None:
            vector_output_provider = GrassVectorOutputProvider(
                grass_interface=self.g_interface,
                temporal_type=self.sim_config.temporal_type,
                drainage_map_id=self.sim_config.drainage_output,
            )
            sim_builder.with_vector_output_provider(vector_output_provider)
        if self.stats_file:
            stats_provider = ExclusiveCSVMassBalanceOutputProvider(
                self.stats_file, overwrite=self.g_interface.overwrite
            )
            sim_builder.with_mass_balance_output_provider(stats_provider)
        if self.hotstart_path:
            sim_builder.with_hotstart(self.hotstart_path)
        self.sim: Simulation = sim_builder.build()
        # Initialize the simulation
        self.sim.initialize()
        return self

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
