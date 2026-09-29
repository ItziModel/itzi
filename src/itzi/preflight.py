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

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from itzi.ensemble_models import EnsembleError
from itzi.manifest import validate_file_destination
from itzi.providers.grass_input import resolve_input_map_lists
from itzi.providers.grass_output import derived_record_name
from itzi.resolution import _last_record_index, verify_resolved_simulation

if TYPE_CHECKING:
    from itzi.ensemble_models import ResolvedSimulation
    from itzi.providers.grass_interface import GrassInterface


def preflight_simulation(simulation: ResolvedSimulation) -> None:
    """Validate one resolved simulation without creating user artifacts."""
    from itzi.providers.grass_interface import GrassInterface

    verify_resolved_simulation(simulation)
    GrassInterface.ensure_min_version()
    config = simulation.simulation_config
    interface = GrassInterface(
        start_time=config.start_time,
        end_time=config.end_time,
        dtype=np.float32,
        region_id=simulation.grass_params.region,
        raster_mask_id=simulation.grass_params.mask,
        effective_mask=(simulation.effective_mask.mode, simulation.effective_mask.source),
        non_blocking_write=False,
    )
    try:
        _validate_inputs(simulation, interface)
        _validate_grass_outputs(simulation, interface)
        if simulation.artifacts.statistics_file is not None:
            validate_file_destination(
                simulation.artifacts.statistics_file,
                overwrite=interface.overwrite,
                description="statistics file",
            )
    finally:
        interface.cleanup()


def _validate_inputs(simulation: ResolvedSimulation, interface: GrassInterface) -> None:
    config = simulation.simulation_config
    mask = interface.get_npmask()
    if np.all(mask):
        raise EnsembleError("effective mask excludes the complete simulation domain")

    map_lists = resolve_input_map_lists(
        interface,
        config.input_map_names,
        config.start_time,
        config.end_time,
        dict(simulation.input_kinds),
    )
    for key in ("ground_elevation", "friction"):
        for map_data in map_lists[key] or ():
            array = interface.read_raster_map(map_data.id)
            active = array[~mask]
            if not np.any(np.isfinite(active)):
                raise EnsembleError(
                    f"input map <{key}> contains only NULL/NaN cells inside the active domain"
                )


def _validate_grass_outputs(simulation: ResolvedSimulation, interface: GrassInterface) -> None:
    config = simulation.simulation_config
    last_index = _last_record_index(
        config.end_time - config.start_time,
        config.record_step,
    )

    mapset = interface.get_current_mapset()

    def reject_existing(name: str, exists: bool, kind: str) -> None:
        if exists and not interface.overwrite:
            raise EnsembleError(f"{kind} <{name}> already exists")

    for name in config.output_map_names.values():
        if not interface.overwrite:
            reject_existing(
                name,
                interface.raster_exists(name, mapset) or interface.stds_exists(name, "strds"),
                "raster output",
            )
        interface.validate_output_stds_temporal_type(name, "strds", config.temporal_type)
        if not interface.overwrite:
            for index in range(last_index + 1):
                child = derived_record_name(name, index)
                reject_existing(child, interface.raster_exists(child, mapset), "raster output")

    if config.drainage_output is None:
        return
    name = config.drainage_output
    if not interface.overwrite:
        reject_existing(
            name,
            interface.vector_exists(name, mapset) or interface.stds_exists(name, "stvds"),
            "drainage output",
        )
    interface.validate_output_stds_temporal_type(name, "stvds", config.temporal_type)
    if interface.overwrite:
        return

    for index in range(last_index + 1):
        child = derived_record_name(name, index)
        reject_existing(child, interface.vector_exists(child, mapset), "drainage output")
