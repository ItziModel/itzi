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

from itzi.ensemble.models import EnsembleError
from itzi.ensemble.resolution import _last_record_index, verify_resolved_simulation
from itzi.grass.names import derived_record_name
from itzi.manifest import validate_file_destination

if TYPE_CHECKING:
    from itzi.ensemble.models import ResolvedSimulation
    from itzi.grass.interface import GrassInterface


def preflight_simulation(simulation: ResolvedSimulation) -> None:
    """Validate one resolved simulation without creating user artifacts."""
    from itzi.grass.interface import GrassInterface
    from itzi.grass.utils import ensure_min_version

    verify_resolved_simulation(simulation)
    ensure_min_version()
    config = simulation.simulation_config
    interface = GrassInterface(
        start_time=config.start_time,
        end_time=config.end_time,
        dtype=np.float32,
        region_id=simulation.grass_params.region,
        raster_mask_id=simulation.grass_params.mask,
        effective_mask=(simulation.effective_mask.mode, simulation.effective_mask.source),
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
    from itzi.grass.utils import resolve_input_map_lists

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
    from itzi.grass.utils import (
        get_current_mapset,
        output_name,
        raster_exists,
        stds_exists,
        vector_exists,
    )

    config = simulation.simulation_config
    last_index = _last_record_index(
        config.end_time - config.start_time,
        config.record_step,
    )

    mapset = get_current_mapset()

    def reject_existing(name: str, exists: bool, kind: str) -> None:
        if exists and not interface.overwrite:
            raise EnsembleError(f"{kind} <{name}> already exists")

    for map_id in config.output_map_names.values():
        name = output_name(map_id)
        if not interface.overwrite:
            reject_existing(
                map_id,
                raster_exists(name, mapset) or stds_exists(map_id, "strds"),
                "raster output",
            )
        interface.validate_output_stds_temporal_type(map_id, "strds", config.temporal_type)
        if not interface.overwrite:
            for index in range(last_index + 1):
                child = derived_record_name(map_id, index)
                reject_existing(child, raster_exists(output_name(child), mapset), "raster output")

    if config.drainage_output is None:
        return
    map_id = config.drainage_output
    name = output_name(map_id)
    if not interface.overwrite:
        reject_existing(
            map_id,
            vector_exists(name, mapset) or stds_exists(map_id, "stvds"),
            "drainage output",
        )
    interface.validate_output_stds_temporal_type(map_id, "stvds", config.temporal_type)
    if interface.overwrite:
        return

    for index in range(last_index + 1):
        child = derived_record_name(map_id, index)
        reject_existing(child, vector_exists(output_name(child), mapset), "drainage output")
