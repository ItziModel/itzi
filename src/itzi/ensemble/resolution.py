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

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from itzi_core import DomainData, SimulationConfig, SurfaceFlowParameters
from pydantic import ValidationError

from itzi.ensemble.models import (
    ArtifactSummary,
    EffectiveMask,
    EnsembleError,
    ExpandedSimulation,
    ResolvedSimulation,
    ValidationFailure,
    format_duration,
    render_template,
)
from itzi.grass.session import GrassParams, GrassSessionManager
from itzi.grass.names import derived_drainage_table_names, derived_record_name


def resolve_ensemble(
    simulations: tuple[ExpandedSimulation, ...],
) -> tuple[ResolvedSimulation | ValidationFailure, ...]:
    """Resolve one ensemble in one GRASS session, retaining member-local failures."""
    if not simulations:
        return ()
    requested_params = _requested_grass_params(simulations[0])

    with GrassSessionManager(requested_params):
        from itzi.grass.utils import active_grass_params, read_domain, resolve_effective_mask

        actual_params = active_grass_params(requested_params)
        domain = read_domain(simulations[0].domain.region)
        effective_mask = resolve_effective_mask(simulations[0].domain.mask)
        input_cache: dict[str, tuple[str, Literal["raster", "strds"]] | Exception] = {}
        swmm_cache: dict[Path, str] = {}
        results: list[ResolvedSimulation | ValidationFailure] = []
        for expanded in simulations:
            try:
                input_names, input_kinds = _resolve_inputs(
                    dict(expanded.input_maps) | dict(expanded.infiltration.input_maps),
                    cache=input_cache,
                )
                swmm_path, swmm_digest = _resolve_swmm_input(expanded, swmm_cache)
                results.append(
                    _build_resolved_simulation(
                        expanded,
                        actual_params,
                        domain,
                        effective_mask,
                        input_names,
                        input_kinds,
                        swmm_path,
                        swmm_digest,
                    )
                )
            except Exception as error:
                results.append(
                    ValidationFailure(
                        coordinates=expanded.coordinates,
                        phase="input_resolution",
                        detail=f"{type(error).__name__}: {error}",
                    )
                )
        return tuple(results)


def _requested_grass_params(expanded: ExpandedSimulation) -> GrassParams:
    context = expanded.domain.grass
    if context is None or context.database is None:
        return GrassParams(region=expanded.domain.region, mask=expanded.domain.mask)
    return GrassParams(
        grassdata=str(Path(context.database).expanduser().resolve()),
        location=context.project,
        mapset=context.mapset,
        region=expanded.domain.region,
        mask=expanded.domain.mask,
        grass_bin=context.executable or "grass",
    )


def _build_resolved_simulation(
    expanded: ExpandedSimulation,
    actual_params: GrassParams,
    domain: DomainData,
    effective_mask: EffectiveMask,
    input_names: dict[str, str],
    input_kinds: dict[str, Literal["raster", "strds"]],
    swmm_path: Path | None,
    swmm_digest: str | None,
) -> ResolvedSimulation:
    assert actual_params.mapset is not None
    simulation_id, normalized_payload = _simulation_identity(
        expanded,
        actual_params,
        domain,
        effective_mask,
        input_names,
        swmm_digest,
    )
    artifacts = _render_artifacts(expanded, simulation_id, actual_params.mapset)
    simulation_config = _build_simulation_config(
        expanded,
        input_names,
        artifacts,
        swmm_path,
    )
    return ResolvedSimulation(
        simulation_id=simulation_id,
        coordinates=expanded.coordinates,
        grass_params=actual_params,
        domain_data=domain,
        effective_mask=effective_mask,
        input_kinds=tuple(sorted(input_kinds.items())),
        simulation_config=simulation_config,
        artifacts=artifacts,
        normalized_payload=normalized_payload,
    )


def _resolve_inputs(
    inputs: dict[str, str],
    *,
    cache: dict[str, tuple[str, Literal["raster", "strds"]] | Exception] | None = None,
) -> tuple[dict[str, str], dict[str, Literal["raster", "strds"]]]:
    from itzi.grass.utils import resolve_input_identifier

    resolved: dict[str, str] = {}
    kinds: dict[str, Literal["raster", "strds"]] = {}
    for key, identifier in inputs.items():
        result = cache.get(identifier) if cache is not None else None
        if result is None:
            try:
                result = resolve_input_identifier(identifier)
            except Exception as error:
                if cache is not None:
                    cache[identifier] = error
                raise
            if cache is not None:
                cache[identifier] = result
        if isinstance(result, Exception):
            raise result
        source, kind = result
        resolved[key] = source
        kinds[key] = kind
    return resolved, kinds


def _resolve_swmm_input(
    expanded: ExpandedSimulation, cache: dict[Path, str]
) -> tuple[Path | None, str | None]:
    if expanded.drainage is None:
        return None, None
    drainage = dict(expanded.drainage)
    configured = Path(str(drainage["swmm_input"])).expanduser()
    path = configured if configured.is_absolute() else expanded.source.path.parent / configured
    path = path.resolve()
    if not path.is_file():
        raise EnsembleError(f"SWMM input file <{path}> not found")
    if path in cache:
        return path, cache[path]
    digest = f"blake2b:{hashlib.blake2b(path.read_bytes(), digest_size=32).hexdigest()}"
    cache[path] = digest
    return path, digest


def _simulation_identity(
    expanded: ExpandedSimulation,
    grass_params: GrassParams,
    domain: DomainData,
    effective_mask: EffectiveMask,
    input_names: dict[str, str],
    swmm_digest: str | None,
) -> tuple[str, str]:
    """Build the versioned identity payload after all source normalization."""
    drainage = dict(expanded.drainage) if expanded.drainage is not None else None
    if drainage is not None:
        assert swmm_digest is not None
        drainage["swmm_digest"] = swmm_digest
        drainage["swmm_input"] = str(
            (expanded.source.path.parent / str(drainage["swmm_input"])).resolve()
        )
    time_payload: dict[str, str | None] = {
        "temporal_type": str(expanded.time.temporal_type),
        "start": expanded.time.start.isoformat() if expanded.time.start is not None else None,
        "end": expanded.time.end.isoformat() if expanded.time.end is not None else None,
        "duration": format_duration(expanded.time.duration),
        "record_step": format_duration(expanded.time.record_step),
    }
    payload = {
        "identity_version": 1,
        "grass": {
            "database": grass_params.grassdata,
            "project": grass_params.location,
            "mapset": grass_params.mapset,
        },
        "domain": domain.model_dump(),
        "mask": {"mode": effective_mask.mode, "source": effective_mask.source},
        "time": time_payload,
        "inputs": input_names,
        "infiltration": {"model": str(expanded.infiltration.model)},
        "options": dict(expanded.options),
        "drainage": drainage,
    }
    canonical = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
    digest = hashlib.blake2b(canonical.encode(), digest_size=4).hexdigest()
    return f"sim-{digest}", canonical


def _render_artifacts(
    expanded: ExpandedSimulation,
    simulation_id: str,
    mapset: str,
) -> ArtifactSummary:
    output_map_names: dict[str, str] = {}
    if expanded.outputs.raster_prefix is not None:
        prefix = render_template(
            expanded.outputs.raster_prefix,
            ensemble=expanded.ensemble_id,
            simulation=simulation_id,
        )
        output_map_names = {
            variable: f"{prefix}_{variable}" for variable in expanded.outputs.raster_variables
        }
    drainage_output = (
        render_template(
            expanded.outputs.drainage_dataset,
            ensemble=expanded.ensemble_id,
            simulation=simulation_id,
        )
        if expanded.outputs.drainage_dataset is not None
        else None
    )
    stats_file = (
        _source_relative_path(
            render_template(
                expanded.outputs.statistics_file,
                ensemble=expanded.ensemble_id,
                simulation=simulation_id,
            ),
            expanded.source.path.parent,
        )
        if expanded.outputs.statistics_file is not None
        else None
    )
    _validate_output_names(
        output_map_names,
        drainage_output,
        _last_record_index(expanded.time.duration, expanded.time.record_step),
    )
    _validate_protected_file_output(stats_file, (expanded.source.path, _drainage_path(expanded)))
    return ArtifactSummary(
        tuple(sorted((key, f"{name}@{mapset}") for key, name in output_map_names.items())),
        f"{drainage_output}@{mapset}" if drainage_output is not None else None,
        stats_file,
    )


def _source_relative_path(value: str, source_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else source_dir / path).resolve()


def _drainage_path(expanded: ExpandedSimulation) -> Path | None:
    if expanded.drainage is None:
        return None
    return _source_relative_path(
        str(dict(expanded.drainage)["swmm_input"]), expanded.source.path.parent
    )


def _validate_protected_file_output(output: Path | None, protected: Iterable[Path | None]) -> None:
    if output is None:
        return
    for path in protected:
        if path is None:
            continue
        source = path.resolve()
        if output == source or (output.exists() and source.exists() and output.samefile(source)):
            raise EnsembleError(f"output file <{output}> aliases protected input <{source}>")


def _last_record_index(duration: timedelta, record_step: timedelta) -> int:
    """Return the final output index, including the initial record at index zero."""
    return -(-duration // record_step)


def _validate_output_names(
    output_map_names: dict[str, str], drainage_output: str | None, last_record_index: int
) -> None:
    from itzi.grass.utils import is_clean_name

    names = list(output_map_names.values())
    if drainage_output is not None:
        names.append(drainage_output)
    for name in names:
        if "@" in name:
            raise EnsembleError(f"GRASS output names must be unqualified: {name!r}")
        if name == "MASK":
            raise EnsembleError("MASK cannot be used as an output name")
        if not is_clean_name(name):
            raise EnsembleError(f"invalid GRASS output name {name!r}")
        child = derived_record_name(name, last_record_index)
        if child == "MASK" or not is_clean_name(child):
            raise EnsembleError(f"invalid derived GRASS output name {child!r}")
    if drainage_output is not None:
        child = derived_record_name(drainage_output, last_record_index)
        for table_name in derived_drainage_table_names(child):
            if not is_clean_name(table_name):
                raise EnsembleError(f"invalid derived drainage table name {table_name!r}")


def _build_simulation_config(
    expanded: ExpandedSimulation,
    input_names: dict[str, str],
    artifacts: ArtifactSummary,
    swmm_path: Path | None,
) -> SimulationConfig:
    start = expanded.time.start
    end = expanded.time.end
    if start is None:
        start = datetime.min  # noqa: DTZ901  # Core's established relative-time adapter.
        end = start + expanded.time.duration
    assert end is not None
    options = dict(expanded.options)
    surface_options = {key: value for key, value in options.items() if key != "dtinf"}
    # Create the dict first to satisfy the type checker
    config_data = {
        "start_time": start,
        "end_time": end,
        "record_step": expanded.time.record_step,
        "temporal_type": expanded.time.temporal_type,
        "hotstart_config": None,
        "input_map_names": input_names,
        "output_map_names": dict(artifacts.output_map_names),
        "surface_flow_parameters": SurfaceFlowParameters(**surface_options),
        "dtinf": options["dtinf"],
        "infiltration_model": expanded.infiltration.model,
        "swmm_inp": swmm_path,
        "drainage_output": artifacts.drainage_output,
    }
    if expanded.drainage is not None:
        config_data.update((key, value) for key, value in expanded.drainage if key != "swmm_input")
    try:
        return SimulationConfig.model_validate(config_data)
    except ValidationError as error:
        details = "; ".join(item["msg"] for item in error.errors(include_url=False))
        raise EnsembleError(details) from error


def validate_resolved_ensemble(
    simulations: tuple[ResolvedSimulation, ...], protected_inputs: tuple[Path, ...] = ()
) -> None:
    """Reject identity and artifact collisions among resolved ensemble members."""
    identities: dict[str, str] = {}
    artifacts: dict[str, str] = {}
    ensemble_sources: set[str] = set()
    protected_paths = {path.resolve() for path in protected_inputs}
    for simulation in simulations:
        ensemble_sources.update(simulation.simulation_config.input_map_names.values())
        if simulation.effective_mask.source is not None:
            ensemble_sources.add(simulation.effective_mask.source)
        if simulation.simulation_config.swmm_inp is not None:
            protected_paths.add(simulation.simulation_config.swmm_inp.resolve())
    for simulation in simulations:
        existing_payload = identities.get(simulation.simulation_id)
        if existing_payload is not None:
            if existing_payload == simulation.normalized_payload:
                raise EnsembleError(f"duplicate resolved simulation {simulation.simulation_id}")
            raise EnsembleError(f"simulation ID digest collision {simulation.simulation_id}")
        identities[simulation.simulation_id] = simulation.normalized_payload

        last_record_index = _last_record_index(
            simulation.simulation_config.end_time - simulation.simulation_config.start_time,
            simulation.simulation_config.record_step,
        )
        for _, name in simulation.artifacts.output_map_names:
            _claim_artifact(artifacts, name, simulation.simulation_id)
            _reject_output_source_alias(name, ensemble_sources)
            for index in range(last_record_index + 1):
                child = derived_record_name(name, index)
                _claim_artifact(artifacts, child, simulation.simulation_id)
                _reject_output_source_alias(child, ensemble_sources)
        if simulation.artifacts.drainage_output is not None:
            drainage_name = simulation.artifacts.drainage_output
            _claim_artifact(artifacts, drainage_name, simulation.simulation_id)
            _reject_output_source_alias(drainage_name, ensemble_sources)
            for index in range(last_record_index + 1):
                child = derived_record_name(drainage_name, index)
                _claim_artifact(artifacts, child, simulation.simulation_id)
                _reject_output_source_alias(child, ensemble_sources)
                for table_name in derived_drainage_table_names(child):
                    _claim_artifact(artifacts, f"table:{table_name}", simulation.simulation_id)
        if simulation.artifacts.statistics_file is not None:
            _validate_protected_file_output(simulation.artifacts.statistics_file, protected_paths)
            _claim_artifact(
                artifacts,
                str(simulation.artifacts.statistics_file),
                simulation.simulation_id,
            )


def _claim_artifact(artifacts: dict[str, str], artifact: str, simulation_id: str) -> None:
    owner = artifacts.get(artifact)
    if owner is None:
        artifacts[artifact] = simulation_id
        return
    if owner == simulation_id:
        raise EnsembleError(f"output artifact {artifact!r} is used more than once")
    raise EnsembleError(
        f"output artifact {artifact!r} is shared by simulations {owner} and {simulation_id}"
    )


def _reject_output_source_alias(map_id: str, sources: set[str]) -> None:
    if map_id in sources:
        raise EnsembleError(f"output {map_id!r} aliases an ensemble input or mask")


def verify_resolved_simulation(simulation: ResolvedSimulation) -> None:
    """Re-resolve worker facts and reject a changed GRASS environment."""
    from itzi.grass.utils import active_grass_params, read_domain, resolve_effective_mask

    actual = active_grass_params(simulation.grass_params)
    if actual != simulation.grass_params:
        raise EnsembleError("GRASS context changed after resolution")
    domain = read_domain(simulation.grass_params.region)
    if domain != simulation.domain_data:
        raise EnsembleError("GRASS domain changed after resolution")
    mask = resolve_effective_mask(simulation.grass_params.mask)
    if mask != simulation.effective_mask:
        raise EnsembleError("effective GRASS mask changed after resolution")
    _verify_input_kinds(simulation.simulation_config.input_map_names, dict(simulation.input_kinds))


def _verify_input_kinds(
    inputs: dict[str, str], expected: dict[str, Literal["raster", "strds"]]
) -> None:
    resolved, kinds = _resolve_inputs(inputs)
    if resolved != inputs or kinds != expected:
        raise EnsembleError("resolved GRASS input sources changed after resolution")
