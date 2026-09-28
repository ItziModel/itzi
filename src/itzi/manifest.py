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

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import cast

import yaml

from itzi.ensemble_models import (
    EnsembleError,
    ExpandedEnsemble,
    JsonValue,
    ResolvedSimulation,
    ValidationFailure,
    render_template,
)


def _manifest_path(ensemble: ExpandedEnsemble) -> Path:
    if ensemble.manifest_template is not None:
        rendered = render_template(
            ensemble.manifest_template, ensemble=ensemble.ensemble_id, simulation=None
        )
        path = Path(rendered).expanduser()
        return (path if path.is_absolute() else ensemble.source.path.parent / path).resolve()
    return ensemble.source.path.parent / "results" / f"{ensemble.ensemble_id}.manifest.yaml"


def validate_file_destination(path: Path, *, overwrite: bool, description: str) -> None:
    """Check a filesystem output destination without creating it or its parents."""
    if path.exists() or path.is_symlink():
        if not path.is_file():
            raise EnsembleError(f"{description} <{path}> is not a regular file")
        if not overwrite:
            raise EnsembleError(f"{description} <{path}> already exists")
        if not os.access(path, os.W_OK):
            raise EnsembleError(f"{description} <{path}> is not writable")

    parent = path.parent
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise EnsembleError(f"parent directory <{parent}> for {description} is not writable")


def _validate_manifest_destination(
    manifest_path: Path,
    ensemble: ExpandedEnsemble,
    simulations: tuple[ResolvedSimulation, ...],
    *,
    overwrite: bool,
) -> None:
    """Keep the parent-owned manifest away from files required by the run."""
    protected = {ensemble.source.path.resolve()}
    for simulation in simulations:
        if simulation.simulation_config.swmm_inp is not None:
            protected.add(simulation.simulation_config.swmm_inp.resolve())
        if simulation.artifacts.statistics_file is not None:
            protected.add(simulation.artifacts.statistics_file.resolve())
    if manifest_path in protected:
        raise EnsembleError(
            f"manifest <{manifest_path}> aliases a protected input or member artifact"
        )
    for path in protected:
        if manifest_path.exists() and path.exists() and manifest_path.samefile(path):
            raise EnsembleError(
                f"manifest <{manifest_path}> aliases a protected input or member artifact"
            )
    validate_file_destination(manifest_path, overwrite=overwrite, description="manifest")


def _initial_member_states(
    simulations: tuple[ResolvedSimulation, ...],
    failures: tuple[ValidationFailure, ...],
    selected_ids: set[str],
    has_selectors: bool,
) -> dict[str, dict]:
    states: dict[str, dict] = {}
    for simulation in simulations:
        selected = simulation.simulation_id in selected_ids
        states[simulation.simulation_id] = {
            "simulation_id": simulation.simulation_id,
            "selected": selected,
            "status": "planned" if selected else "not_selected",
            "coordinates": _manifest_coordinates(simulation.coordinates),
            "artifacts": {
                "rasters": dict(simulation.artifacts.output_map_names),
                "drainage": simulation.artifacts.drainage_output,
                "statistics": str(simulation.artifacts.statistics_file)
                if simulation.artifacts.statistics_file is not None
                else None,
            },
        }
    for index, failure in enumerate(failures):
        key = f"validation-{index}"
        states[key] = {
            "simulation_id": None,
            "selected": not has_selectors,
            "status": "not_selected" if has_selectors else "validation_failed",
            "coordinates": _manifest_coordinates(failure.coordinates),
            "failure": {"phase": failure.phase, "detail": failure.detail},
        }
    return states


def _manifest_coordinates(
    coordinates: tuple[tuple[str, JsonValue], ...],
) -> dict[str, JsonValue | dict[str, JsonValue]]:
    return {
        path: dict(cast(tuple[tuple[str, JsonValue], ...], value))
        if isinstance(value, tuple)
        else value
        for path, value in coordinates
    }


def _manifest_document(ensemble: ExpandedEnsemble, states: dict[str, dict]) -> dict:
    return {
        "manifest_version": 1,
        "last_updated_at": datetime.now().astimezone().isoformat(),
        "ensemble": {"id": ensemble.ensemble_id, "name": ensemble.ensemble_name},
        "source": {
            "path": str(ensemble.source.path),
            "document_index": ensemble.source.document_index,
            "file_digest": ensemble.source.file_digest,
            "document_digest": ensemble.source.document_digest,
        },
        "members": list(states.values()),
    }


def _create_manifest(
    path: Path, ensemble: ExpandedEnsemble, states: dict[str, dict], *, overwrite: bool
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as file_obj:
        yaml.safe_dump(_manifest_document(ensemble, states), file_obj, sort_keys=False)


def _update_manifest(path: Path, ensemble: ExpandedEnsemble, states: dict[str, dict]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as file_obj:
        temporary_path = Path(file_obj.name)
        yaml.safe_dump(_manifest_document(ensemble, states), file_obj, sort_keys=False)
        file_obj.flush()
        os.fsync(file_obj.fileno())
    try:
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
