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

import json
import string
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from itzi_core import (
    INPUT_ARRAY_KEYS,
    DomainData,
    InfiltrationModelType,
    SimulationConfig,
    TemporalType,
)
from pydantic import BaseModel, ConfigDict, StrictStr, model_validator
from pydantic import JsonValue as PydanticJsonValue

from itzi.grass.session import GrassParams

GREEN_AMPT_KEYS = (
    "effective_porosity",
    "capillary_pressure",
    "hydraulic_conductivity",
    "soil_water_content",
)
DIRECT_INPUT_KEYS = tuple(
    key for key in INPUT_ARRAY_KEYS if key not in {"infiltration", *GREEN_AMPT_KEYS}
)
MAX_ENSEMBLE_MEMBERS = 100
MAX_BATCH_MEMBERS = 200

type JsonValue = (
    None | bool | int | float | str | tuple[JsonValue, ...] | tuple[tuple[str, JsonValue], ...]
)


class EnsembleError(ValueError):
    """Configuration error associated with a single YAML document."""


@dataclass(frozen=True)
class SourceDocument:
    """Stable source metadata for one document in a YAML stream."""

    path: Path
    document_index: int
    file_digest: str
    document_digest: str


@dataclass(frozen=True)
class DocumentFailure:
    """A document-local error that does not prevent later documents loading."""

    path: Path
    document_index: int
    line: int | None
    column: int | None
    phase: Literal["parse", "schema", "expansion"]
    detail: str

    def format(self) -> str:
        location = f"{self.path} document {self.document_index}"
        if self.line is not None and self.column is not None:
            location += f" line {self.line}, column {self.column}"
        return f"{location}: {self.phase}: {self.detail}"


class StrictModel(BaseModel):
    """Base model for the public YAML schema."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class GrassContextConfig(StrictModel):
    database: StrictStr | None = None
    project: StrictStr | None = None
    mapset: StrictStr | None = None
    executable: StrictStr | None = None

    @model_validator(mode="after")
    def validate_context(self) -> GrassContextConfig:
        values = (self.database, self.project, self.mapset)
        if any(value is not None for value in values) and not all(
            value is not None for value in values
        ):
            raise ValueError("database, project, and mapset must be supplied together")
        if self.executable is not None and self.database is None:
            raise ValueError("executable requires database, project, and mapset")
        return self


class DomainConfig(StrictModel):
    grass: GrassContextConfig | None = None
    region: StrictStr | None = None
    mask: StrictStr | None = None


@dataclass(frozen=True)
class NormalizedTime:
    """Time semantics retained until the final core configuration boundary."""

    temporal_type: TemporalType
    start: datetime | None
    end: datetime | None
    duration: timedelta
    record_step: timedelta
    had_timezone_offset: bool


@dataclass(frozen=True)
class NormalizedInfiltration:
    model: InfiltrationModelType
    input_maps: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class OutputTemplates:
    raster_prefix: str | None
    raster_variables: tuple[str, ...]
    statistics_file: str | None
    drainage_dataset: str | None


@dataclass(frozen=True)
class ExpandedSimulation:
    """One fully scalar pre-GRASS application configuration."""

    source: SourceDocument
    ensemble_id: str
    coordinates: tuple[tuple[str, JsonValue], ...]
    domain: DomainConfig
    time: NormalizedTime
    input_maps: tuple[tuple[str, str], ...]
    infiltration: NormalizedInfiltration
    options: tuple[tuple[str, float], ...]
    drainage: tuple[tuple[str, str | float], ...] | None
    outputs: OutputTemplates


@dataclass(frozen=True)
class ExpandedEnsemble:
    """One YAML document after Cartesian expansion."""

    source: SourceDocument
    ensemble_id: str
    ensemble_name: str | None
    manifest_template: str | None
    simulations: tuple[ExpandedSimulation, ...]


@dataclass(frozen=True)
class EffectiveMask:
    """The immutable mask source selected during GRASS resolution."""

    mode: Literal["explicit", "active", "none"]
    source: str | None


@dataclass(frozen=True)
class ArtifactSummary:
    """Rendered, application-owned output destinations for one member."""

    output_map_names: tuple[tuple[str, str], ...]
    drainage_output: str | None
    statistics_file: Path | None


@dataclass(frozen=True)
class ResolvedSimulation:
    """Spawn-serializable execution payload with one final core configuration."""

    simulation_id: str
    coordinates: tuple[tuple[str, JsonValue], ...]
    grass_params: GrassParams
    domain_data: DomainData
    effective_mask: EffectiveMask
    input_kinds: tuple[tuple[str, Literal["raster", "strds"]], ...]
    simulation_config: SimulationConfig
    artifacts: ArtifactSummary
    normalized_payload: str


@dataclass(frozen=True)
class ValidationFailure:
    """A coordinate that could not become an executable resolved simulation."""

    coordinates: tuple[tuple[str, JsonValue], ...]
    phase: str
    detail: str


@dataclass(frozen=True)
class ResolvedEnsemble:
    """One expanded ensemble with its resolved members and validation results."""

    ensemble: ExpandedEnsemble
    simulations: tuple[ResolvedSimulation, ...]
    failures: tuple[ValidationFailure, ...]
    artifact_failure: str | None = None


@dataclass(frozen=True)
class LoadedYamlStream:
    """Valid ensembles and document-local failures from one YAML file."""

    ensembles: tuple[ExpandedEnsemble, ...]
    failures: tuple[DocumentFailure, ...]


def parse_duration(value: str) -> timedelta:
    """Parse a strictly positive ``HH:MM:SS`` duration."""
    try:
        hours_str, minutes_str, seconds_str = value.split(":")
        hours = int(hours_str)
        minutes = int(minutes_str)
        seconds = int(seconds_str)
    except ValueError as error:
        raise ValueError("must use HH:MM:SS") from error
    if hours < 0 or not 0 <= minutes <= 59 or not 0 <= seconds <= 59:
        raise ValueError("must use HH:MM:SS")
    duration = timedelta(hours=hours, minutes=minutes, seconds=seconds)
    if duration <= timedelta():
        raise ValueError("must be strictly positive")
    return duration


def format_duration(value: timedelta) -> str:
    """Render a positive duration for canonical internal data."""
    if value <= timedelta():
        raise ValueError("duration must be strictly positive")
    total_seconds = value.days * 86400 + value.seconds
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    rendered = f"{hours:02}:{minutes:02}:{seconds:02}"
    if value.microseconds:
        return f"{rendered}.{value.microseconds:06}".rstrip("0")
    return rendered


def _canonical_json(value: PydanticJsonValue) -> str:
    """Serialize a JSON-compatible value in a stable form."""
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except ValueError as error:
        raise EnsembleError("NaN and infinity are not supported") from error


def render_template(template: str, *, ensemble: str, simulation: str | None) -> str:
    """Render the intentionally small output-template language."""
    formatter = string.Formatter()
    values = {"ensemble": ensemble, "simulation": simulation}
    try:
        fragments = formatter.parse(template)
        rendered: list[str] = []
        for literal, field_name, format_spec, conversion in fragments:
            rendered.append(literal)
            if field_name is None:
                continue
            if field_name not in values:
                raise EnsembleError(f"unsupported template placeholder {{{field_name}}}")
            if values[field_name] is None:
                raise EnsembleError(f"{{{field_name}}} is not permitted in this template")
            if format_spec or conversion:
                raise EnsembleError(
                    "template format specifications and conversions are not supported"
                )
            rendered.append(str(values[field_name]))
    except ValueError as error:
        raise EnsembleError(f"invalid output template: {error}") from error
    return "".join(rendered)


def check_batch_limits(ensembles: tuple[ExpandedEnsemble, ...]) -> None:
    """Reject a valid batch that is too large before any resolver is started."""
    count = sum(len(ensemble.simulations) for ensemble in ensembles)
    if count > MAX_BATCH_MEMBERS:
        raise EnsembleError(
            f"batch expands to {count} simulations, exceeding the {MAX_BATCH_MEMBERS} limit"
        )


def check_unique_ensemble_ids(ensembles: tuple[ExpandedEnsemble, ...]) -> None:
    """Reject a batch-global duplicate ensemble identifier."""
    seen: set[str] = set()
    for ensemble in ensembles:
        if ensemble.ensemble_id in seen:
            raise EnsembleError(f"duplicate ensemble ID {ensemble.ensemble_id!r}")
        seen.add(ensemble.ensemble_id)
