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
import math
import re
import string
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from itzi_core import (
    ARRAY_DEFINITIONS,
    ArrayCategory,
    DomainData,
    InfiltrationModelType,
    SimulationConfig,
    TemporalType,
)
from itzi_core.const import DefaultValues
from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from itzi.grass_session import GrassParams

INPUT_KEYS = tuple(
    definition.key
    for definition in ARRAY_DEFINITIONS
    if ArrayCategory.INPUT in definition.category
)
OUTPUT_KEYS = tuple(
    definition.key
    for definition in ARRAY_DEFINITIONS
    if ArrayCategory.OUTPUT in definition.category
)
GREEN_AMPT_KEYS = (
    "effective_porosity",
    "capillary_pressure",
    "hydraulic_conductivity",
    "soil_water_content",
)
DIRECT_INPUT_KEYS = tuple(
    key for key in INPUT_KEYS if key not in {"infiltration", *GREEN_AMPT_KEYS}
)
SURFACE_FLOW_DEFAULTS = {
    "hmin": DefaultValues.HFMIN,
    "cfl": DefaultValues.CFL,
    "theta": DefaultValues.THETA,
    "g": DefaultValues.G,
    "dtmax": DefaultValues.DTMAX,
    "slope_threshold": DefaultValues.SLOPE_THRESHOLD,
    "max_slope": DefaultValues.MAX_SLOPE,
    "max_error": DefaultValues.MAX_ERROR,
}
MAX_ENSEMBLE_MEMBERS = 100
MAX_BATCH_MEMBERS = 200
DURATION_RE = re.compile(
    r"P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?\Z"
)
ENSEMBLE_ID_START_CHARS = string.ascii_letters + string.digits
ENSEMBLE_ID_CHARS = ENSEMBLE_ID_START_CHARS + "._-"

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
class LoadedYamlStream:
    """Valid ensembles and document-local failures from one YAML file."""

    ensembles: tuple[ExpandedEnsemble, ...]
    failures: tuple[DocumentFailure, ...]


def parse_iso_duration(value: str) -> timedelta:
    """Parse the supported, non-calendar ISO 8601 duration subset."""
    match = DURATION_RE.fullmatch(value)
    if match is None or not any(match.groupdict().values()):
        raise ValueError("must be an ISO 8601 duration without years or months")
    groups = match.groupdict(default="0")
    if "T" in value and not any(groups[key] != "0" for key in ("hours", "minutes", "seconds")):
        raise ValueError("must include a time component after T")
    duration = timedelta(
        days=int(groups["days"]),
        hours=int(groups["hours"]),
        minutes=int(groups["minutes"]),
        seconds=float(groups["seconds"]),
    )
    if duration <= timedelta():
        raise ValueError("must be strictly positive")
    return duration


def format_iso_duration(value: timedelta) -> str:
    """Render a positive duration in a stable ISO 8601 representation."""
    total_seconds = value.total_seconds()
    if total_seconds <= 0:
        raise ValueError("duration must be strictly positive")
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = ["P"]
    if days:
        parts.append(f"{int(days)}D")
    if hours or minutes or seconds or not days:
        parts.append("T")
        if hours:
            parts.append(f"{int(hours)}H")
        if minutes:
            parts.append(f"{int(minutes)}M")
        if seconds or (not hours and not minutes):
            rendered_seconds = format(seconds, ".6f").rstrip("0").rstrip(".")
            parts.append(f"{rendered_seconds}S")
    return "".join(parts)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical_value(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical_value(item) for item in value]
    if isinstance(value, datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, date):
        return {"$date": value.isoformat()}
    if isinstance(value, timedelta):
        return {"$duration": format_iso_duration(value)}
    if isinstance(value, float) and not math.isfinite(value):
        raise EnsembleError("NaN and infinity are not supported")
    return value


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
