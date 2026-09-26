"""YAML ensemble configuration loading and deterministic expansion.

This module intentionally has no GRASS imports.  A configuration stream is
validated and expanded in the parent process before a resolver opens a GRASS
session.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import string
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from itzi_core import (
    ARRAY_DEFINITIONS,
    ArrayCategory,
    DomainData,
    InfiltrationModelType,
    SimulationConfig,
    TemporalType,
)
from itzi_core.const import DefaultValues
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

import itzi.messenger as msgr
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
DRAINAGE_DEFAULTS = {
    "orifice_coeff": DefaultValues.ORIFICE_COEFF,
    "free_weir_coeff": DefaultValues.FREE_WEIR_COEFF,
    "submerged_weir_coeff": DefaultValues.SUBMERGED_WEIR_COEFF,
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
type SweepString = StrictStr | list[StrictStr]
type SweepFloat = StrictFloat | list[StrictFloat]


class EnsembleError(ValueError):
    """Configuration error associated with a single YAML document."""


@dataclass(frozen=True)
class SourceDocument:
    """Stable source metadata for one document in a YAML stream."""

    path: Path
    document_index: int
    start_line: int
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


class EnsembleMetadata(StrictModel):
    id: StrictStr
    name: StrictStr | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if (
            not value
            or value[0] not in ENSEMBLE_ID_START_CHARS
            or any(character not in ENSEMBLE_ID_CHARS for character in value[1:])
        ):
            raise ValueError("must match [A-Za-z0-9][A-Za-z0-9._-]*")
        return value


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


class RelativeTimeConfig(StrictModel):
    duration: StrictStr
    record_step: StrictStr
    start: None = None
    end: None = None


class AbsoluteDurationTimeConfig(StrictModel):
    start: StrictStr
    duration: StrictStr
    record_step: StrictStr
    end: None = None


class AbsoluteEndTimeConfig(StrictModel):
    start: StrictStr
    end: StrictStr
    record_step: StrictStr
    duration: None = None


type TimeConfig = RelativeTimeConfig | AbsoluteDurationTimeConfig | AbsoluteEndTimeConfig


class NoInfiltration(StrictModel):
    type: Literal["none"]
    label: StrictStr | None = None


class ConstantInfiltration(StrictModel):
    type: Literal["constant"]
    label: StrictStr | None = None
    rate: StrictStr


class GreenAmptInfiltration(StrictModel):
    type: Literal["green-ampt"]
    label: StrictStr | None = None
    effective_porosity: StrictStr
    capillary_pressure: StrictStr
    hydraulic_conductivity: StrictStr
    soil_water_content: StrictStr | None = None


type InfiltrationAlternative = Annotated[
    NoInfiltration | ConstantInfiltration | GreenAmptInfiltration,
    Field(discriminator="type"),
]


class InputSweepConfig(StrictModel):
    ground_elevation: SweepString
    friction: SweepString
    water_depth: SweepString | None = None
    water_surface_elevation: SweepString | None = None
    losses: SweepString | None = None
    rainfall_rate: SweepString | None = None
    inflow: SweepString | None = None
    boundary_value: SweepString | None = None
    boundary_type: SweepString | None = None
    infiltration: InfiltrationAlternative | list[InfiltrationAlternative] = NoInfiltration(
        type="none"
    )

    @field_validator(
        "ground_elevation",
        "friction",
        "water_depth",
        "water_surface_elevation",
        "losses",
        "rainfall_rate",
        "inflow",
        "boundary_value",
        "boundary_type",
    )
    @classmethod
    def validate_string_sweep(cls, value: SweepString | None) -> SweepString | None:
        return _validate_sweep(value)

    @field_validator("infiltration")
    @classmethod
    def validate_infiltration_sweep(
        cls, value: InfiltrationAlternative | list[InfiltrationAlternative]
    ) -> InfiltrationAlternative | list[InfiltrationAlternative]:
        return _validate_sweep(value)

    @model_validator(mode="after")
    def validate_initial_conditions(self) -> InputSweepConfig:
        if self.water_depth is not None and self.water_surface_elevation is not None:
            raise ValueError("water_depth and water_surface_elevation are mutually exclusive")
        return self


class OptionSweepConfig(StrictModel):
    hmin: SweepFloat | None = None
    cfl: SweepFloat | None = None
    theta: SweepFloat | None = None
    g: SweepFloat | None = None
    dtmax: SweepFloat | None = None
    slope_threshold: SweepFloat | None = None
    max_slope: SweepFloat | None = None
    max_error: SweepFloat | None = None
    dtinf: SweepFloat | None = None

    @field_validator("*")
    @classmethod
    def validate_numeric_sweep(cls, value: SweepFloat | None) -> SweepFloat | None:
        return _validate_sweep(value, numeric=True)


class DrainageSweepConfig(StrictModel):
    swmm_input: SweepString
    orifice_coeff: SweepFloat
    free_weir_coeff: SweepFloat
    submerged_weir_coeff: SweepFloat

    @field_validator("swmm_input")
    @classmethod
    def validate_input_sweep(cls, value: SweepString) -> SweepString:
        return _validate_sweep(value)

    @field_validator("orifice_coeff", "free_weir_coeff", "submerged_weir_coeff")
    @classmethod
    def validate_coeff_sweep(cls, value: SweepFloat) -> SweepFloat:
        return _validate_sweep(value, numeric=True)


class RasterOutputs(StrictModel):
    prefix: StrictStr
    variables: list[StrictStr]

    @field_validator("variables")
    @classmethod
    def validate_variables(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must not be empty")
        unknown = sorted(set(value) - set(OUTPUT_KEYS))
        if unknown:
            raise ValueError(f"contains unsupported output variables: {', '.join(unknown)}")
        if len(value) != len(set(value)):
            raise ValueError("contains duplicate output variables")
        return value


class StatisticsOutputs(StrictModel):
    file: StrictStr

    @field_validator("file")
    @classmethod
    def validate_file(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be empty")
        return value


class DrainageOutputs(StrictModel):
    vector_dataset: StrictStr


class ManifestOutputs(StrictModel):
    file: StrictStr

    @field_validator("file")
    @classmethod
    def validate_file(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be empty")
        return value


class OutputConfig(StrictModel):
    rasters: RasterOutputs | None = None
    statistics: StatisticsOutputs | None = None
    drainage: DrainageOutputs | None = None
    manifest: ManifestOutputs | None = None


class HotstartOutputConfig(StrictModel):
    wallclock_step: StrictStr
    file: StrictStr

    @field_validator("wallclock_step")
    @classmethod
    def validate_step(cls, value: str) -> str:
        if parse_iso_duration(value) <= timedelta():
            raise ValueError("must be strictly positive")
        return value

    @field_validator("file")
    @classmethod
    def validate_file(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be empty")
        return value


class YamlEnsembleDocumentV1(StrictModel):
    schema_version: Literal[1]
    ensemble: EnsembleMetadata
    domain: DomainConfig
    time: TimeConfig
    input: InputSweepConfig
    options: OptionSweepConfig
    drainage: DrainageSweepConfig | None = None
    outputs: OutputConfig
    hotstart: HotstartOutputConfig | None = None

    @model_validator(mode="after")
    def validate_drainage_output(self) -> YamlEnsembleDocumentV1:
        if self.outputs.drainage is not None and self.drainage is None:
            raise ValueError("outputs.drainage requires a complete drainage configuration")
        return self


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
    hotstart_file: str | None
    hotstart_interval: timedelta | None


@dataclass(frozen=True)
class ExpandedSimulation:
    """One fully scalar pre-GRASS application configuration."""

    source: SourceDocument
    ensemble_id: str
    ensemble_name: str | None
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

    source: SourceDocument
    ensemble_id: str
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

    source: SourceDocument
    ensemble_id: str
    coordinates: tuple[tuple[str, JsonValue], ...]
    phase: str
    detail: str


@dataclass(frozen=True)
class ResolvedEnsemble:
    """Resolved member payloads and local validation failures for one ensemble."""

    source: SourceDocument
    ensemble_id: str
    ensemble_name: str | None
    manifest_file: Path
    simulations: tuple[ResolvedSimulation, ...]
    failures: tuple[ValidationFailure, ...]


@dataclass(frozen=True)
class LoadedYamlStream:
    """Valid ensembles and document-local failures from one YAML file."""

    ensembles: tuple[ExpandedEnsemble, ...]
    failures: tuple[DocumentFailure, ...]


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects duplicate and merge mapping keys."""


def _construct_mapping(
    loader: _StrictSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "YAML merge keys are not supported",
                key_node.start_mark,
            )
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "mapping keys must be hashable",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate mapping key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _validate_sweep(
    value: Any,
    *,
    numeric: bool = False,
) -> Any:
    """Validate one scalar-or-list sweep field after Pydantic's type validation."""
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError("sweep lists must not be empty")
    for item in values:
        if isinstance(item, list):
            raise ValueError("nested sweep lists are not supported")  # noqa: TRY004
        if numeric and (not isinstance(item, float) or not math.isfinite(item)):
            raise ValueError("must contain finite numbers")

    semantic_values = [_canonical_json(_semantic_value(item)) for item in values]
    if len(semantic_values) != len(set(semantic_values)):
        raise ValueError("contains duplicate semantic values")
    return value


def _semantic_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        dumped = value.model_dump(mode="python")
        dumped.pop("label", None)
        return dumped
    return value


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


def normalize_time(value: TimeConfig) -> NormalizedTime:
    """Validate a public time form without leaking core's relative sentinels."""
    duration = parse_iso_duration(value.duration) if value.duration is not None else None
    record_step = parse_iso_duration(value.record_step)
    start = _parse_datetime(value.start) if value.start is not None else None
    end = _parse_datetime(value.end) if value.end is not None else None
    aware = [timestamp for timestamp in (start, end) if timestamp is not None and timestamp.tzinfo]
    naive = [
        timestamp for timestamp in (start, end) if timestamp is not None and not timestamp.tzinfo
    ]
    if aware and naive:
        raise EnsembleError(
            "absolute timestamps must either all have timezone offsets or all be naive"
        )
    had_timezone_offset = bool(aware)
    if aware:
        start = start.replace(tzinfo=None) if start is not None else None
        end = end.replace(tzinfo=None) if end is not None else None

    if start is None:
        assert duration is not None
        return NormalizedTime(
            temporal_type=TemporalType.RELATIVE,
            start=None,
            end=None,
            duration=duration,
            record_step=record_step,
            had_timezone_offset=False,
        )
    if end is None:
        assert duration is not None
        end = start + duration
    else:
        duration = end - start
    if duration <= timedelta():
        raise EnsembleError("simulation duration must be strictly positive")
    return NormalizedTime(
        temporal_type=TemporalType.ABSOLUTE,
        start=start,
        end=end,
        duration=duration,
        record_step=record_step,
        had_timezone_offset=had_timezone_offset,
    )


def _parse_datetime(value: str) -> datetime:
    if "T" not in value:
        raise EnsembleError("timestamps must use ISO 8601 date and time notation")
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise EnsembleError(f"invalid ISO 8601 timestamp {value!r}") from error


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


def _document_digest(document: Any) -> str:
    payload = {"schema": "itzi-yaml-document-v1", "document": _canonical_value(document)}
    return hashlib.blake2b(_canonical_json(payload).encode(), digest_size=32).hexdigest()


def _is_document_marker(line: str, *, terminated: bool) -> bool:
    """Return whether a physical line is a supported YAML document marker."""
    if terminated and line.endswith("\r"):
        line = line[:-1]
    if not line.startswith("---"):
        return False
    suffix = line[3:]
    if suffix and suffix[0] not in " \t":
        return False
    suffix = suffix.lstrip(" \t")
    return not suffix or suffix.startswith("#")


def _segments(text: str) -> tuple[tuple[int, int, str], ...]:
    """Split at supported explicit document markers without parsing adjacent docs."""
    markers: list[tuple[int, int]] = []
    offset = 0
    lines = text.split("\n")
    for index, line in enumerate(lines):
        terminated = index < len(lines) - 1
        if _is_document_marker(line, terminated=terminated):
            markers.append((offset, index + 1))
        offset += len(line) + int(terminated)
    if not markers:
        return ((1, 0, text),)
    segments: list[tuple[int, int, str]] = []
    first = text[: markers[0][0]]
    if first.strip():
        segments.append((1, 0, first))
    for index, (marker_offset, start_line) in enumerate(markers):
        end = markers[index + 1][0] if index + 1 < len(markers) else len(text)
        segments.append((start_line, marker_offset, text[marker_offset:end]))
    return tuple(segments)


def _parse_segment(segment: str) -> dict[str, Any]:
    if any(line.startswith("%") for line in segment.split("\n")):
        raise EnsembleError("YAML directives are not supported")
    value = yaml.load(segment, Loader=_StrictSafeLoader)
    if value is None:
        raise EnsembleError("empty YAML documents are not supported")
    if not isinstance(value, dict):
        raise EnsembleError("document root must be a mapping")
    return value


def load_yaml_stream(path: str | Path) -> LoadedYamlStream:
    """Load every explicitly delimited YAML document, retaining local failures."""
    source_path = Path(path).expanduser().resolve()
    try:
        content = source_path.read_bytes()
    except OSError as error:
        raise EnsembleError(f"cannot read {source_path}: {error}") from error
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EnsembleError(f"{source_path}: YAML must be UTF-8") from error

    file_digest = hashlib.blake2b(content, digest_size=32).hexdigest()
    ensembles: list[ExpandedEnsemble] = []
    failures: list[DocumentFailure] = []
    for document_index, (start_line, _offset, segment) in enumerate(_segments(text)):
        try:
            raw_document = _parse_segment(segment)
        except yaml.YAMLError as error:
            marker = getattr(error, "problem_mark", None)
            failures.append(
                DocumentFailure(
                    path=source_path,
                    document_index=document_index,
                    line=start_line + marker.line if marker is not None else None,
                    column=marker.column + 1 if marker is not None else None,
                    phase="parse",
                    detail=str(getattr(error, "problem", error)),
                )
            )
            continue
        except EnsembleError as error:
            failures.append(
                DocumentFailure(
                    path=source_path,
                    document_index=document_index,
                    line=start_line,
                    column=1,
                    phase="parse",
                    detail=str(error),
                )
            )
            continue

        source = SourceDocument(
            path=source_path,
            document_index=document_index,
            start_line=start_line,
            file_digest=file_digest,
            document_digest=_document_digest(raw_document),
        )
        try:
            document = YamlEnsembleDocumentV1.model_validate(raw_document)
            ensemble = expand_yaml_document(source, document)
            if ensemble.simulations[0].time.had_timezone_offset:
                msgr.warning(
                    f"{source.path} document {source.document_index}: timezone offsets are "
                    "informational and are ignored by GRASS"
                )
            ensembles.append(ensemble)
        except ValidationError as error:
            failures.append(
                DocumentFailure(
                    path=source_path,
                    document_index=document_index,
                    line=start_line,
                    column=1,
                    phase="schema",
                    detail=_format_validation_error(error),
                )
            )
        except EnsembleError as error:
            failures.append(
                DocumentFailure(
                    path=source_path,
                    document_index=document_index,
                    line=start_line,
                    column=1,
                    phase="expansion",
                    detail=str(error),
                )
            )
    return LoadedYamlStream(tuple(ensembles), tuple(failures))


def _format_validation_error(error: ValidationError) -> str:
    details = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"])
        details.append(f"{location}: {item['msg']}" if location else item["msg"])
    return "; ".join(details)


def expand_yaml_document(
    source: SourceDocument, document: YamlEnsembleDocumentV1
) -> ExpandedEnsemble:
    """Expand sweep dimensions in canonical path order into scalar simulations."""
    if document.hotstart is not None:
        raise EnsembleError("YAML hotstart output requires Stage 2 checkpoint support")
    normalized_time = normalize_time(document.time)
    dimensions = _collect_dimensions(document)
    count = _checked_product(tuple(len(values) for _, values in dimensions))
    if count > MAX_ENSEMBLE_MEMBERS:
        sizes = ", ".join(f"{path}={len(values)}" for path, values in dimensions)
        raise EnsembleError(
            f"ensemble expands to {count} simulations, exceeding the {MAX_ENSEMBLE_MEMBERS} limit ({sizes})"
        )

    simulations = []
    for selected_values in itertools.product(*(values for _, values in dimensions)):
        selected = {
            path: value for (path, _), value in zip(dimensions, selected_values, strict=True)
        }
        coordinates = tuple(
            (path, _freeze_json(_semantic_value(value)))
            for path, value in sorted(selected.items())
        )
        input_maps = _selected_input_maps(document.input, selected)
        infiltration = _normalize_infiltration(
            selected.get("input.infiltration", document.input.infiltration)
        )
        options = _selected_options(document.options, selected)
        drainage = _selected_drainage(document.drainage, selected)
        outputs = _output_templates(document)
        simulations.append(
            ExpandedSimulation(
                source=source,
                ensemble_id=document.ensemble.id,
                ensemble_name=document.ensemble.name,
                coordinates=coordinates,
                domain=document.domain,
                time=normalized_time,
                input_maps=tuple(sorted(input_maps.items())),
                infiltration=infiltration,
                options=tuple(sorted(options.items())),
                drainage=tuple(sorted(drainage.items())) if drainage is not None else None,
                outputs=outputs,
            )
        )
    manifest_template = document.outputs.manifest.file if document.outputs.manifest else None
    if manifest_template is not None:
        render_template(manifest_template, ensemble=document.ensemble.id, simulation=None)
    return ExpandedEnsemble(
        source=source,
        ensemble_id=document.ensemble.id,
        ensemble_name=document.ensemble.name,
        manifest_template=manifest_template,
        simulations=tuple(simulations),
    )


def _collect_dimensions(
    document: YamlEnsembleDocumentV1,
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    dimensions: list[tuple[str, tuple[Any, ...]]] = []
    for key in DIRECT_INPUT_KEYS:
        value = getattr(document.input, key)
        if isinstance(value, list):
            dimensions.append((f"input.{key}", tuple(value)))
    if isinstance(document.input.infiltration, list):
        dimensions.append(("input.infiltration", tuple(document.input.infiltration)))
    for key in type(document.options).model_fields:
        value = getattr(document.options, key)
        if isinstance(value, list):
            dimensions.append((f"options.{key}", tuple(value)))
    if document.drainage is not None:
        for key in type(document.drainage).model_fields:
            value = getattr(document.drainage, key)
            if isinstance(value, list):
                dimensions.append((f"drainage.{key}", tuple(value)))
    return tuple(sorted(dimensions, key=lambda item: item[0]))


def _checked_product(sizes: tuple[int, ...]) -> int:
    product = 1
    for size in sizes:
        if size <= 0:
            raise EnsembleError("sweep lists must not be empty")
        product *= size
        if product > MAX_ENSEMBLE_MEMBERS:
            return product
    return product


def _selected_input_maps(
    input_config: InputSweepConfig, selected: dict[str, Any]
) -> dict[str, str]:
    values: dict[str, str] = {}
    for key in DIRECT_INPUT_KEYS:
        value = selected.get(f"input.{key}", getattr(input_config, key))
        if value is not None:
            assert isinstance(value, str)
            values[key] = value
    return values


def _normalize_infiltration(
    value: InfiltrationAlternative | list[InfiltrationAlternative],
) -> NormalizedInfiltration:
    assert not isinstance(value, list), "infiltration must be scalar after expansion"
    if isinstance(value, NoInfiltration):
        return NormalizedInfiltration(InfiltrationModelType.NULL, ())
    if isinstance(value, ConstantInfiltration):
        return NormalizedInfiltration(
            InfiltrationModelType.CONSTANT, (("infiltration", value.rate),)
        )
    assert isinstance(value, GreenAmptInfiltration)
    maps = {
        "effective_porosity": value.effective_porosity,
        "capillary_pressure": value.capillary_pressure,
        "hydraulic_conductivity": value.hydraulic_conductivity,
    }
    if value.soil_water_content is not None:
        maps["soil_water_content"] = value.soil_water_content
    return NormalizedInfiltration(InfiltrationModelType.GREEN_AMPT, tuple(sorted(maps.items())))


def _selected_options(options: OptionSweepConfig, selected: dict[str, Any]) -> dict[str, float]:
    values = dict(SURFACE_FLOW_DEFAULTS)
    values["dtinf"] = DefaultValues.DTINF
    for key in type(options).model_fields:
        value = selected.get(f"options.{key}", getattr(options, key))
        if value is not None:
            assert isinstance(value, float)
            values[key] = value
    return values


def _selected_drainage(
    drainage: DrainageSweepConfig | None, selected: dict[str, Any]
) -> dict[str, str | float] | None:
    if drainage is None:
        return None
    values: dict[str, str | float] = {}
    for key in type(drainage).model_fields:
        value = selected.get(f"drainage.{key}", getattr(drainage, key))
        assert not isinstance(value, list), "drainage must be scalar after expansion"
        values[key] = value
    return values


def _output_templates(document: YamlEnsembleDocumentV1) -> OutputTemplates:
    raster_prefix = document.outputs.rasters.prefix if document.outputs.rasters else None
    variables = tuple(document.outputs.rasters.variables) if document.outputs.rasters else ()
    statistics_file = document.outputs.statistics.file if document.outputs.statistics else None
    drainage_dataset = (
        document.outputs.drainage.vector_dataset if document.outputs.drainage else None
    )
    hotstart_file = document.hotstart.file if document.hotstart else None
    interval = parse_iso_duration(document.hotstart.wallclock_step) if document.hotstart else None
    for template in (raster_prefix, statistics_file, drainage_dataset, hotstart_file):
        if template is not None:
            render_template(template, ensemble=document.ensemble.id, simulation="simulation")
    return OutputTemplates(
        raster_prefix=raster_prefix,
        raster_variables=variables,
        statistics_file=statistics_file,
        drainage_dataset=drainage_dataset,
        hotstart_file=hotstart_file,
        hotstart_interval=interval,
    )


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


def _freeze_json(value: Any) -> JsonValue:
    if isinstance(value, BaseModel):
        return _freeze_json(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return tuple((str(key), _freeze_json(item)) for key, item in sorted(value.items()))
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported coordinate value {value!r}")


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
