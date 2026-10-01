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

import itertools
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import cast

import yaml
from itzi_core import InfiltrationModelType, SurfaceFlowParameters, TemporalType
from itzi_core.const import DefaultValues
from pydantic import JsonValue as PydanticJsonValue
from pydantic import TypeAdapter, ValidationError
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver

import itzi.messenger as msgr
from itzi.ensemble.models import (
    DIRECT_INPUT_KEYS,
    MAX_ENSEMBLE_MEMBERS,
    DocumentFailure,
    EnsembleError,
    ExpandedEnsemble,
    ExpandedSimulation,
    JsonValue,
    LoadedYamlStream,
    NormalizedInfiltration,
    NormalizedTime,
    OutputTemplates,
    SourceDocument,
    parse_duration,
    render_template,
)
from itzi.ensemble.schema import (
    ConstantInfiltration,
    DrainageSweepConfig,
    GreenAmptInfiltration,
    InfiltrationAlternative,
    InputSweepConfig,
    NoInfiltration,
    ParameterSweepConfig,
    SweepFloat,
    SweepString,
    TimeConfig,
    YamlEnsembleDocumentV1,
)

type RawYamlScalar = None | bool | int | float | str | bytes | date | datetime
type RawYamlKey = RawYamlScalar | tuple[RawYamlKey, ...]
type RawYamlValue = (
    RawYamlScalar
    | list[RawYamlValue]
    | tuple[RawYamlValue, ...]
    | dict[RawYamlKey, RawYamlValue]
    | set[RawYamlKey]
)
type RawYamlMapping = dict[RawYamlKey, RawYamlValue]
type YamlDocument = dict[str, PydanticJsonValue]
type SweepScalar = str | float | InfiltrationAlternative

_YAML_DOCUMENT_ADAPTER = TypeAdapter(YamlDocument)


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects duplicate and merge mapping keys."""


def _construct_mapping(
    loader: _StrictSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> RawYamlMapping:
    mapping: RawYamlMapping = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "YAML merge keys are not supported",
                key_node.start_mark,
            )
        key = cast(RawYamlKey, loader.construct_object(key_node, deep=deep))
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "mapping keys must be hashable",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate mapping key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = cast(RawYamlValue, loader.construct_object(value_node, deep=deep))
    return mapping


_StrictSafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def normalize_time(value: TimeConfig) -> NormalizedTime:
    """Validate a public time form without leaking core's relative sentinels."""
    duration = parse_duration(value.duration) if value.duration is not None else None
    record_step = parse_duration(value.record_step)
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


def _parse_segment(segment: str) -> YamlDocument:
    if any(line.startswith("%") for line in segment.split("\n")):
        raise EnsembleError("YAML directives are not supported")
    value = cast(RawYamlValue, yaml.load(segment, Loader=_StrictSafeLoader))
    if value is None:
        raise EnsembleError("empty YAML documents are not supported")
    if not isinstance(value, dict):
        raise EnsembleError("document root must be a mapping")
    try:
        return _YAML_DOCUMENT_ADAPTER.validate_python(value, strict=True)
    except ValidationError as error:
        raise EnsembleError("document must contain JSON-compatible values") from error


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
    normalized_time = normalize_time(document.time)
    dimensions = _collect_dimensions(document)
    count = math.prod(len(values) for _, values in dimensions)
    if count > MAX_ENSEMBLE_MEMBERS:
        sizes = ", ".join(f"{path}={len(values)}" for path, values in dimensions)
        raise EnsembleError(
            f"ensemble expands to {count} simulations, exceeding the {MAX_ENSEMBLE_MEMBERS} limit ({sizes})"
        )

    simulations = []
    for selected_values in itertools.product(*(values for _, values in dimensions)):
        selected: dict[str, SweepScalar] = {
            path: value for (path, _), value in zip(dimensions, selected_values, strict=True)
        }
        coordinates = tuple(
            (path, _freeze_json(value)) for path, value in sorted(selected.items())
        )
        input_maps = _selected_input_maps(document.input, selected)
        infiltration_value = selected.get("input.infiltration", document.input.infiltration)
        assert isinstance(
            infiltration_value,
            (NoInfiltration, ConstantInfiltration, GreenAmptInfiltration),
        )
        infiltration = _normalize_infiltration(infiltration_value)
        parameters = _selected_parameters(document.parameters, selected)
        drainage = _selected_drainage(document.drainage, selected)
        outputs = _output_templates(document)
        simulations.append(
            ExpandedSimulation(
                source=source,
                ensemble_id=document.ensemble.id,
                coordinates=coordinates,
                grass=document.grass,
                time=normalized_time,
                input_maps=tuple(sorted(input_maps.items())),
                infiltration=infiltration,
                parameters=tuple(sorted(parameters.items())),
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
        ensemble_description=document.ensemble.description,
        manifest_template=manifest_template,
        simulations=tuple(simulations),
    )


def _collect_dimensions(
    document: YamlEnsembleDocumentV1,
) -> tuple[tuple[str, tuple[SweepScalar, ...]], ...]:
    dimensions: list[tuple[str, tuple[SweepScalar, ...]]] = []
    for key in DIRECT_INPUT_KEYS:
        value = cast(SweepString | None, getattr(document.input, key))
        if isinstance(value, list):
            dimensions.append((f"input.{key}", cast(tuple[SweepScalar, ...], tuple(value))))
    if isinstance(document.input.infiltration, list):
        dimensions.append(
            (
                "input.infiltration",
                cast(tuple[SweepScalar, ...], tuple(document.input.infiltration)),
            )
        )
    for key in type(document.parameters).model_fields:
        value = cast(SweepFloat | None, getattr(document.parameters, key))
        if isinstance(value, list):
            dimensions.append((f"parameters.{key}", cast(tuple[SweepScalar, ...], tuple(value))))
    if document.drainage is not None:
        for key in type(document.drainage).model_fields:
            value = cast(SweepString | SweepFloat | None, getattr(document.drainage, key))
            if isinstance(value, list):
                dimensions.append((f"drainage.{key}", cast(tuple[SweepScalar, ...], tuple(value))))
    return tuple(sorted(dimensions, key=lambda item: item[0]))


def _selected_input_maps(
    input_config: InputSweepConfig, selected: dict[str, SweepScalar]
) -> dict[str, str]:
    values: dict[str, str] = {}
    for key in DIRECT_INPUT_KEYS:
        value = selected.get(f"input.{key}", cast(SweepString | None, getattr(input_config, key)))
        if value is not None:
            assert isinstance(value, str)
            values[key] = value
    return values


def _normalize_infiltration(value: InfiltrationAlternative) -> NormalizedInfiltration:
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


def _selected_parameters(
    parameters: ParameterSweepConfig, selected: dict[str, SweepScalar]
) -> dict[str, float]:
    values: dict[str, float] = SurfaceFlowParameters().model_dump()
    values["dtinf"] = DefaultValues.DTINF
    for key in type(parameters).model_fields:
        value = selected.get(
            f"parameters.{key}", cast(SweepFloat | None, getattr(parameters, key))
        )
        if value is not None:
            assert isinstance(value, float)
            values[key] = value
    return values


def _selected_drainage(
    drainage: DrainageSweepConfig | None, selected: dict[str, SweepScalar]
) -> dict[str, str | float] | None:
    if drainage is None:
        return None
    values: dict[str, str | float] = {}
    for key in type(drainage).model_fields:
        value = selected.get(
            f"drainage.{key}", cast(SweepString | SweepFloat | None, getattr(drainage, key))
        )
        if value is None:
            continue
        assert isinstance(value, (str, float)), "drainage must be scalar after expansion"
        values[key] = value
    return values


def _output_templates(document: YamlEnsembleDocumentV1) -> OutputTemplates:
    raster_prefix = document.outputs.rasters.prefix if document.outputs.rasters else None
    variables = tuple(document.outputs.rasters.variables) if document.outputs.rasters else ()
    statistics_file = document.outputs.statistics.file if document.outputs.statistics else None
    drainage_dataset = (
        document.outputs.drainage.vector_dataset if document.outputs.drainage else None
    )
    for template in (raster_prefix, statistics_file, drainage_dataset):
        if template is not None:
            render_template(template, ensemble=document.ensemble.id, simulation="simulation")
    return OutputTemplates(
        raster_prefix=raster_prefix,
        raster_variables=variables,
        statistics_file=statistics_file,
        drainage_dataset=drainage_dataset,
    )


def _freeze_json(value: SweepScalar) -> JsonValue:
    if isinstance(value, (str, int, float)):
        return value
    return tuple(sorted(value.model_dump(exclude_none=True).items()))
