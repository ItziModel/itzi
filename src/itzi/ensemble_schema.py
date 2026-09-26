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

import math
from datetime import timedelta
from typing import Annotated, Any, Literal

from pydantic import (
    Field,
    StrictFloat,
    StrictStr,
    field_validator,
    model_validator,
)

from itzi.ensemble_models import (
    ENSEMBLE_ID_CHARS,
    ENSEMBLE_ID_START_CHARS,
    OUTPUT_KEYS,
    DomainConfig,
    StrictModel,
    _canonical_json,
    _semantic_value,
    parse_iso_duration,
)

type SweepString = StrictStr | list[StrictStr]
type SweepFloat = StrictFloat | list[StrictFloat]


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
