"""
Copyright (C) 2015-2026 Laurent Courty

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

from configparser import ConfigParser
from datetime import datetime, timedelta
from pathlib import Path
from typing import NoReturn

from itzi_core import (
    INPUT_ARRAY_KEYS,
    OUTPUT_ARRAY_KEYS,
    HotstartRunConfig,
    InfiltrationModelType,
    SimulationConfig,
    SurfaceFlowParameters,
    TemporalType,
)
from itzi_core.const import DefaultValues
from pydantic import BaseModel, ConfigDict, ValidationError

import itzi.messenger as msgr
from itzi.grass_session import GrassParams

DEPRECATED_INPUT_ALIASES: list[tuple[str, str]] = [
    # (old, new)
    ("dem", "ground_elevation"),
    ("rain", "rainfall_rate"),
    ("bctype", "boundary_type"),
    ("bcval", "boundary_value"),
    ("drainage_capacity", "losses"),
    ("effective_pororosity", "effective_porosity"),
    ("start_h", "water_depth"),
]

DEPRECATED_OUTPUT_ALIASES: list[tuple[str, str]] = [
    # (old,new)
    ("drainage_cap", "mean_losses"),
    ("h", "water_depth"),
    ("wse", "water_surface_elevation"),
    ("hmax", "max_water_depth"),
    ("v", "flow_speed"),
    ("vdir", "flow_velocity_direction"),
    ("vmax", "max_flow_speed"),
    ("qx", "flow_rate_x"),
    ("qy", "flow_rate_y"),
    ("boundaries", "mean_boundary_flow"),
    ("verror", "created_volume"),
    ("volume_error", "created_volume"),
    ("inflow", "mean_inflow"),
    ("infiltration", "mean_infiltration"),
    ("rainfall", "mean_rainfall"),
    ("losses", "mean_losses"),
    ("drainage_stats", "mean_drainage_flow"),
]

TIME_DATE_FORMAT = "%Y-%m-%d %H:%M"
RELATIVE_TIME_ERROR = "{}: invalid format (should be HH:MM:SS)"
ABSOLUTE_TIME_ERROR = "{}: invalid format (should be yyyy-mm-dd HH:MM)"
TIME_COMBINATION_ERROR = (
    "accepted combinations: duration alone, start_time and duration, start_time and end_time"
)

TIME_OPTION_KEYS = ("start_time", "end_time", "duration", "record_step")
GREEN_AMPT_KEYS = (
    "effective_porosity",
    "capillary_pressure",
    "hydraulic_conductivity",
)
GRASS_MANDATORY_KEYS = ("grassdata", "location", "mapset")
GRASS_OPTION_KEYS = (*GRASS_MANDATORY_KEYS, "region", "mask", "grass_bin")
SURFACE_FLOW_OPTION_KEYS = tuple(SurfaceFlowParameters.model_fields)


def _read_parser(filename: str) -> ConfigParser:
    """Read the INI file or fail fast if it is missing."""
    params = ConfigParser(allow_no_value=True)
    if not params.read(filename):
        msgr.fatal(f"File <{filename}> not found")
    return params


def _read_string_options(
    params: ConfigParser, section: str, option_names: tuple[str, ...]
) -> dict[str, str]:
    """Collect the string-valued options present in a section."""
    values: dict[str, str] = {}
    for option_name in option_names:
        value = params.get(section, option_name, fallback=None)
        if value is not None:
            values[option_name] = value
    return values


def _read_float_options(
    params: ConfigParser, section: str, option_names: tuple[str, ...]
) -> dict[str, float]:
    """Collect the float-valued options present in a section."""
    values: dict[str, float] = {}
    for option_name in option_names:
        value = params.getfloat(section, option_name, fallback=None)
        if value is not None:
            values[option_name] = value
    return values


def _fatal_validation_error(error: ValidationError) -> NoReturn:
    """Convert a Pydantic validation error into an Itzi fatal error."""
    messages = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"])
        if location:
            messages.append(f"{location}: {item['msg']}")
        else:
            messages.append(str(item["msg"]))
    msgr.fatal("; ".join(messages))
    raise AssertionError("unreachable")


def _warn_about_deprecated_alias(alias_kind: str, old_name: str, new_name: str) -> None:
    """Emit a warning for a deprecated configuration alias."""
    msgr.warning(f"{alias_kind} '{old_name}' is deprecated. Use '{new_name}' instead.")


def _read_time_values(params: ConfigParser) -> dict[str, str | None]:
    """Read raw time values from the config file."""
    time_values: dict[str, str | None] = dict.fromkeys(TIME_OPTION_KEYS)
    time_values.update(_read_string_options(params, "time", TIME_OPTION_KEYS))
    return time_values


def _read_hotstart_values(params: ConfigParser) -> HotstartRunConfig | None:
    """Read optional hotstart settings from the config file."""
    if not params.has_section("hotstart"):
        return None

    hotstart_values = {
        "wallclock_step": params.get("hotstart", "wallclock_step", fallback=None),
        "save_file_name": params.get("hotstart", "save_file", fallback=None),
    }
    if all(value is None for value in hotstart_values.values()):
        return None

    try:
        return HotstartRunConfig.model_validate(hotstart_values)
    except ValidationError as error:
        _fatal_validation_error(error)


def _read_input_map_names(params: ConfigParser) -> dict[str, str | None]:
    """Read input map names and normalize deprecated aliases."""
    map_names: dict[str, str | None] = dict.fromkeys(INPUT_ARRAY_KEYS)

    for old_input_name, new_input_name in DEPRECATED_INPUT_ALIASES:
        if params.has_option("input", old_input_name):
            _warn_about_deprecated_alias("Input", old_input_name, new_input_name)
            map_names[new_input_name] = params.get("input", old_input_name)

    for input_name in INPUT_ARRAY_KEYS:
        if params.has_option("input", input_name):
            map_names[input_name] = params.get("input", input_name)

    return map_names


def _normalize_output_values(raw_values: str | None) -> list[str]:
    """Normalize requested output names and deprecated aliases."""
    if raw_values is None:
        return []

    aliases = dict(DEPRECATED_OUTPUT_ALIASES)
    output_values = []
    for output_name in raw_values.split(","):
        output_name = output_name.strip()
        if not output_name:
            continue
        canonical_name = aliases.get(output_name, output_name)
        if canonical_name != output_name:
            _warn_about_deprecated_alias("Output", output_name, canonical_name)
        if canonical_name not in output_values:
            output_values.append(canonical_name)
    return output_values


def _generate_output_map_names(prefix: str, output_values: list[str]) -> dict[str, str | None]:
    """Build the output map dictionary from the selected outputs."""
    output_map_names: dict[str, str | None] = dict.fromkeys(OUTPUT_ARRAY_KEYS)
    for value in output_values:
        if value in output_map_names:
            output_map_names[value] = f"{prefix}_{value}"
    return output_map_names


def _read_output_config(params: ConfigParser) -> tuple[str, list[str], dict[str, str | None]]:
    """Read output settings and derive output map names."""
    prefix = params.get("output", "prefix", fallback=None)
    if prefix is None:
        prefix = f"itzi_results_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    output_values = _normalize_output_values(params.get("output", "values", fallback=None))
    output_map_names = _generate_output_map_names(prefix, output_values)
    return prefix, output_values, output_map_names


def _read_surface_flow_parameters(params: ConfigParser) -> SurfaceFlowParameters:
    """Build validated surface flow parameters from config options."""
    surface_flow_values = _read_float_options(params, "options", SURFACE_FLOW_OPTION_KEYS)
    try:
        return SurfaceFlowParameters(**surface_flow_values)
    except ValidationError as error:
        _fatal_validation_error(error)


def _resolve_swmm_input_path(swmm_inp: str, config_file: str) -> Path:
    """Resolve a SWMM input path, searching cwd first and then the config directory."""
    configured_path = Path(swmm_inp)
    if configured_path.is_absolute():
        if configured_path.is_file():
            return configured_path
        msgr.fatal(f"SWMM input file <{configured_path}> not found")

    cwd_candidate = Path.cwd() / configured_path
    if cwd_candidate.is_file():
        return cwd_candidate

    config_candidate = Path(config_file).resolve().parent / configured_path
    if config_candidate.is_file():
        return config_candidate

    msgr.fatal(
        "SWMM input file "
        f"<{swmm_inp}> not found in current directory <{Path.cwd()}> "
        f"or config directory <{config_candidate.parent}>"
    )
    raise AssertionError("unreachable")


def _read_grass_params(params: ConfigParser) -> GrassParams:
    """Build GRASS session parameters from the config file."""
    return GrassParams(**_read_string_options(params, "grass", GRASS_OPTION_KEYS))


class SimulationTimes(BaseModel):
    """Parsed and validated simulation time settings."""

    model_config = ConfigDict(frozen=True)

    start: datetime
    end: datetime
    record_step: timedelta | None
    temporal_type: TemporalType

    @classmethod
    def from_raw_values(cls, raw_values: dict[str, str | None]) -> SimulationTimes:
        """Parse and validate the configured simulation times."""
        temporal_type = cls._resolve_temporal_type(raw_values)
        duration = cls._parse_timedelta(raw_values["duration"])
        record_step = cls._parse_timedelta(raw_values["record_step"])
        start = cls._parse_datetime(raw_values["start_time"])
        end = cls._parse_datetime(raw_values["end_time"])

        if start is None:
            start = datetime.min
        if end is None:
            assert duration is not None
            end: datetime = start + duration
        if start >= end:
            msgr.fatal("Simulation duration must be positive")
        if duration is None:
            duration: timedelta = end - start

        return cls(
            start=start,
            end=end,
            record_step=record_step,
            temporal_type=temporal_type,
        )

    @staticmethod
    def _resolve_temporal_type(raw_values: dict[str, str | None]) -> TemporalType:
        """Infer whether the simulation uses relative or absolute time."""
        has_duration_only = (
            raw_values["duration"] is not None
            and raw_values["start_time"] is None
            and raw_values["end_time"] is None
        )
        has_start_and_duration = (
            raw_values["start_time"] is not None
            and raw_values["duration"] is not None
            and raw_values["end_time"] is None
        )
        has_start_and_end = (
            raw_values["start_time"] is not None
            and raw_values["end_time"] is not None
            and raw_values["duration"] is None
        )
        if not (has_duration_only or has_start_and_duration or has_start_and_end):
            msgr.fatal(TIME_COMBINATION_ERROR)
        if has_duration_only:
            return TemporalType.RELATIVE
        return TemporalType.ABSOLUTE

    @staticmethod
    def _parse_timedelta(value: str | None) -> timedelta | None:
        """Parse a `HH:MM:SS` string into a `timedelta`."""
        if value is None:
            return None
        try:
            hours_str, minutes_str, seconds_str = value.split(":")
            hours = int(hours_str)
            minutes = int(minutes_str)
            seconds = int(seconds_str)
        except (TypeError, ValueError):
            msgr.fatal(RELATIVE_TIME_ERROR.format(value))

        if hours < 0 or not 0 <= minutes <= 59 or not 0 <= seconds <= 59:
            msgr.fatal(RELATIVE_TIME_ERROR.format(value))
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        """Parse an absolute simulation timestamp."""
        if value is None:
            return None
        try:
            return datetime.strptime(value, TIME_DATE_FORMAT)
        except ValueError:
            msgr.fatal(ABSOLUTE_TIME_ERROR.format(value))


class ConfigReader:
    """Parse an INI file and expose validated simulation parameters."""

    def __init__(self, filename: str | None) -> None:
        """Read, normalize, and validate a simulation config file."""
        if filename is None:
            msgr.fatal("Not a valid configuration file")

        self.config_file = filename

        params = _read_parser(filename)
        self.hotstart_config = _read_hotstart_values(params)
        raw_time_values = _read_time_values(params)
        self.input_map_names = _read_input_map_names(params)
        self.out_prefix, self.out_values, self.output_map_names = _read_output_config(params)
        self.stats_file = params.get("statistics", "stats_file", fallback=None) or None
        self.sim_times = SimulationTimes.from_raw_values(raw_time_values)
        self._check_general_input(self.input_map_names)
        infiltration_model = self._resolve_infiltration_model(self.input_map_names)

        self.grass_params = _read_grass_params(params)
        self._check_grass_params(self.grass_params)

        surface_flow_parameters = _read_surface_flow_parameters(params)
        assert self.sim_times.record_step is not None

        swmm_inp = params.get("drainage", "swmm_inp", fallback=None)
        if swmm_inp is not None:
            resolved_swmm_inp = _resolve_swmm_input_path(swmm_inp, self.config_file)
        else:
            resolved_swmm_inp = None

        try:
            self.sim_config = SimulationConfig(
                start_time=self.sim_times.start,
                end_time=self.sim_times.end,
                record_step=self.sim_times.record_step,
                temporal_type=self.sim_times.temporal_type,
                hotstart_config=self.hotstart_config,
                input_map_names={
                    key: value for key, value in self.input_map_names.items() if value is not None
                },
                output_map_names={
                    key: value for key, value in self.output_map_names.items() if value is not None
                },
                surface_flow_parameters=surface_flow_parameters,
                dtinf=params.getfloat("options", "dtinf", fallback=DefaultValues.DTINF),
                infiltration_model=infiltration_model,
                swmm_inp=resolved_swmm_inp,
                drainage_output=params.get("drainage", "output", fallback=None),
                orifice_coeff=params.getfloat(
                    "drainage", "orifice_coeff", fallback=DefaultValues.ORIFICE_COEFF
                ),
                free_weir_coeff=params.getfloat(
                    "drainage", "free_weir_coeff", fallback=DefaultValues.FREE_WEIR_COEFF
                ),
                submerged_weir_coeff=params.getfloat(
                    "drainage",
                    "submerged_weir_coeff",
                    fallback=DefaultValues.SUBMERGED_WEIR_COEFF,
                ),
            )
        except ValidationError as error:
            _fatal_validation_error(error)

    def _check_grass_params(self, grass_params: GrassParams) -> None:
        """Ensure mandatory GRASS settings are provided together."""
        grass_values = grass_params.model_dump()
        grass_any = any(grass_values[key] for key in GRASS_MANDATORY_KEYS)
        grass_all = all(grass_values[key] for key in GRASS_MANDATORY_KEYS)
        if grass_any and not grass_all:
            msgr.fatal(f"{GRASS_MANDATORY_KEYS} are mutualy inclusive")

    def _resolve_infiltration_model(
        self, input_map_names: dict[str, str | None]
    ) -> InfiltrationModelType:
        """Infer the infiltration model from the configured input maps."""
        infiltration_input = input_map_names["infiltration"]
        ga_any = any(input_map_names[key] for key in GREEN_AMPT_KEYS)
        ga_all = all(input_map_names[key] for key in GREEN_AMPT_KEYS)

        if not infiltration_input and not ga_any:
            return InfiltrationModelType.NULL
        if infiltration_input and not ga_any:
            return InfiltrationModelType.CONSTANT
        if infiltration_input and ga_any:
            msgr.fatal("Infiltration model incompatible with user-defined rate")
        if ga_any and not ga_all:
            msgr.fatal(f"{GREEN_AMPT_KEYS} are mutualy inclusive")
        return InfiltrationModelType.GREEN_AMPT

    def _check_general_input(self, input_map_names: dict[str, str | None]) -> None:
        """Validate mandatory and mutually exclusive input maps."""
        if not all(
            [
                input_map_names["ground_elevation"],
                input_map_names["friction"],
                self.sim_times.record_step,
            ]
        ):
            msgr.fatal("inputs <ground_elevation>, <friction> and <record_step> are mandatory")
        if input_map_names["water_depth"] and input_map_names["water_surface_elevation"]:
            msgr.fatal(
                "inputs <water_depth> and <water_surface_elevation> are mutually exclusive."
            )
