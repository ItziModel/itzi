"""Test the reading and parsing of the config file."""

import logging
from configparser import ConfigParser
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from itzi_core.const import DefaultValues, InfiltrationModelType, TemporalType
from itzi_core.data_containers import SurfaceFlowParameters

from itzi.configreader import ConfigReader


def write_config_file(tmp_path, config_dict: dict[str, dict[str, str]]) -> str:
    """Write a config dictionary to a temporary INI file."""
    parser = ConfigParser()
    parser.read_dict(config_dict)
    config_file = tmp_path / "config.ini"
    with config_file.open("w") as file_obj:
        parser.write(file_obj)
    return str(config_file)


def make_config_dict(
    *,
    time: dict[str, str] | None = None,
    input_maps: dict[str, str] | None = None,
    output: dict[str, str] | None = None,
    hotstart: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
    drainage: dict[str, str] | None = None,
    statistics: dict[str, str] | None = None,
    grass: dict[str, str] | None = None,
) -> dict[str, dict[str, str]]:
    """Create a minimal valid config dictionary for ConfigReader tests."""

    config_dict = {
        "time": time or {"duration": "00:01:00", "record_step": "00:00:30"},
        "input": input_maps or {"ground_elevation": "z", "friction": "n"},
        "output": output or {"prefix": "out", "values": "water_depth"},
    }

    if hotstart:
        config_dict["hotstart"] = hotstart
    if options:
        config_dict["options"] = options
    if drainage:
        config_dict["drainage"] = drainage
    if statistics:
        config_dict["statistics"] = statistics
    if grass:
        config_dict["grass"] = grass
    return config_dict


def test_reader_uses_defaults_when_optional_sections_are_missing(tmp_path):
    config_file = write_config_file(tmp_path, make_config_dict())

    reader = ConfigReader(config_file)
    sim_config = reader.get_sim_params()
    grass_params = reader.get_grass_params()

    assert sim_config.hotstart_config is None
    assert sim_config.surface_flow_parameters == SurfaceFlowParameters()
    assert reader.get_stats_file() is None
    assert sim_config.dtinf == DefaultValues.DTINF
    assert sim_config.swmm_inp is None
    assert sim_config.drainage_output is None
    assert sim_config.orifice_coeff == DefaultValues.ORIFICE_COEFF
    assert sim_config.free_weir_coeff == DefaultValues.FREE_WEIR_COEFF
    assert sim_config.submerged_weir_coeff == DefaultValues.SUBMERGED_WEIR_COEFF
    assert sim_config.input_map_names == {"ground_elevation": "z", "friction": "n"}
    assert sim_config.output_map_names == {"water_depth": "out_water_depth"}
    assert grass_params.model_dump() == {
        "grassdata": None,
        "location": None,
        "mapset": None,
        "region": None,
        "mask": None,
        "grass_bin": None,
    }


@pytest.mark.parametrize(
    ("statistics", "expected_stats_file"),
    [
        (None, None),
        ({"stats_file": ""}, None),
        ({"stats_file": "statistics.csv"}, "statistics.csv"),
    ],
)
def test_reader_exposes_stats_file_separately_from_simulation_config(
    tmp_path,
    statistics,
    expected_stats_file,
):
    config_file = write_config_file(
        tmp_path,
        make_config_dict(statistics=statistics),
    )

    reader = ConfigReader(config_file)

    assert reader.get_stats_file() == expected_stats_file
    assert "stats_file" not in type(reader.get_sim_params()).model_fields


def test_reader_normalizes_deprecated_aliases(tmp_path, caplog):
    config_file = write_config_file(
        tmp_path,
        make_config_dict(
            input_maps={
                "dem": "z",
                "friction": "n",
                "rain": "legacy_rain",
                "bctype": "legacy_boundary_type",
                "bcval": "legacy_boundary_value",
                "start_h": "legacy_depth",
                "drainage_capacity": "legacy_losses",
            },
            output={"prefix": "legacy", "values": "h, drainage_cap, hmax, v, vdir, vmax, qx, qy"},
        ),
    )

    itzi_logger = logging.getLogger("itzi")
    with caplog.at_level(logging.WARNING, logger="itzi"):
        itzi_logger.addHandler(caplog.handler)
        try:
            sim_config = ConfigReader(config_file).get_sim_params()
        finally:
            itzi_logger.removeHandler(caplog.handler)

    assert sim_config.input_map_names["ground_elevation"] == "z"
    assert sim_config.input_map_names["rainfall_rate"] == "legacy_rain"
    assert sim_config.input_map_names["boundary_type"] == "legacy_boundary_type"
    assert sim_config.input_map_names["boundary_value"] == "legacy_boundary_value"
    assert sim_config.input_map_names["water_depth"] == "legacy_depth"
    assert sim_config.input_map_names["losses"] == "legacy_losses"
    assert sim_config.output_map_names["water_depth"] == "legacy_water_depth"
    assert sim_config.output_map_names["mean_losses"] == "legacy_mean_losses"
    assert sim_config.output_map_names["max_water_depth"] == "legacy_max_water_depth"
    assert sim_config.output_map_names["flow_speed"] == "legacy_flow_speed"
    assert (
        sim_config.output_map_names["flow_velocity_direction"] == "legacy_flow_velocity_direction"
    )
    assert sim_config.output_map_names["max_flow_speed"] == "legacy_max_flow_speed"
    assert sim_config.output_map_names["flow_rate_x"] == "legacy_flow_rate_x"
    assert sim_config.output_map_names["flow_rate_y"] == "legacy_flow_rate_y"

    warning_messages = [record.message for record in caplog.records]
    for old_name, new_name in [
        ("dem", "ground_elevation"),
        ("rain", "rainfall_rate"),
        ("bctype", "boundary_type"),
        ("bcval", "boundary_value"),
        ("start_h", "water_depth"),
        ("drainage_capacity", "losses"),
    ]:
        assert any(
            f"Input '{old_name}' is deprecated. Use '{new_name}' instead." in message
            for message in warning_messages
        )
    for old_name, new_name in [
        ("h", "water_depth"),
        ("drainage_cap", "mean_losses"),
        ("hmax", "max_water_depth"),
        ("v", "flow_speed"),
        ("vdir", "flow_velocity_direction"),
        ("vmax", "max_flow_speed"),
        ("qx", "flow_rate_x"),
        ("qy", "flow_rate_y"),
    ]:
        assert any(
            f"Output '{old_name}' is deprecated. Use '{new_name}' instead." in message
            for message in warning_messages
        )


def test_reader_prefers_canonical_names_over_input_aliases(tmp_path):
    config_file = write_config_file(
        tmp_path,
        make_config_dict(
            input_maps={
                "dem": "legacy_elevation",
                "ground_elevation": "canonical_elevation",
                "friction": "n",
            }
        ),
    )

    sim_config = ConfigReader(config_file).get_sim_params()

    assert sim_config.input_map_names["ground_elevation"] == "canonical_elevation"


def test_reader_rejects_max_slope_below_slope_threshold(tmp_path):
    config_file = write_config_file(
        tmp_path,
        make_config_dict(options={"slope_threshold": "0.9", "max_slope": "0.8"}),
    )

    with pytest.raises(
        RuntimeError, match="max_slope must be greater than or equal to slope_threshold"
    ):
        ConfigReader(config_file)


@pytest.mark.parametrize("legacy_name", ["verror", "volume_error"])
def test_reader_normalizes_legacy_created_volume_aliases(tmp_path, caplog, legacy_name):
    config_file = write_config_file(
        tmp_path,
        make_config_dict(output={"prefix": "legacy", "values": legacy_name}),
    )

    itzi_logger = logging.getLogger("itzi")
    with caplog.at_level(logging.WARNING, logger="itzi"):
        itzi_logger.addHandler(caplog.handler)
        try:
            sim_config = ConfigReader(config_file).get_sim_params()
        finally:
            itzi_logger.removeHandler(caplog.handler)

    assert sim_config.output_map_names == {"created_volume": "legacy_created_volume"}
    assert any(
        f"Output '{legacy_name}' is deprecated. Use 'created_volume' instead." in record.message
        for record in caplog.records
    )


@pytest.mark.parametrize(
    ("time_section", "expected_temporal_type", "expected_start", "expected_end"),
    [
        (
            {"duration": "01:00:00", "record_step": "00:10:00"},
            TemporalType.RELATIVE,
            datetime.min,
            datetime.min + timedelta(hours=1),
        ),
        (
            {
                "start_time": "2025-01-02 03:04",
                "duration": "01:30:00",
                "record_step": "00:10:00",
            },
            TemporalType.ABSOLUTE,
            datetime(2025, 1, 2, 3, 4),
            datetime(2025, 1, 2, 4, 34),
        ),
        (
            {
                "start_time": "2025-01-02 03:04",
                "end_time": "2025-01-02 04:34",
                "record_step": "00:10:00",
            },
            TemporalType.ABSOLUTE,
            datetime(2025, 1, 2, 3, 4),
            datetime(2025, 1, 2, 4, 34),
        ),
    ],
)
def test_reader_accepts_supported_time_combinations(
    tmp_path,
    time_section,
    expected_temporal_type,
    expected_start,
    expected_end,
):
    config_file = write_config_file(tmp_path, make_config_dict(time=time_section))

    sim_config = ConfigReader(config_file).get_sim_params()

    assert sim_config.temporal_type == expected_temporal_type
    assert sim_config.start_time == expected_start
    assert sim_config.end_time == expected_end
    assert sim_config.record_step == timedelta(minutes=10)


@pytest.mark.parametrize(
    "time_section",
    [
        {
            "end_time": "2025-01-02 04:34",
            "duration": "01:30:00",
            "record_step": "00:10:00",
        },
        {"start_time": "2025-01-02 03:04", "record_step": "00:10:00"},
        {
            "start_time": "2025-01-02 03:04",
            "end_time": "2025-01-02 04:34",
            "duration": "01:30:00",
            "record_step": "00:10:00",
        },
    ],
)
def test_reader_rejects_invalid_time_combinations(tmp_path, time_section):
    config_file = write_config_file(tmp_path, make_config_dict(time=time_section))

    with pytest.raises(RuntimeError, match="accepted combinations"):
        ConfigReader(config_file)


def test_reader_rejects_mutually_exclusive_initial_conditions(tmp_path):
    config_file = write_config_file(
        tmp_path,
        make_config_dict(
            input_maps={
                "ground_elevation": "z",
                "friction": "n",
                "water_depth": "start_h",
                "water_surface_elevation": "start_wse",
            }
        ),
    )

    with pytest.raises(RuntimeError, match="mutually exclusive"):
        ConfigReader(config_file)


def test_reader_infers_green_ampt_model_from_complete_parameter_set(tmp_path):
    config_file = write_config_file(
        tmp_path,
        make_config_dict(
            input_maps={
                "ground_elevation": "z",
                "friction": "n",
                "effective_porosity": "porosity",
                "capillary_pressure": "pressure",
                "hydraulic_conductivity": "conductivity",
            }
        ),
    )

    sim_config = ConfigReader(config_file).get_sim_params()

    assert sim_config.infiltration_model == InfiltrationModelType.GREEN_AMPT


def test_reader_requires_all_green_ampt_maps(tmp_path):
    config_file = write_config_file(
        tmp_path,
        make_config_dict(
            input_maps={
                "ground_elevation": "z",
                "friction": "n",
                "effective_porosity": "porosity",
                "capillary_pressure": "pressure",
            }
        ),
    )

    with pytest.raises(RuntimeError, match="mutualy inclusive"):
        ConfigReader(config_file)


def test_reader_falls_back_to_config_dir_for_relative_swmm_inp(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    config_dir = tmp_path / "config_dir"
    cwd.mkdir()
    config_dir.mkdir()
    monkeypatch.chdir(cwd)

    config_swmm_inp = config_dir / "swmm_config.inp"
    config_swmm_inp.write_text("[TITLE]\n", encoding="utf-8")

    config_file = write_config_file(
        config_dir,
        make_config_dict(drainage={"swmm_inp": "swmm_config.inp"}),
    )

    sim_config = ConfigReader(config_file).get_sim_params()

    assert sim_config.swmm_inp == Path(config_swmm_inp)


def test_reader_fails_fast_when_swmm_inp_is_missing(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    config_dir = tmp_path / "config_dir"
    cwd.mkdir()
    config_dir.mkdir()
    monkeypatch.chdir(cwd)

    config_file = write_config_file(
        config_dir,
        make_config_dict(drainage={"swmm_inp": "missing_swmm.inp"}),
    )

    with pytest.raises(RuntimeError, match="SWMM input file <missing_swmm.inp> not found"):
        ConfigReader(config_file)
