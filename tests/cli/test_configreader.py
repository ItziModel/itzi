"""Test the reading and parsing of the config file."""

from configparser import ConfigParser
from datetime import datetime, timedelta
import logging
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
        "input": input_maps or {"dem": "z", "friction": "n"},
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
    assert sim_config.stats_file is None
    assert sim_config.dtinf == DefaultValues.DTINF
    assert sim_config.swmm_inp is None
    assert sim_config.drainage_output is None
    assert sim_config.orifice_coeff == DefaultValues.ORIFICE_COEFF
    assert sim_config.free_weir_coeff == DefaultValues.FREE_WEIR_COEFF
    assert sim_config.submerged_weir_coeff == DefaultValues.SUBMERGED_WEIR_COEFF
    assert grass_params.model_dump() == {
        "grassdata": None,
        "location": None,
        "mapset": None,
        "region": None,
        "mask": None,
        "grass_bin": None,
    }


def test_reader_normalizes_deprecated_aliases(tmp_path, caplog):
    config_file = write_config_file(
        tmp_path,
        make_config_dict(
            input_maps={
                "dem": "z",
                "friction": "n",
                "start_h": "legacy_depth",
                "drainage_capacity": "legacy_losses",
            },
            output={"prefix": "legacy", "values": "h, drainage_cap"},
        ),
    )

    itzi_logger = logging.getLogger("itzi")
    with caplog.at_level(logging.WARNING, logger="itzi"):
        itzi_logger.addHandler(caplog.handler)
        try:
            sim_config = ConfigReader(config_file).get_sim_params()
        finally:
            itzi_logger.removeHandler(caplog.handler)

    assert sim_config.input_map_names["water_depth"] == "legacy_depth"
    assert sim_config.input_map_names["losses"] == "legacy_losses"
    assert sim_config.output_map_names["water_depth"] == "legacy_water_depth"
    assert sim_config.output_map_names["mean_losses"] == "legacy_mean_losses"

    warning_messages = [record.message for record in caplog.records]
    assert any("Input 'start_h' is deprecated" in message for message in warning_messages)
    assert any(
        "Input 'drainage_capacity' is deprecated" in message for message in warning_messages
    )
    assert any("Output 'h' is deprecated" in message for message in warning_messages)
    assert any("Output 'drainage_cap' is deprecated" in message for message in warning_messages)


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
                "dem": "z",
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
                "dem": "z",
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
                "dem": "z",
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
