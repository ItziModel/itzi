from __future__ import annotations

from configparser import ConfigParser
import os
from uuid import uuid4

import grass.script as gscript
import pytest

from itzi import SimulationRunner
from itzi.configreader import ConfigReader
from itzi_core.const import TemporalType


def _build_runner(
    test_data_temp_path: str,
    prefix: str,
    temporal_type: TemporalType,
) -> SimulationRunner:
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    config_dict = {
        "input": {
            "dem": f"z@{current_mapset}",
            "friction": f"n@{current_mapset}",
            "water_depth": f"start_h@{current_mapset}",
        },
        "time": {
            "record_step": "00:00:10",
        },
        "output": {
            "prefix": prefix,
            "values": "water_depth",
        },
        "options": {
            "hmin": "0.0001",
            "dtmax": "0.3",
            "cfl": "0.2",
        },
    }
    if temporal_type == TemporalType.RELATIVE:
        config_dict["time"]["duration"] = "00:00:10"
    else:
        config_dict["time"].update(
            {
                "start_time": "2020-01-01 00:00",
                "duration": "00:00:10",
            }
        )

    parser = ConfigParser()
    parser.read_dict(config_dict)
    config_file = os.path.join(test_data_temp_path, f"{prefix}.ini")
    with open(config_file, "w") as file_handle:
        parser.write(file_handle)

    conf_data = ConfigReader(config_file)
    return SimulationRunner(conf_data.get_sim_params(), conf_data.get_grass_params())


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
@pytest.mark.parametrize(
    ("existing_temporal_type", "simulation_temporal_type"),
    [
        (TemporalType.RELATIVE, TemporalType.ABSOLUTE),
        (TemporalType.ABSOLUTE, TemporalType.RELATIVE),
    ],
)
def test_runner_creation_fails_when_overwriting_strds_with_different_temporal_type(
    monkeypatch: pytest.MonkeyPatch,
    test_data_temp_path: str,
    existing_temporal_type: TemporalType,
    simulation_temporal_type: TemporalType,
) -> None:
    monkeypatch.setenv("GRASS_OVERWRITE", "1")

    prefix = f"out_overwrite_{existing_temporal_type}_{simulation_temporal_type}_{uuid4().hex[:8]}"
    initial_runner = _build_runner(test_data_temp_path, prefix, existing_temporal_type)
    initial_runner.run().finalize()

    with pytest.raises(RuntimeError, match=r"temporal type"):
        _build_runner(test_data_temp_path, prefix, simulation_temporal_type)
