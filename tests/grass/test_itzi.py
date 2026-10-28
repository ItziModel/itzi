import os
from collections.abc import Iterator
from configparser import ConfigParser
from contextlib import contextmanager
from datetime import timedelta
from uuid import uuid4

import grass.script as gscript
import numpy as np
import pytest

from itzi import SimulationRunner
from itzi.configreader import ConfigReader
from itzi.grass.session import GrassSessionManager


def _create_timed_rain_inputs(name_prefix: str) -> dict[str, str]:
    import grass.temporal as tgis

    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    suffix = uuid4().hex[:8]
    zero_depth_map = f"{name_prefix}_start_h_zero_{suffix}"
    rain_map_names = [
        f"{name_prefix}_rain_0_{suffix}",
        f"{name_prefix}_rain_10_{suffix}",
        f"{name_prefix}_rain_20_{suffix}",
    ]
    strds_name = f"{name_prefix}_rain_{suffix}"

    gscript.mapcalc(f"{zero_depth_map}=0")
    gscript.mapcalc(f"{rain_map_names[0]}=0")
    gscript.mapcalc(f"{rain_map_names[1]}=360")
    gscript.mapcalc(f"{rain_map_names[2]}=360")
    tgis.open_new_stds(
        name=strds_name,
        type="strds",
        temporaltype="relative",
        semantic="mean",
        title=strds_name,
        descr=strds_name,
        overwrite=gscript.overwrite(),
    )
    tgis.register_maps_in_space_time_dataset(
        name=strds_name,
        type="raster",
        maps=",".join(rain_map_names),
        start="0",
        increment="10",
        unit="seconds",
        interval=True,
    )

    return {
        "rainfall_rate": f"{strds_name}@{current_mapset}",
        "water_depth": f"{zero_depth_map}@{current_mapset}",
    }


@contextmanager
def _build_timed_rain_runner(
    test_data_temp_path: str,
    *,
    input_names: dict[str, str],
    duration: str,
    record_step: str,
    prefix: str,
) -> Iterator[SimulationRunner]:
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    config_dict = {
        "input": {
            "ground_elevation": f"z@{current_mapset}",
            "friction": f"n@{current_mapset}",
            "water_depth": input_names["water_depth"],
            "rainfall_rate": input_names["rainfall_rate"],
        },
        "time": {
            "duration": duration,
            "record_step": record_step,
        },
        "output": {
            "prefix": prefix,
            "values": "water_depth",
        },
        "options": {
            "hmin": "0.0001",
            "dtmax": "20",
            "cfl": "0.2",
        },
    }
    parser = ConfigParser()
    parser.read_dict(config_dict)
    config_file = os.path.join(test_data_temp_path, f"{prefix}.ini")
    with open(config_file, "w") as file_handle:
        parser.write(file_handle)

    conf_data = ConfigReader(config_file)
    with GrassSessionManager(conf_data.grass_params):
        runner = SimulationRunner(
            conf_data.sim_config, stats_file=conf_data.stats_file
        ).initialize()
        try:
            yield runner
        finally:
            runner.finalize()


@pytest.mark.usefixtures("grass_5by5_sim")
def test_number_of_output():
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    assert current_mapset == "5by5"

    h_map_list = gscript.list_grouped("raster", pattern="*out_5by5_water_depth_*")[current_mapset]
    assert len(h_map_list) == 3

    hmax_map_list = gscript.list_grouped("raster", pattern="*out_5by5_max_water_depth_*")[
        current_mapset
    ]
    assert len(hmax_map_list) == 3

    wse_map_list = gscript.list_grouped("raster", pattern="*out_5by5_water_surface_elevation_*")[
        current_mapset
    ]
    assert len(wse_map_list) == 3

    fr_map_list = gscript.list_grouped("raster", pattern="*out_5by5_froude_*")[current_mapset]
    assert len(fr_map_list) == 3

    v_map_list = gscript.list_grouped("raster", pattern="*out_5by5_flow_speed_*")[current_mapset]
    assert len(v_map_list) == 3

    vmax_map_list = gscript.list_grouped("raster", pattern="*out_5by5_max_flow_speed_*")[
        current_mapset
    ]
    assert len(vmax_map_list) == 3

    vdir_map_list = gscript.list_grouped("raster", pattern="*out_5by5_flow_velocity_direction_*")[
        current_mapset
    ]
    assert len(vdir_map_list) == 3

    qx_map_list = gscript.list_grouped("raster", pattern="*out_5by5_flow_rate_x_*")[current_mapset]
    assert len(qx_map_list) == 3

    qy_map_list = gscript.list_grouped("raster", pattern="*out_5by5_flow_rate_y_*")[current_mapset]
    assert len(qy_map_list) == 3

    created_volume_map_list = gscript.list_grouped("raster", pattern="*out_5by5_created_volume_*")[
        current_mapset
    ]
    assert len(created_volume_map_list) == 3


@pytest.mark.usefixtures("grass_5by5")
def test_region_mask(test_data_path):
    """Check if temporary mask and region are set and teared down."""
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    assert current_mapset == "5by5"
    # Get data from initial region and mask
    init_ncells = int(gscript.parse_command("g.region", flags="pg")["cells"])
    init_nulls = int(gscript.parse_command("r.univar", map="z", flags="g")["null_cells"])
    # Set simulation (should set region and mask)
    config_file = os.path.join(test_data_path, "5by5", "5by5_mask.ini")
    conf_data = ConfigReader(config_file)
    sim_params = conf_data.sim_config
    grass_params = conf_data.grass_params
    with GrassSessionManager(grass_params):
        sim_runner = SimulationRunner(sim_params).initialize()
        try:
            sim_runner.run()
        finally:
            sim_runner.finalize()
    # Check temporary mask and region
    assert (
        int(gscript.parse_command("r.univar", map="out_5by5_max_flow_speed_0002", flags="g")["n"])
        == 9
    )
    # Check tear down
    assert int(gscript.parse_command("g.region", flags="pg")["cells"]) == init_ncells
    assert int(gscript.parse_command("r.univar", map="z", flags="g")["null_cells"]) == init_nulls


@pytest.mark.usefixtures("grass_5by5")
def test_fails_when_region_has_no_dem_data(test_data_temp_path):
    gscript.run_command(
        "g.region", res=10, s=100, n=150, w=100, e=150, save="outside_dem", flags="o"
    )

    config_dict = {
        "input": {
            "ground_elevation": "z@5by5",
            "friction": "n@5by5",
            "water_depth": "start_h@5by5",
        },
        "time": {
            "duration": "00:01:00",
            "record_step": "00:00:30",
        },
        "output": {
            "prefix": "out_5by5_no_dem_overlap",
            "values": "water_depth",
        },
        "options": {
            "hmin": "0.0001",
            "dtmax": "0.3",
            "cfl": "0.2",
        },
        "grass": {
            "region": "outside_dem",
        },
    }
    parser = ConfigParser()
    parser.read_dict(config_dict)
    config_file = os.path.join(test_data_temp_path, "outside_dem.ini")
    with open(config_file, "w") as file_handle:
        parser.write(file_handle)

    conf_data = ConfigReader(config_file)

    with pytest.raises(
        RuntimeError, match=r"input map <ground_elevation> contains only NULL/NaN cells"
    ):
        with GrassSessionManager(conf_data.grass_params):
            SimulationRunner(conf_data.sim_config, stats_file=conf_data.stats_file).initialize()


@pytest.mark.usefixtures("grass_5by5")
@pytest.mark.parametrize("temporal_type", ["relative", "absolute"])
def test_empty_strds_fails_with_clear_error(test_data_temp_path: str, temporal_type: str) -> None:
    import grass.temporal as tgis

    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    strds_name = f"empty_{temporal_type}_rain_{uuid4().hex[:8]}"
    tgis.open_new_stds(
        name=strds_name,
        type="strds",
        temporaltype=temporal_type,
        semantic="mean",
        title=strds_name,
        descr=strds_name,
        overwrite=gscript.overwrite(),
    )

    input_names = {
        "rainfall_rate": f"{strds_name}@{current_mapset}",
        "water_depth": f"start_h@{current_mapset}",
    }
    with pytest.raises(
        RuntimeError,
        match=rf"STRDS <{strds_name}@{current_mapset}> has no temporal extent",
    ):
        with _build_timed_rain_runner(
            test_data_temp_path,
            input_names=input_names,
            duration="00:00:20",
            record_step="00:00:20",
            prefix=f"out_empty_{temporal_type}_{uuid4().hex[:8]}",
        ):
            pass


@pytest.mark.usefixtures("grass_5by5")
def test_relative_strds_with_unsupported_unit_fails_with_clear_error(
    test_data_temp_path: str,
) -> None:
    import grass.temporal as tgis

    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    suffix = uuid4().hex[:8]
    strds_name = f"monthly_rain_{suffix}"
    rain_map_name = f"monthly_rain_map_{suffix}"
    gscript.mapcalc(f"{rain_map_name}=10")
    tgis.open_new_stds(
        name=strds_name,
        type="strds",
        temporaltype="relative",
        semantic="mean",
        title=strds_name,
        descr=strds_name,
        overwrite=gscript.overwrite(),
    )
    tgis.register_maps_in_space_time_dataset(
        name=strds_name,
        type="raster",
        maps=rain_map_name,
        start="0",
        increment="1",
        unit="months",
        interval=True,
    )

    input_names = {
        "rainfall_rate": f"{strds_name}@{current_mapset}",
        "water_depth": f"start_h@{current_mapset}",
    }
    with pytest.raises(
        RuntimeError,
        match=(
            rf"STRDS <{strds_name}@{current_mapset}> uses unsupported relative time unit "
            r"<months>"
        ),
    ):
        with _build_timed_rain_runner(
            test_data_temp_path,
            input_names=input_names,
            duration="00:00:20",
            record_step="00:00:20",
            prefix=f"out_monthly_relative_{uuid4().hex[:8]}",
        ):
            pass


@pytest.mark.usefixtures("grass_5by5")
@pytest.mark.parametrize(
    ("target_seconds", "expected_rain_mm_per_hour", "expected_input_deadline_seconds"),
    [
        (9, 0.0, 10),
        (10, 360.0, 20),
        (12, 360.0, 20),
    ],
)
def test_timed_grass_rain_switches_cleanly_around_boundary(
    test_data_temp_path,
    target_seconds: int,
    expected_rain_mm_per_hour: float,
    expected_input_deadline_seconds: int,
):
    input_names = _create_timed_rain_inputs("timed_boundary")
    with _build_timed_rain_runner(
        test_data_temp_path,
        input_names=input_names,
        duration="00:00:20",
        record_step="00:00:20",
        prefix=f"out_timed_boundary_{target_seconds}_{uuid4().hex[:8]}",
    ) as sim_runner:
        simulation = sim_runner.sim
        simulation.update_until(timedelta(seconds=target_seconds))

        np.testing.assert_allclose(
            simulation.raster_domain.get_array("rainfall_rate"),
            expected_rain_mm_per_hour / (1000 * 3600),
        )
        assert simulation.next_ts["input"] == simulation.start_time + timedelta(
            seconds=expected_input_deadline_seconds
        )


@pytest.mark.usefixtures("grass_5by5")
def test_timed_grass_rain_is_applied_before_a_step_crosses_its_boundary(test_data_temp_path):
    input_names = _create_timed_rain_inputs("timed_crossing")
    with _build_timed_rain_runner(
        test_data_temp_path,
        input_names=input_names,
        duration="00:00:20",
        record_step="00:00:15",
        prefix=f"out_timed_crossing_{uuid4().hex[:8]}",
    ) as sim_runner:
        simulation = sim_runner.sim
        simulation.update()

        assert simulation.sim_time == simulation.start_time + timedelta(seconds=10)
        domain_volume = float(
            np.sum(simulation.raster_domain.get_array("water_depth"))
            * simulation.raster_domain.cell_area
        )
        assert domain_volume == pytest.approx(0.0, abs=1e-6)
        np.testing.assert_allclose(
            simulation.raster_domain.get_array("rainfall_rate"),
            360.0 / (1000 * 3600),
        )
