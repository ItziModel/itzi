""" """

from configparser import ConfigParser
import os

import grass.script as gscript
import pytest

from itzi import SimulationRunner
from itzi.configreader import ConfigReader
from itzi.itzi_error import ItziFatal


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5_sim")
def test_number_of_output():
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    assert current_mapset == "5by5"

    h_map_list = gscript.list_grouped("raster", pattern="*out_5by5_water_depth_*")[current_mapset]
    assert len(h_map_list) == 4

    wse_map_list = gscript.list_grouped("raster", pattern="*out_5by5_water_surface_elevation_*")[
        current_mapset
    ]
    assert len(wse_map_list) == 3

    fr_map_list = gscript.list_grouped("raster", pattern="*out_5by5_froude_*")[current_mapset]
    assert len(fr_map_list) == 3

    v_map_list = gscript.list_grouped("raster", pattern="*out_5by5_v_*")[current_mapset]
    assert len(v_map_list) == 4

    vdir_map_list = gscript.list_grouped("raster", pattern="*out_5by5_vdir_*")[current_mapset]
    assert len(vdir_map_list) == 3

    qx_map_list = gscript.list_grouped("raster", pattern="*out_5by5_qx_*")[current_mapset]
    assert len(qx_map_list) == 3

    qy_map_list = gscript.list_grouped("raster", pattern="*out_5by5_qy_*")[current_mapset]
    assert len(qy_map_list) == 3

    verr_map_list = gscript.list_grouped("raster", pattern="*out_5by5_volume_error_*")[
        current_mapset
    ]
    assert len(verr_map_list) == 3


@pytest.mark.forked
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
    sim_params = conf_data.get_sim_params()
    grass_params = conf_data.get_grass_params()
    sim_runner = SimulationRunner(sim_params, grass_params)
    # Run simulation
    sim_runner.run().finalize()
    # Check temporary mask and region
    assert int(gscript.parse_command("r.univar", map="out_5by5_v_max", flags="g")["n"]) == 9
    # Check tear down
    assert int(gscript.parse_command("g.region", flags="pg")["cells"]) == init_ncells
    assert int(gscript.parse_command("r.univar", map="z", flags="g")["null_cells"]) == init_nulls


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
def test_fails_when_region_has_no_dem_data(test_data_temp_path):
    gscript.run_command(
        "g.region", res=10, s=100, n=150, w=100, e=150, save="outside_dem", flags="o"
    )

    config_dict = {
        "input": {
            "dem": "z@5by5",
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

    with pytest.raises(ItziFatal, match=r"input map <dem> contains only NULL/NaN cells"):
        SimulationRunner(conf_data.get_sim_params(), conf_data.get_grass_params())
