"""Make sure that the tutorial from the official documentation works."""

import os
import pathlib
import shutil
from collections.abc import Iterator
from configparser import ConfigParser
from io import StringIO

import grass.script as gscript
import numpy as np
import pandas as pd
import pytest
from grass.script.setup import SessionHandle
from itzi_core.data_containers import DrainageLinkAttributes, DrainageNodeAttributes

from itzi import SimulationRunner
from itzi.configreader import ConfigReader
from itzi.grass import session

DATA_EPSG = "3358"


@pytest.fixture(scope="class")
def tutorial_test_file(test_data_path: str) -> pathlib.Path:
    """Return the tutorial DEM managed by Git LFS."""
    return pathlib.Path(test_data_path) / "tutorial_files" / "elev_lid792_1m.gtiff"


@pytest.fixture(scope="class")
def grass_tutorial_session(
    grass_xy_session: SessionHandle, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[SessionHandle]:
    """Use the shared session file when switching to a projected project."""
    import grass.temporal as tgis
    from grass.pygrass.gis import set_current_mapset

    original = gscript.gisenv()
    tmp_path = tmp_path_factory.mktemp("grass_tutorial")
    gscript.create_project(tmp_path, name="itzi_tutorial", epsg=DATA_EPSG)
    set_current_mapset("PERMANENT", "itzi_tutorial", str(tmp_path))
    tgis.stop_subprocesses()
    tgis.core.c_library_interface = None
    try:
        session.GrassSessionManager.ensure_temporal_initialized()
        yield grass_xy_session
    finally:
        tgis.stop_subprocesses()
        tgis.core.c_library_interface = None
        set_current_mapset(original["MAPSET"], original["LOCATION_NAME"], original["GISDBASE"])
        session._initialized_temporal_session = None


@pytest.fixture(scope="class")
def itzi_tutorial_template(
    grass_tutorial_session: SessionHandle,
    tutorial_test_file: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Build the tutorial input data once and save a pristine mapset copy."""
    # import DEM
    gscript.run_command("r.in.gdal", input=tutorial_test_file, output="elev_lid792_1m")
    # Adjust the region
    gscript.run_command("g.region", raster="elev_lid792_1m", res="5")
    # Resample DEM
    gscript.run_command("r.resamp.interp", input="elev_lid792_1m", output="elev_lid792_5m")
    univar_dem = gscript.parse_command("r.univar", map="elev_lid792_5m", flags="g")
    assert int(univar_dem["null_cells"]) == 0
    # Create raster mask
    gscript.run_command(
        "r.watershed", elevation="elev_lid792_5m", drainage="elev_lid792_5m_drainage"
    )
    gscript.run_command(
        "r.water.outlet",
        input="elev_lid792_5m_drainage",
        output="watershed",
        coordinates="638888,220011",
    )
    gscript.run_command("r.mask", rast="watershed")
    # Create boundary condition maps
    gscript.write_command("v.in.ascii", input="-", output="watershed_out", stdin="638888|220011")
    gscript.run_command(
        "v.to.rast",
        input="watershed_out",
        type="point",
        output="bctype",
        use="val",
        value=4,
    )
    gscript.run_command(
        "v.to.rast",
        input="watershed_out",
        type="point",
        output="bcvalue",
        use="val",
        value=0,
    )
    # Create rainfall and friction maps
    gscript.run_command("r.mapcalc", exp="rain=100")
    gscript.run_command("r.mapcalc", exp="n=0.05")
    current = gscript.gisenv()
    mapset_path = pathlib.Path(current["GISDBASE"]) / current["LOCATION_NAME"] / "PERMANENT"
    template_path = tmp_path_factory.mktemp("grass_tutorial_template") / "PERMANENT"
    shutil.copytree(mapset_path, template_path)
    return mapset_path, template_path


@pytest.fixture
def itzi_tutorial(
    itzi_tutorial_template: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    """Restore fresh tutorial data without restarting GRASS helpers."""
    mapset_path, template_path = itzi_tutorial_template
    shutil.rmtree(mapset_path)
    shutil.copytree(template_path, mapset_path)
    os.chdir(tmp_path)


@pytest.mark.slow
@pytest.mark.usefixtures("itzi_tutorial", "test_data_path")
class TestItziTutorial:
    def test_tutorial(itzi_tutorial, test_data_path):
        """Run the tutorial simulation. Check the results."""
        # Run the simulation
        config_file = os.path.join(test_data_path, "tutorial_files", "tutorial.ini")
        conf_data = ConfigReader(config_file)

        with session.GrassSessionManager(conf_data.grass_params):
            sim_runner = SimulationRunner(
                conf_data.sim_config,
                stats_file=conf_data.stats_file,
            ).initialize()
            try:
                sim_runner.run()
            finally:
                sim_runner.finalize()
        # Check the results
        hmax_maps = gscript.list_grouped("raster", pattern="nc_itzi_tutorial_max_water_depth_*")[
            "PERMANENT"
        ]
        h_max_univar = gscript.parse_command("r.univar", map=max(hmax_maps), flags="g")
        assert float(h_max_univar["max"]) == pytest.approx(2.298454, abs=1e-2)
        assert float(h_max_univar["mean_of_abs"]) == pytest.approx(0.0355, abs=1e-3)

        # Test consistency of stats file
        stat_file_path = pathlib.Path("nc_itzi_tutorial.csv")
        df_stats = pd.read_csv(stat_file_path, na_values="-")
        df_stats.set_index("simulation_time", drop=True, inplace=True, verify_integrity=True)
        df_stats.index = pd.to_timedelta(df_stats.index)
        volume_columns = [
            "boundary_volume",
            "rainfall_volume",
            "infiltration_volume",
            "inflow_volume",
            "losses_volume",
            "drainage_network_volume",
            "created_volume",
        ]
        sum_inputs = df_stats[volume_columns].sum(axis=1)
        assert np.all(np.isclose(sum_inputs, df_stats["volume_change"], atol=1)), (
            "'volume_change' inconsistent with other values"
        )

    def test_tutorial_drainage(itzi_tutorial, test_data_path, helpers):
        """Run the tutorial simulation with drainage."""
        # Set the config file dynamically to make sure it can find the INP file
        inp_file = os.path.join(test_data_path, "tutorial_files", "tutorial_drainage.inp")
        config_dict = {
            "time": {"duration": "00:30:00", "record_step": "00:00:30"},
            "input": {
                "ground_elevation": "elev_lid792_5m",
                "friction": "n",
                "rainfall_rate": "rain",
                "boundary_type": "bctype",
                "boundary_value": "bcvalue",
            },
            "output": {
                "prefix": "nc_itzi_tutorial_drainage",
                "values": "water_depth, flow_speed, flow_velocity_direction",
            },
            "statistics": {"stats_file": "nc_itzi_tutorial_drainage.csv"},
            "options": {"cfl": 0.7, "theta": 0.9, "dtmax": 0.5},
            "drainage": {"swmm_inp": inp_file, "output": "nc_itzi_tutorial_drainage"},
        }
        parser = ConfigParser()
        parser.read_dict(config_dict)
        config_file = "tutorial_drainage.ini"
        with open(config_file, "w") as f:
            parser.write(f)

        # Run the simulation
        conf_data = ConfigReader(config_file)
        with session.GrassSessionManager(conf_data.grass_params):
            sim_runner = SimulationRunner(
                conf_data.sim_config,
                stats_file=conf_data.stats_file,
            ).initialize()
            try:
                sim_runner.run()
            finally:
                sim_runner.finalize()

        # Test consistency of stats file
        stat_file_path = pathlib.Path("nc_itzi_tutorial_drainage.csv")
        df_stats = pd.read_csv(stat_file_path, na_values="-")
        df_stats.set_index("simulation_time", drop=True, inplace=True, verify_integrity=True)
        df_stats.index = pd.to_timedelta(df_stats.index)
        volume_columns = [
            "boundary_volume",
            "rainfall_volume",
            "infiltration_volume",
            "inflow_volume",
            "losses_volume",
            "drainage_network_volume",
            "created_volume",
        ]
        sum_inputs = df_stats[volume_columns].sum(axis=1)
        assert np.all(np.isclose(sum_inputs, df_stats["volume_change"], atol=1)), (
            "'volume_change' inconsistent with other values"
        )

        # Check the results at the entry node
        select_cols = ["start_time", "coupling_flow"]
        node_results = gscript.read_command(
            "t.vect.db.select",
            input="nc_itzi_tutorial_drainage",
            where="node_id=='J0'",
        )
        # To pandas
        df_node = pd.read_csv(StringIO(node_results), sep="|")[select_cols]
        df_node.set_index("start_time", drop=True, inplace=True, verify_integrity=True)
        # convert indices to timedelta
        df_node.index = pd.to_timedelta(df_node.index, unit="s")
        # to series
        ts_drainage = df_node.squeeze()
        # Check stability
        drainage_roughness = helpers.roughness(ts_drainage)
        drainage_autocorrelation = ts_drainage.autocorr(lag=1)
        assert drainage_roughness < 3
        assert drainage_autocorrelation > 0.99

        ## Check integrity of vector file ##
        map_name = "nc_itzi_tutorial_drainage_0047"

        gscript.run_command("db.connect", flags="d")
        # Number of geometric features
        topology = gscript.parse_command("v.info", flags="t", map=map_name)
        assert int(topology["points"]) == 2
        assert int(topology["lines"]) == 1

        # Check layer connection for nodes
        v_info_layer1 = gscript.parse_command("v.info", flags="e", layer=1, map=map_name)
        assert 2 == int(v_info_layer1["num_dblinks"])
        assert 1 == int(v_info_layer1["attribute_layer_number"])
        assert "node" == v_info_layer1["attribute_layer_name"]
        assert f"{map_name}_node" == v_info_layer1["attribute_table"]
        assert "cat" == v_info_layer1["attribute_primary_key"]

        # Check layer connection for links
        v_info_layer2 = gscript.parse_command("v.info", flags="e", layer=2, map=map_name)
        assert 2 == int(v_info_layer2["num_dblinks"])
        assert 2 == int(v_info_layer2["attribute_layer_number"])
        assert "link" == v_info_layer2["attribute_layer_name"]
        assert f"{map_name}_link" == v_info_layer2["attribute_table"]
        assert "cat" == v_info_layer2["attribute_primary_key"]

        # Make sure points and lines are only written in layers 1 and 2, respectively
        v_category = gscript.parse_command(
            "v.category", input=map_name, option="report", flags="g"
        )
        layer_list = []
        for layer in v_category:
            layer_lst = layer.split(" ")[:2]
            layer_str = " ".join(layer_lst)
            layer_list.append(layer_str)
        assert "1 point" in layer_list
        assert "2 point" not in layer_list
        assert "2 line" in layer_list
        assert "1 line" not in layer_list

        # Check links DB table
        v_db_select = gscript.read_command("v.db.select", map=map_name, layer=2).split("\n")[:-1]
        # Number of links exported to DB
        link_entries = v_db_select[1:]
        assert len(link_entries) == 2
        # link DB columns are as expected
        actual_link_columns = v_db_select[0].split("|")
        expected_link_columns = ["cat"] + list(DrainageLinkAttributes.model_fields.keys())
        assert expected_link_columns == actual_link_columns

        # Check nodes DB table
        v_db_select = gscript.read_command("v.db.select", map=map_name, layer=1).split("\n")[:-1]
        # Number of nodes exported to DB
        nodes_entries = v_db_select[1:]
        assert len(nodes_entries) == 3
        # node DB columns are as expected
        actual_nodes_columns = v_db_select[0].split("|")
        expected_nodes_columns = ["cat"] + list(DrainageNodeAttributes.model_fields.keys())
        assert expected_nodes_columns == actual_nodes_columns

    def test_tutorial_drainage_without_vector_output(itzi_tutorial, test_data_path):
        """Run coupled drainage without creating vector output."""
        config_file = os.path.join(test_data_path, "tutorial_files", "tutorial_drainage.ini")
        conf_data = ConfigReader(config_file)
        with session.GrassSessionManager(conf_data.grass_params):
            sim_runner = SimulationRunner(
                conf_data.sim_config,
                stats_file=conf_data.stats_file,
            ).initialize()
            try:
                assert sim_runner.sim.report.vector_provider is None
                sim_runner.run()
            finally:
                sim_runner.finalize()
