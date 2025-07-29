"""Make sure that the tutorial from the official documentation works."""

import os
from io import StringIO
import requests
import pathlib
import tempfile
from configparser import ConfigParser
from dataclasses import fields

import pytest
import pandas as pd
import grass.script as gscript

from itzi import SimulationRunner
from itzi.data_containers import DrainageNodeAttributes, DrainageLinkAttributes

DEM_URL = "https://zenodo.org/api/records/15009114/files/elev_lid792_1m.gtiff/content"
MD5_SUM = "224f73dfa37244722b879a5f653682c9"
DATA_EPSG = "3358"


@pytest.fixture(scope="session")
def tutorial_test_file(test_data_temp_path, helpers):
    """Download the tutorial main file."""
    file_name = "elev_lid792_1m.gtiff"
    file_path = os.path.join(test_data_temp_path, file_name)
    # Check if the file exists and has the right hash
    try:
        assert helpers.md5(file_path) == MD5_SUM
    except Exception:
        # Download the file
        print("downloading file from Zenodo...")
        file_response = requests.get(DEM_URL, stream=True, timeout=5)
        if file_response.status_code == 200:
            with open(file_path, "wb") as data_file:
                for chunk in file_response.iter_content(chunk_size=8192):
                    data_file.write(chunk)
            print(f"File successfully downloaded to {file_path}")
        else:
            print(f"Failed to download file: Status code {file_response.status_code}")
    return file_path


@pytest.fixture(scope="class")
def grass_tutorial_session(test_data_temp_path):
    """Create a GRASS session in a new location and PERMANENT mapset"""
    # Keep all generated files in the test_data_temp_path
    os.chdir(test_data_temp_path)
    tmpdir = tempfile.TemporaryDirectory()
    gscript.create_project(tmpdir.name, name="itzi_tutorial", epsg=DATA_EPSG)
    grass_session = gscript.setup.init(
        path=tmpdir.name, location="itzi_tutorial", mapset="PERMANENT"
    )
    os.environ["GRASS_VERBOSE"] = "1"
    yield grass_session
    grass_session.finish()
    tmpdir.cleanup()


@pytest.fixture(scope="class")
def itzi_tutorial(grass_tutorial_session, tutorial_test_file):
    """Create the GRASS env for the tutorial.
    Repeat the steps described in the tutorial.
    """
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
    watershed_out_file = pathlib.Path("watershed_out.txt")
    with watershed_out_file.open("w") as file:
        file.write("638888|220011")
    gscript.run_command("v.in.ascii", input="watershed_out.txt", output="watershed_out")
    watershed_out_file.unlink()
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
    return None


@pytest.mark.slow
@pytest.mark.usefixtures("itzi_tutorial", "test_data_path")
class TestItziTutorial:
    def test_tutorial(itzi_tutorial, test_data_path):
        """Run the tutorial simulation. Check the results."""
        # Run the simulation
        config_file = os.path.join(test_data_path, "tutorial_files", "tutorial.ini")
        sim_runner = SimulationRunner()
        sim_runner.initialize(config_file)
        sim_runner.run().finalize()
        # Check the results
        h_max_univar = gscript.parse_command(
            "r.univar", map="nc_itzi_tutorial_water_depth_max", flags="g"
        )
        assert float(h_max_univar["max"]) == pytest.approx(2.298454, abs=1e-2)
        assert float(h_max_univar["mean_of_abs"]) == pytest.approx(0.0355, abs=1e-3)

    def test_tutorial_drainage(itzi_tutorial, test_data_path, test_data_temp_path, helpers):
        """Run the tutorial simulation with drainage."""
        # Set the config file dynamically to make sure it can find the INP file
        inp_file = os.path.join(test_data_path, "tutorial_files", "tutorial_drainage.inp")
        config_dict = {
            "time": {"duration": "00:30:00", "record_step": "00:00:30"},
            "input": {
                "dem": "elev_lid792_5m",
                "friction": "n",
                "rain": "rain",
                "bctype": "bctype",
                "bcvalue": "bcvalue",
            },
            "output": {"prefix": "nc_itzi_tutorial_drainage", "values": "water_depth, v, vdir"},
            "statistics": {"stats_file": "nc_itzi_tutorial_drainage.csv"},
            "options": {"cfl": 0.7, "theta": 0.9, "dtmax": 0.5},
            "drainage": {"swmm_inp": inp_file, "output": "nc_itzi_tutorial_drainage"},
        }
        parser = ConfigParser()
        parser.read_dict(config_dict)
        config_file = os.path.join(test_data_temp_path, "tutorial_drainage.ini")
        with open(config_file, "w") as f:
            parser.write(f)

        # Run the simulation
        sim_runner = SimulationRunner()
        sim_runner.initialize(config_file)
        sim_runner.run().finalize()

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
        expected_link_columns = ["cat"] + [field.name for field in fields(DrainageLinkAttributes)]
        assert expected_link_columns == actual_link_columns

        # Check nodes DB table
        v_db_select = gscript.read_command("v.db.select", map=map_name, layer=1).split("\n")[:-1]
        # Number of nodes exported to DB
        nodes_entries = v_db_select[1:]
        assert len(nodes_entries) == 3
        # node DB columns are as expected
        actual_nodes_columns = v_db_select[0].split("|")
        expected_nodes_columns = ["cat"] + [field.name for field in fields(DrainageNodeAttributes)]
        assert expected_nodes_columns == actual_nodes_columns
