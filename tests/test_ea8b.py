"""
Integration tests using the EA test case 8b.
The results from itzi are compared with the those from XPSTORM.
"""

import os
import zipfile
from io import StringIO
from configparser import ConfigParser

import pandas as pd
import requests
import pytest
import grass.script as gscript

from itzi import SimulationRunner

TEST8B_URL = (
    "https://zenodo.org/api/records/15256842/files/Test8B_dataset_2010.zip/content"
)
TEST8B_MD5 = "84b865cedd28f8156cfe70b84004b62c"


@pytest.fixture(scope="session")
def test8b_file(test_data_temp_path, helpers):
    """Download the test 8b main file."""
    file_name = "Test8B_dataset_2010.zip"
    file_path = os.path.join(test_data_temp_path, file_name)
    # Check if the file exists and has the right hash
    try:
        assert helpers.md5(file_path) == TEST8B_MD5
    except Exception:
        # Download the file
        print("downloading file from Zenodo...")
        file_response = requests.get(TEST8B_URL, stream=True, timeout=5)
        if file_response.status_code == 200:
            with open(file_path, "wb") as data_file:
                for chunk in file_response.iter_content(chunk_size=8192):
                    data_file.write(chunk)
            print(f"File successfully downloaded to {file_path}")
        else:
            print(f"Failed to download file: Status code {file_response.status_code}")
    return file_path


@pytest.fixture(scope="class")
def ea_test8b(grass_xy_session, test8b_file, test_data_temp_path):
    """Create the GRASS env for ea test 8a."""
    # Keep all generated files in the test_data_temp_path
    os.chdir(test_data_temp_path)
    # Unzip the file
    with zipfile.ZipFile(test8b_file, "r") as zip_ref:
        zip_ref.extractall()
    unzip_path = os.path.join(test_data_temp_path, "Test8B dataset 2010")
    # Create new mapset
    mapset_name = "ea8b"
    gscript.run_command("g.mapset", mapset=mapset_name, flags="c")
    # Define the region
    region = gscript.parse_command(
        "g.region", res=2, s=664408, w=263976, e=264940, n=664808, flags="g"
    )
    assert int(region["rows"]) == 200
    assert int(region["cols"]) == 482
    # DEM
    dem_path = os.path.join(unzip_path, "Test8DEM.asc")
    gscript.run_command("r.in.gdal", input=dem_path, output="dem50cm")
    gscript.run_command("r.resamp.stats", input="dem50cm", output="dem2m")
    univar_dem = gscript.parse_command("r.univar", map="dem2m", flags="g")
    assert int(univar_dem["null_cells"]) == 0
    # Buildings
    buildings_path = os.path.join(unzip_path, "Test8Buildings.asc")
    gscript.run_command("r.in.gdal", input=buildings_path, output="buildings")
    gscript.mapcalc("dem2m_buildings=if(isnull(buildings), dem2m, dem2m+5)")
    # Manning
    road_path = os.path.join(unzip_path, "Test8RoadPavement.asc")
    gscript.run_command("r.in.gdal", input=road_path, output="road50cm")
    gscript.mapcalc("n=if(isnull(road50cm), 0.05, 0.02)")
    univar_n = gscript.parse_command("r.univar", map="n", flags="g")
    assert float(univar_n["min"]) == 0.02
    assert float(univar_n["max"]) == 0.05
    assert int(univar_n["null_cells"]) == 0
    # Output points #
    stages_path = os.path.join(unzip_path, "Test8Output.csv")
    gscript.run_command(
        "v.in.ascii",
        input=stages_path,
        output="output_points",
        format="point",
        sep="comma",
        skip=1,
        cat=1,
        x=2,
        y=3,
    )
    # Manhole location
    gscript.write_command(
        "v.in.ascii", input="-", stdin="264895|664747", output="manhole_location"
    )
    return None


@pytest.fixture(scope="session")
def ea_test8b_reference(test_data_path):
    """Take the results from xpstorm as reference."""
    col_names = ["Time", "results"]
    file_path = os.path.join(test_data_path, "EA_test_8", "b", "xpstorm.csv")
    df_ref = pd.read_csv(file_path, index_col=0, names=col_names)
    # Convert to seconds
    df_ref.index *= 60.0
    # Round time to 10 ms
    df_ref.index = df_ref.index.round(decimals=2)
    # convert indices to timedelta
    df_ref.index = pd.to_timedelta(df_ref.index, unit="s")
    # to series
    ds_ref = df_ref.squeeze()
    return ds_ref


@pytest.fixture(scope="class")
def ea_test8b_sim(ea_test8b, test_data_path, test_data_temp_path):
    """ """
    # Keep all generated files in the test_data_temp_path
    os.chdir(test_data_temp_path)
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    assert current_mapset == "ea8b"
    inp_file = os.path.join(
        test_data_path, "EA_test_8", "b", "test8b_drainage_ponding.inp"
    )
    config_dict = {
        "time": {"duration": "03:20:00", "record_step": "00:00:30"},
        "input": {"dem": "dem2m_buildings", "friction": "n"},
        "output": {"prefix": "out", "values": "h, drainage_stats"},
        "options": {"theta": 0.7, "cfl": 0.5},
        "drainage": {
            "swmm_inp": inp_file,
            "orifice_coeff": 1,
            "output": "out_drainage",
        },
    }
    parser = ConfigParser()
    parser.read_dict(config_dict)
    conf_file = os.path.join(test_data_temp_path, "ea2dt8b.ini")
    with open(conf_file, "w") as f:
        parser.write(f)
    sim_runner = SimulationRunner()
    sim_runner.initialize(conf_file)
    sim_runner.run().finalize()
    return sim_runner


@pytest.fixture(scope="class")
def ea8b_itzi_drainage_results(ea_test8b_sim):
    """Extract linkage flow from the drainage network"""
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    assert current_mapset == "ea8b"
    select_col = ["start_time", "linkage_flow"]
    itzi_results = gscript.read_command("t.vect.db.select", input="out_drainage")
    # translate to Pandas dataframe and keep only linkage_flow with start_time over 3000
    df_itzi_results = pd.read_csv(StringIO(itzi_results), sep="|")[select_col]
    df_itzi_results = df_itzi_results[df_itzi_results.start_time >= 3000]
    df_itzi_results.set_index(
        "start_time", drop=True, inplace=True, verify_integrity=True
    )
    # convert indices to timedelta
    df_itzi_results.index = pd.to_timedelta(df_itzi_results.index, unit="s")
    # to series
    ds_itzi_results = df_itzi_results.squeeze()
    return ds_itzi_results


@pytest.mark.slow
def test_ea8b(ea_test8b_reference, ea8b_itzi_drainage_results, helpers):
    """Compare results with XPSTORM"""
    # Extract results at output points
    # itzi_results = gscript.read_command('t.rast.what', points='manhole_location',
    #                                     strds='out_drainage_stats', null_value='*',
    #                                     separator='comma', layout='col')
    ds_itzi_results = ea8b_itzi_drainage_results
    ds_ref = ea_test8b_reference
    # calculate NSE
    nse = helpers.get_nse(ds_itzi_results, ds_ref)
    rsr = helpers.get_rsr(ds_itzi_results, ds_ref)
    assert nse > 0.99
    assert rsr < 0.01
