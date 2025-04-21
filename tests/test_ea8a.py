"""
Integration tests using the EA test case 8a.
The results from itzi are compared with the those from LISFLOOD-FP.
"""

import os
from io import StringIO
import zipfile
from collections import namedtuple

import pytest
import requests
import numpy as np
import pandas as pd
import grass.script as gscript

from itzi import SimulationRunner

MapInfo = namedtuple("MapInfo", ["name", "start", "end", "value"])

TEST8A_URL = (
    "https://zenodo.org/api/records/15256842/files/Test8A_dataset_2010.zip/content"
)
TEST8A_MD5 = "46b589ee000ff87c9077fcc51fa71e8e"


@pytest.fixture(scope="session")
def test8a_file(test_data_temp_path, helpers):
    """Download the test 8a main file."""
    file_name = "Test8A_dataset_2010.zip"
    file_path = os.path.join(test_data_temp_path, file_name)
    # Check if the file exists and has the right hash
    try:
        assert helpers.md5(file_path) == TEST8A_MD5
    except Exception:
        # Download the file
        print("downloading file from Zenodo...")
        file_response = requests.get(TEST8A_URL, stream=True, timeout=5)
        if file_response.status_code == 200:
            with open(file_path, "wb") as data_file:
                for chunk in file_response.iter_content(chunk_size=8192):
                    data_file.write(chunk)
            print(f"File successfully downloaded to {file_path}")
        else:
            print(f"Failed to download file: Status code {file_response.status_code}")
    return file_path


@pytest.fixture(scope="function")
def ea_test8a(grass_xy_session, test8a_file, test_data_path, test_data_temp_path):
    """Create the GRASS env for ea test 8a."""

    # Unzip the file
    with zipfile.ZipFile(test8a_file, "r") as zip_ref:
        zip_ref.extractall()
    unzip_path = os.path.join(test_data_temp_path, "Test8A dataset 2010")
    # Create new mapset
    mapset_name = "ea8a"
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
    # Manning
    road_path = os.path.join(unzip_path, "Test8RoadPavement.asc")
    gscript.run_command("r.in.gdal", input=road_path, output="road50cm")
    gscript.mapcalc("n=if(isnull(road50cm), 0.05, 0.02)")
    univar_n = gscript.parse_command("r.univar", map="n", flags="g")
    assert float(univar_n["min"]) == 0.02
    assert float(univar_n["max"]) == 0.05
    assert int(univar_n["null_cells"]) == 0
    # Rainfall
    rain_maps = [
        MapInfo("rain0", 0, 60, 0),
        MapInfo("rain1", 60, 240, 400),
        MapInfo("rain2", 240, 18000, 0),
    ]
    for rain_map in rain_maps:
        gscript.mapcalc(f"{rain_map.name}={rain_map.value}")
        gscript.run_command(
            "t.register",
            type="raster",
            maps=rain_map.name,
            start=rain_map.start,
            end=rain_map.end,
            unit="seconds",
        )
    gscript.run_command(
        "t.create",
        type="strds",
        temporaltype="relative",
        output="rainfall",
        title="rainfall",
        descr="rainfall",
    )
    gscript.run_command(
        "t.register", type="raster", input="rainfall", maps=[i.name for i in rain_maps]
    )
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
    # Point inflow #
    # Read the inflow point location
    point_path = os.path.join(unzip_path, "Test8A-inflow-location.csv")
    gscript.run_command(
        "v.in.ascii",
        input=point_path,
        output="inflow_point",
        skip=1,
        format="point",
        sep="comma",
    )
    # Transform to raster using the CSV flux values, and register map in STRDS
    inflow_path = os.path.join(test_data_path, "EA_test_8", "a", "point-inflow.csv")
    df_flow = pd.read_csv(inflow_path)
    for idx, row in df_flow.iterrows():
        gscript.run_command(
            "v.to.rast",
            input="inflow_point",
            output=f"inflow{idx}",
            type="point",
            use="val",
            value=row.flux,
        )
        gscript.run_command("r.null", map=f"inflow{idx}", null=0)
        gscript.run_command(
            "t.register",
            type="raster",
            maps=f"inflow{idx}",
            start=int(row.start),
            end=int(row.end),
            unit="seconds",
        )
    gscript.run_command(
        "t.create",
        type="strds",
        temporaltype="relative",
        output="inflow",
        title="inflow",
        descr="inflow",
    )
    gscript.run_command(
        "t.register",
        type="raster",
        input="inflow",
        maps=[f"inflow{i}" for i, r in df_flow.iterrows()],
    )
    # Linear interpolation of the STRDS
    gscript.run_command("t.rast.gapfill", input="inflow", basename="inflow", quiet=True)
    return None


@pytest.fixture(scope="session")
def ea_test8a_reference(test_data_path):
    """Take the results from LISFLOOD-FP as reference."""
    col_names = ["Time (min)"] + list(range(1, 10))
    file_path = os.path.join(test_data_path, "EA_test_8", "a", "ea2dt8a.stage")
    df_ref = pd.read_csv(
        file_path, sep="    ", header=15, index_col=0, engine="python", names=col_names
    )
    # Convert to minutes
    df_ref.index /= 60.0
    # round entries
    df_ref.index = np.round(df_ref.index, 1)
    return df_ref


@pytest.fixture(scope="function")
def ea_test8a_sim(ea_test8a, test_data_path):
    """ """
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    assert current_mapset == "ea8a"
    config_file = os.path.join(test_data_path, "EA_test_8", "a", "ea2dt8a.ini")
    sim_runner = SimulationRunner()
    sim_runner.initialize(config_file)
    sim_runner.run().finalize()
    return sim_runner


@pytest.mark.slow
def test_ea8a(ea_test8a_reference, ea_test8a_sim):
    """Compare results with LISFLOOD-FP"""
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    assert current_mapset == "ea8a"
    # Extract results at output points
    itzi_results = gscript.read_command(
        "t.rast.what",
        points="output_points",
        strds="out_h",
        null_value="*",
        separator="comma",
        layout="col",
    )
    # Read results as Pandas dataframe
    col_names = ["Time (min)"] + list(range(1, 10))
    df_itzi = pd.read_csv(
        StringIO(itzi_results), index_col=0, header=1, names=col_names, na_values="*"
    )
    df_itzi.fillna(0.0, inplace=True)
    df_itzi.index = pd.to_numeric(df_itzi.index)
    df_itzi.index /= 60.0
    del df_itzi["Time (min)"]
    # Compute the absolute error
    abs_error = np.abs(df_itzi - ea_test8a_reference)
    # Compute MAE for each point
    points_values = []
    for pt_idx in range(1, 9):
        col_idx = [ea_test8a_reference[pt_idx], df_itzi[pt_idx], abs_error[pt_idx]]
        col_keys = ["lisflood", "itzi", "absolute error"]
        new_df = pd.concat(col_idx, axis=1, keys=col_keys)
        new_df.index.name = "Time (min)"
        # Keep only non null values
        new_df = new_df[new_df.itzi.notnull()]
        points_values.append(new_df)
    # Check if MAE is below threshold
    for df_err in points_values:
        mae = np.mean(df_err["absolute error"])
        assert mae <= 0.04
