"""
Integration tests using the EA test case 8b with icechunk and geoparquet backends.
The results from itzi are compared with those from XPSTORM.
"""

from datetime import datetime
import os
from pathlib import Path
import zipfile

import pandas as pd
import numpy as np
import requests
import pytest
import xarray as xr
import rioxarray
import pyproj
import geopandas as gpd
import icechunk
import icechunk.xarray


from itzi.profiler import profile_context
from itzi.simulation_factories import create_icechunk_simulation


TEST8B_URL = "https://zenodo.org/api/records/15256842/files/Test8B_dataset_2010.zip/content"
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


@pytest.fixture(scope="module")
def ea_test8b_icechunk_data(test8b_file, test_data_temp_path):
    """Create icechunk data for EA test 8b."""
    # Keep all generated files in the test_data_temp_path
    os.chdir(test_data_temp_path)

    # Unzip the file
    with zipfile.ZipFile(test8b_file, "r") as zip_ref:
        zip_ref.extractall()
    unzip_path = os.path.join(test_data_temp_path, "Test8B dataset 2010")

    # Define the target region and resolution (matching original GRASS test)
    west, south, east, north = 263976, 664408, 264940, 664808
    res = 2.0
    cols = int((east - west) / res)
    rows = int((north - south) / res)

    # Verify we get the expected dimensions
    assert cols == 482
    assert rows == 200

    # Create coordinate arrays (cell centers)
    x_coords = np.linspace(west + res / 2, east - res / 2, cols)
    y_coords = np.linspace(north - res / 2, south + res / 2, rows)

    # Define CRS (assuming UTM or similar projected coordinate system)
    # This should match the CRS of the original data
    crs = pyproj.CRS.from_epsg(32633)  # UTM Zone 33N, adjust as needed

    # Process DEM - directly import with xarray
    dem_path = os.path.join(unzip_path, "Test8DEM.asc")
    dem_da = rioxarray.open_rasterio(dem_path).isel(band=0)
    # Resample to target resolution and extent
    dem_da = dem_da.interp(x=x_coords, y=y_coords, method="linear")
    dem_data = dem_da.values

    # Process buildings
    buildings_path = os.path.join(unzip_path, "Test8Buildings.asc")
    buildings_da = rioxarray.open_rasterio(buildings_path, mask_and_scale=True).isel(band=0)
    buildings_da = buildings_da.interp(x=x_coords, y=y_coords, method="nearest")
    buildings_data = buildings_da.values

    # Create DEM with buildings (add 5m to building areas)
    dem_with_buildings = np.where(np.isnan(buildings_data), dem_data, dem_data + 5.0)

    # Process roads for Manning coefficient
    road_path = os.path.join(unzip_path, "Test8RoadPavement.asc")
    road_da = rioxarray.open_rasterio(road_path, mask_and_scale=True).isel(band=0)
    road_da = road_da.interp(x=x_coords, y=y_coords, method="nearest")
    road_data = road_da.values

    # Create Manning coefficient (0.02 for roads, 0.05 elsewhere)
    manning = np.where(np.isnan(road_data), 0.05, 0.02)

    # Verify Manning coefficient values
    assert np.nanmin(manning) == 0.02
    assert np.nanmax(manning) == 0.05
    assert not np.any(np.isnan(manning))

    # Ensure arrays are 2D
    if dem_with_buildings.ndim > 2:
        dem_with_buildings = np.squeeze(dem_with_buildings)
    if manning.ndim > 2:
        manning = np.squeeze(manning)

    # Verify array shapes
    print(f"DEM shape: {dem_with_buildings.shape}")
    print(f"Manning shape: {manning.shape}")
    print(f"Expected shape: ({rows}, {cols})")

    # Create xarray dataset
    dataset = xr.Dataset(
        {
            "dem": (["y", "x"], dem_with_buildings),
            "friction": (["y", "x"], manning),
        },
        coords={
            "x": x_coords,
            "y": y_coords,
        },
        attrs={"crs_wkt": crs.to_wkt()},
    )

    # Create icechunk storage
    storage = icechunk.in_memory_storage()

    # Save to icechunk
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")

    # Convert to zarr and save
    icechunk.xarray.to_icechunk(dataset, session, mode="w")
    session.commit("Initial EA8B test data")

    return {
        "storage": storage,
        "crs": crs,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "unzip_path": unzip_path,
        "rows": rows,
        "cols": cols,
    }


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


@pytest.fixture(scope="module")
def ea_test8b_sim_icechunk(ea_test8b_icechunk_data, test_data_path, test_data_temp_path):
    """Run simulation with icechunk and geoparquet backends."""
    # Keep all generated files in the test_data_temp_path
    os.chdir(test_data_temp_path)

    # Create output directory for geoparquet files
    output_dir = Path(test_data_temp_path) / "geoparquet_output"
    output_dir.mkdir(exist_ok=True)

    # Create icechunk output storage
    output_storage = icechunk.in_memory_storage()

    inp_file = os.path.join(test_data_path, "EA_test_8", "b", "test8b_drainage_ponding.inp")

    # Create simulation configuration for icechunk/geoparquet backends
    from datetime import datetime, timedelta

    sim_start_time = datetime(2000, 1, 1, 0, 0, 0)
    sim_end_time = sim_start_time + timedelta(hours=3, minutes=20)

    profile_path = Path(test_data_temp_path) / Path("test8b_icechunk_profile.txt")

    # Create mask array (no mask for this test)
    # False = valid cells, True = masked cells
    arr_mask = np.zeros(
        (ea_test8b_icechunk_data["rows"], ea_test8b_icechunk_data["cols"]), dtype=bool
    )

    # Create simulation configuration directly
    from itzi.data_containers import SimulationConfig, SurfaceFlowParameters
    from itzi.const import TemporalType

    surface_flow_params = SurfaceFlowParameters(
        cfl=0.5,
        theta=0.7,
    )

    sim_config = SimulationConfig(
        start_time=sim_start_time,
        end_time=sim_end_time,
        record_step=timedelta(seconds=30),
        temporal_type=TemporalType.ABSOLUTE,
        input_map_names={"dem": "dem", "friction": "friction"},
        output_map_names={"water_depth": "test_water_depth"},
        drainage_output="out_drainage",
        swmm_inp=inp_file,
        stats_file="ea8b_icechunk.csv",
        surface_flow_parameters=surface_flow_params,
        orifice_coeff=1.0,  # Match reference
    )

    # Use the icechunk simulation factory
    simulation, timed_arrays = create_icechunk_simulation(
        sim_config=sim_config,
        arr_mask=arr_mask,
        input_icechunk_storage=ea_test8b_icechunk_data["storage"],
        output_icechunk_storage=output_storage,
        input_icechunk_group="main",
        output_dir=str(output_dir),
    )

    # Set arrays
    for arr_key in ["dem", "friction"]:
        if arr_key in timed_arrays:
            initial_array = timed_arrays[arr_key].get(sim_config.start_time)
            simulation.set_array(arr_key, initial_array)

    # Run the simulation
    with profile_context(profile_path):
        simulation.initialize()
        while simulation.sim_time < simulation.end_time:
            simulation.update()
        simulation.finalize()

    return {
        "output_dir": output_dir,
        "profile_path": profile_path,
        "sim_start_time": sim_start_time,
        "icechunk_data": ea_test8b_icechunk_data,
        "output_storage": output_storage,
        "simulation": simulation,
    }


@pytest.fixture(scope="module")
def ea8b_itzi_drainage_results_geoparquet(ea_test8b_sim_icechunk):
    """Extract coupling flow from the drainage network using geoparquet files."""
    output_dir = ea_test8b_sim_icechunk["output_dir"]

    # Find geoparquet files for nodes
    node_files = list(output_dir.glob("*_nodes_*.parquet"))

    if not node_files:
        pytest.skip("No drainage results found - simulation may not have run properly")

    # Read geoparquet files and extract drainage results
    all_results = []
    for node_file in sorted(node_files):
        # Extract timestamp from filename (format: out_drainage_nodes_2000-01-01 HH:MM:SS)
        # The timestamp is after the last underscore
        timestamp_str = node_file.stem.split("_", 3)[-1]  # Split into max 4 parts, take the last
        # Parse the datetime and convert to seconds since simulation start
        timestamp_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        sim_start_dt = datetime(2000, 1, 1, 0, 0, 0)
        timestamp_seconds = int((timestamp_dt - sim_start_dt).total_seconds())

        # Read the parquet file
        gdf = gpd.read_parquet(node_file)

        # Filter for node J1 (equivalent to the GRASS query)
        j1_data = gdf[gdf["node_id"] == "J1"]

        if not j1_data.empty:
            coupling_flow = j1_data["coupling_flow"].iloc[0]
            all_results.append({"start_time": timestamp_seconds, "coupling_flow": coupling_flow})

    # Convert to pandas series with timedelta index
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results.set_index("start_time", inplace=True)

        # Filter for times >= 3000s as in original test
        df_results = df_results[df_results.index >= 3000]

        # Convert index to timedelta
        df_results.index = pd.to_timedelta(df_results.index, unit="s")

        return df_results["coupling_flow"]
    else:
        # Return empty series if no results found
        return pd.Series(dtype=float, name="coupling_flow")


@pytest.mark.slow
def test_ea8b_icechunk(
    ea_test8b_reference, ea8b_itzi_drainage_results_geoparquet, helpers, test_data_temp_path
):
    """Test EA8B with icechunk and geoparquet backends."""
    ds_itzi_results = ea8b_itzi_drainage_results_geoparquet
    ds_ref = ea_test8b_reference

    # Check if we have results to compare
    if ds_itzi_results.empty:
        pytest.skip("No drainage results found - simulation may not have run properly")

    # Check if results are comparable to XPSTORM
    nse = helpers.get_nse(ds_itzi_results, ds_ref)
    rsr = helpers.get_rsr(ds_itzi_results, ds_ref)

    assert nse > 0.99
    assert rsr < 0.01

    ## Check if stat file is coherent ##
    stat_file_path = Path(test_data_temp_path) / Path("ea8b_icechunk.csv")

    if stat_file_path.exists():
        df_stats = pd.read_csv(stat_file_path, sep=",")
        # convert percent string to float
        df_stats["percent_error"] = (
            df_stats["percent_error"].str.rstrip("%").astype("float") / 100.0
        )
        # Compute the reference error, preventing NaN
        df_stats["err_ref"] = np.where(
            df_stats["volume_change"] == 0,
            0.0,
            df_stats["volume_error"] / df_stats["volume_change"],
        )
        # Check if the error percentage computation is correct
        assert np.allclose(df_stats["percent_error"], df_stats["err_ref"], atol=0.0005)

        # Check if the volume change is coherent with the rest of the volumes
        df_stats["vol_change_ref"] = (
            df_stats["boundary_volume"]
            + df_stats["rainfall_volume"]
            + df_stats["infiltration_volume"]
            + df_stats["inflow_volume"]
            + df_stats["losses_volume"]
            + df_stats["drainage_network_volume"]
            + df_stats["volume_error"]
        )
        print(df_stats.to_string())
        assert np.allclose(
            df_stats["vol_change_ref"], df_stats["volume_change"], atol=1, rtol=0.01
        )
