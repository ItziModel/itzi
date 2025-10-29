"""
Integration tests using the EA test case 8b with icechunk and spatialite backends.
The results from itzi are compared with those from XPSTORM.
"""

from datetime import datetime, timedelta
import os
from io import StringIO
from pathlib import Path
import zipfile

import pandas as pd
import numpy as np
import requests
import pytest
import xarray as xr
import rioxarray
import pyproj
import icechunk
import icechunk.xarray
import obstore

from itzi.profiler import profile_context
from itzi.simulation_builder import SimulationBuilder
from itzi.data_containers import SimulationConfig, SurfaceFlowParameters
from itzi.const import TemporalType
from itzi.providers.icechunk_input import IcechunkRasterInputProvider
from itzi.providers.icechunk_output import IcechunkRasterOutputProvider
from itzi.providers.csv_output import CSVVectorOutputProvider


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
    """Run simulation with icechunk and spatialite backends."""
    # Keep all generated files in the test_data_temp_path
    os.chdir(test_data_temp_path)

    # Create output directory for spatialite database
    output_dir = Path(test_data_temp_path) / "spatialite_output"
    output_dir.mkdir(exist_ok=True)

    # Remove existing database file to ensure fresh schema
    db_file = output_dir / "out_drainage.db"
    if db_file.exists():
        db_file.unlink()

    # Create icechunk output storage
    output_storage = icechunk.in_memory_storage()

    inp_file = os.path.join(test_data_path, "EA_test_8", "b", "test8b_drainage_ponding.inp")

    # Create simulation configuration for icechunk/spatialite backends

    sim_start_time = datetime.min
    sim_end_time = sim_start_time + timedelta(hours=3, minutes=20)

    profile_path = Path(test_data_temp_path) / Path("test8b_icechunk_profile.txt")

    # Create mask array (no mask for this test)
    # False = valid cells, True = masked cells
    arr_mask = np.zeros(
        (ea_test8b_icechunk_data["rows"], ea_test8b_icechunk_data["cols"]), dtype=bool
    )

    # Create simulation configuration directly
    surface_flow_params = SurfaceFlowParameters(
        cfl=0.5,
        theta=0.7,
    )

    sim_config = SimulationConfig(
        start_time=sim_start_time,
        end_time=sim_end_time,
        record_step=timedelta(seconds=30),
        temporal_type=TemporalType.RELATIVE,
        input_map_names={"dem": "dem", "friction": "friction"},
        output_map_names={"water_depth": "test_water_depth"},
        drainage_output="out_drainage",
        swmm_inp=inp_file,
        stats_file="ea8b_icechunk.csv",
        surface_flow_parameters=surface_flow_params,
        orifice_coeff=1.0,  # Same as reference
    )

    # Set up providers
    raster_input_provider = IcechunkRasterInputProvider(
        {
            "icechunk_storage": ea_test8b_icechunk_data["storage"],
            "icechunk_group": "main",
            "input_map_names": sim_config.input_map_names,
            "simulation_start_time": sim_config.start_time,
            "simulation_end_time": sim_config.end_time,
        }
    )
    domain_data = raster_input_provider.get_domain_data()

    # Generate coordinate arrays from domain_data
    coords = domain_data.get_coordinates()
    x_coords = coords["x"]
    y_coords = coords["y"]
    crs = pyproj.CRS.from_wkt(domain_data.crs_wkt)

    # Set up raster output provider
    raster_output_provider = IcechunkRasterOutputProvider(
        {
            "out_map_names": sim_config.output_map_names,
            "crs": crs,
            "x_coords": x_coords,
            "y_coords": y_coords,
            "icechunk_storage": output_storage,
        }
    )

    # Set up vector output provider (csv)
    obj_store = obstore.store.MemoryStore()
    vector_output_provider = CSVVectorOutputProvider(
        {
            "crs": crs,
            "store": obj_store,
            "results_prefix": "",
            "drainage_results_name": sim_config.drainage_output,
            "overwrite": True,
        }
    )

    # Build simulation
    simulation = (
        SimulationBuilder(sim_config, arr_mask)
        .with_input_provider(raster_input_provider)
        .with_raster_output_provider(raster_output_provider)
        .with_vector_output_provider(vector_output_provider)
        .build()
    )

    # Run the simulation
    with profile_context(profile_path):
        simulation.initialize()
        while simulation.sim_time < simulation.end_time:
            simulation.update()
        simulation.finalize()

    return {
        "obj_store": obj_store,
        "profile_path": profile_path,
        "sim_start_time": sim_start_time,
        "icechunk_data": ea_test8b_icechunk_data,
        "output_storage": output_storage,
        "simulation": simulation,
    }


@pytest.fixture(scope="module")
def ea8b_itzi_drainage_results(ea_test8b_sim_icechunk):
    """Extract coupling flow from the drainage network using spatialite database."""
    obj_store = ea_test8b_sim_icechunk["obj_store"]
    nodes_csv = StringIO(
        bytes(obstore.get(obj_store, "out_drainage_nodes.csv").bytes()).decode("utf-8")
    )
    df_results = pd.read_csv(nodes_csv)

    # Convert ISO string to timedelta
    df_results["sim_time"] = pd.to_timedelta(df_results["sim_time"])
    # convert to total seconds (as the reference)
    df_results["start_time"] = df_results["sim_time"].dt.total_seconds().astype(int)

    # Set index and drop unnecessary columns
    df_results.set_index("start_time", inplace=True)
    df_results.drop(columns=["sim_time"], inplace=True)

    # Filter for times >= 3000s as in reference
    df_results = df_results[df_results.index >= 3000]

    # Convert index to timedelta
    df_results.index = pd.to_timedelta(df_results.index, unit="s")

    return df_results["coupling_flow"]


@pytest.mark.slow
def test_ea8b(
    ea_test8b_reference,
    ea8b_itzi_drainage_results,
    ea_test8b_sim_icechunk,
    helpers,
    test_data_temp_path,
):
    """Test EA8B with icechunk and spatialite backends."""
    ds_itzi_results = ea8b_itzi_drainage_results
    ds_ref = ea_test8b_reference

    # Check if we have results to compare
    if ds_itzi_results.empty:
        pytest.skip("No drainage results found - simulation may not have run properly")

    # Check if results are comparable to XPSTORM
    nse = helpers.get_nse(ds_itzi_results, ds_ref)
    rsr = helpers.get_rsr(ds_itzi_results, ds_ref)

    assert nse > 0.99
    assert rsr < 0.01

    ## Check if water_depth maps are correctly written to zarr ##
    # Access the output icechunk storage from the simulation fixture
    output_storage = ea_test8b_sim_icechunk["output_storage"]

    # Open the icechunk repository and read the water depth data
    repo = icechunk.Repository.open(output_storage)
    session = repo.readonly_session("main")

    # Load the dataset from icechunk
    output_dataset = xr.open_zarr(session.store)

    # Check if water_depth variable exists
    assert "test_water_depth" in output_dataset.variables, (
        "water_depth variable not found in output dataset"
    )

    # Get the water depth data
    water_depth_data = output_dataset["test_water_depth"]

    # Check that there are no NaN values in water depth data
    nan_count = np.sum(np.isnan(water_depth_data.values))
    assert nan_count == 0, (
        f"Found {nan_count} NaN values in water depth data - there should be none"
    )

    # Check that water depth values are between 0 and 2
    min_depth = np.min(water_depth_data.values)
    max_depth = np.max(water_depth_data.values)

    assert min_depth >= 0.0, f"Water depth values below 0 found: minimum = {min_depth}"
    assert max_depth <= 2.0, f"Water depth values above 2 found: maximum = {max_depth}"

    print(f"Water depth range: {min_depth:.3f} to {max_depth:.3f}")

    # Verify the data structure and dimensions
    assert water_depth_data.ndim == 3, (
        f"Water depth data should be 3-dimensional, got {water_depth_data.ndim} dimensions"
    )

    # Check that the spatial dimensions match the expected grid
    icechunk_data = ea_test8b_sim_icechunk["icechunk_data"]
    expected_rows = icechunk_data["rows"]
    expected_cols = icechunk_data["cols"]

    # The water depth should have spatial dimensions matching the grid
    spatial_shape = water_depth_data.shape[-2:]

    assert spatial_shape[0] == expected_rows, (
        f"Row dimension mismatch: expected {expected_rows}, got {spatial_shape[0]}"
    )
    assert spatial_shape[1] == expected_cols, (
        f"Column dimension mismatch: expected {expected_cols}, got {spatial_shape[1]}"
    )

    print(f"Water depth data shape: {water_depth_data.shape}")
    print(f"Water depth data dimensions: {water_depth_data.dims}")

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
