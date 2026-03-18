from collections import namedtuple
from datetime import datetime, timedelta
import os
from pathlib import Path
import zipfile

import pytest
import numpy as np
import pandas as pd
import pyproj
import icechunk
import obstore

from itzi.providers.domain_data import DomainData
from itzi.profiler import profile_context
from itzi.simulation_builder import SimulationBuilder
from itzi.data_containers import SimulationConfig, SurfaceFlowParameters
from itzi.const import TemporalType
from itzi.providers.xarray_input import XarrayRasterInputProvider
from itzi.providers.icechunk_output import IcechunkRasterOutputProvider
from itzi.providers.csv_output import CSVVectorOutputProvider


Domain5by5Data = namedtuple(
    "Domain5by5Data",
    [
        "domain_data",  # DomainData instance
        "arr_dem_flat",  # DEM with z=0
        "arr_dem_high",  # DEM with z=132
        "arr_n",  # Manning's n = 0.05
        "arr_start_h",  # Initial depth: 0.2 at center [2,2]
        "arr_start_wse",  # Initial WSE: 132.2 at center [2,2]
        "arr_mask",  # All False (no mask)
        "arr_bctype",  # Boundary condition type for open boundaries
        "arr_rain",  # Rainfall in m/s
        "arr_inf",  # Infiltration in m/s
        "arr_loss",  # Losses in m/s
        "arr_inflow",  # Inflow in m/s
    ],
)


@pytest.fixture(scope="module")
def domain_5by5() -> Domain5by5Data:
    """Create a 5x5 domain with all base arrays.

    This fixture provides the foundational data for all 5x5 tests:
    - 5x5 grid at 10m resolution
    - Domain extends: north=50, south=0, east=50, west=0
    - Total area: 2500 m²
    """
    # Domain dimensions
    rows, cols = 5, 5
    north, south, east, west = 50.0, 0.0, 50.0, 0.0

    # Create DomainData
    domain_data = DomainData(
        north=north, south=south, east=east, west=west, rows=rows, cols=cols, crs_wkt=""
    )

    # DEM arrays
    arr_dem_flat = np.zeros(domain_data.shape, dtype=np.float32)
    arr_dem_high = np.full(domain_data.shape, 132.0, dtype=np.float32)

    # Manning's n
    arr_n = np.full(domain_data.shape, 0.05, dtype=np.float32)

    # Initial water depth: 0.2m at center cell [2, 2], 0 elsewhere
    arr_start_h = np.zeros(domain_data.shape, dtype=np.float32)
    arr_start_h[2, 2] = 0.2

    # Initial water surface elevation: 132.2m at center cell [2, 2]
    # (high DEM + 0.2m depth)
    arr_start_wse = np.zeros(domain_data.shape, dtype=np.float32)
    arr_start_wse[2, 2] = 132.2

    # No mask - whole domain active
    arr_mask = np.full(domain_data.shape, False, dtype=np.bool_)

    # Boundary condition type: 2 (open) at all 16 edge cells
    arr_bctype = np.zeros(domain_data.shape, dtype=np.float32)
    # Top and bottom rows
    arr_bctype[0, :] = 2
    arr_bctype[4, :] = 2
    # Left and right columns (excluding corners already set)
    arr_bctype[:, 0] = 2
    arr_bctype[:, 4] = 2

    # Rate arrays in m/s
    # Rainfall: 10 mm/h = 10/(1000*3600) m/s
    arr_rain = np.full(domain_data.shape, 10.0 / (1000 * 3600), dtype=np.float32)
    # Infiltration: 2 mm/h = 2/(1000*3600) m/s
    arr_inf = np.full(domain_data.shape, 2.0 / (1000 * 3600), dtype=np.float32)
    # Losses: 1.5 mm/h = 1.5/(1000*3600) m/s
    arr_loss = np.full(domain_data.shape, 1.5 / (1000 * 3600), dtype=np.float32)
    # Inflow: 0.1 m/s (already in m/s)
    arr_inflow = np.full(domain_data.shape, 0.1, dtype=np.float32)

    return Domain5by5Data(
        domain_data=domain_data,
        arr_dem_flat=arr_dem_flat,
        arr_dem_high=arr_dem_high,
        arr_n=arr_n,
        arr_start_h=arr_start_h,
        arr_start_wse=arr_start_wse,
        arr_mask=arr_mask,
        arr_bctype=arr_bctype,
        arr_rain=arr_rain,
        arr_inf=arr_inf,
        arr_loss=arr_loss,
        arr_inflow=arr_inflow,
    )


# EA Test 8B fixtures

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
        import requests

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
    import xarray as xr
    import rioxarray
    import pyproj

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
    crs = pyproj.CRS.from_epsg(32633)  # UTM Zone 33N

    # Process DEM - directly import with xarray
    dem_path = os.path.join(unzip_path, "Test8DEM.asc")
    dem_da = rioxarray.open_rasterio(dem_path).isel(band=0)
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

    # Ensure arrays are 2D
    if dem_with_buildings.ndim > 2:
        dem_with_buildings = np.squeeze(dem_with_buildings)
    if manning.ndim > 2:
        manning = np.squeeze(manning)

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

    return {
        "dataset": dataset,
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
    """Run simulation with icechunk and spatialite backends.

    This fixture runs the full EA8b simulation and saves:
    - Hotstart at split point (1h40m) for hotstart tests
    - Final raster state for comparison
    - Hotstart at end for validation tests
    """

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
    raster_input_provider = XarrayRasterInputProvider(
        {
            "dataset": ea_test8b_icechunk_data["dataset"],
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
    obj_store = obstore.store.LocalStore(prefix=Path(test_data_temp_path))
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

    # Split time: 1 hour 40 minutes (halfway through the simulation)
    split_time = sim_start_time + timedelta(hours=1, minutes=40)

    # Run the simulation
    with profile_context(profile_path):
        simulation.initialize()
        # Run to split point
        while simulation.sim_time < split_time:
            simulation.update()

        # Save hotstart at split point for hotstart test
        hotstart_split = simulation.create_hotstart()
        hotstart_split_path = Path(test_data_temp_path) / "ea8b_hotstart_split.zip"
        with open(hotstart_split_path, "wb") as f:
            f.write(hotstart_split.getvalue())

        # Continue running to end
        while simulation.sim_time < simulation.end_time:
            simulation.update()

        # save hotstart at end (for existing test)
        hotstart = simulation.create_hotstart()
        # Save hotstart to file
        hotstart_path = Path(test_data_temp_path) / "ea8b_hotstart.zip"
        with open(hotstart_path, "wb") as f:
            f.write(hotstart.getvalue())

        # finalize
        simulation.finalize()

        # Save final raster state for hotstart test comparison
        final_state_path = Path(test_data_temp_path) / "ea8b_final_state.npz"
        final_state = {}
        for key in simulation.raster_domain.k_all:
            final_state[f"raster_{key}"] = simulation.raster_domain.get_array(key)
        np.savez(final_state_path, **final_state)

    return {
        "obj_store": obj_store,
        "profile_path": profile_path,
        "sim_start_time": sim_start_time,
        "icechunk_data": ea_test8b_icechunk_data,
        "output_storage": output_storage,
        "simulation": simulation,
        "hotstart_path": hotstart_path,
        "hotstart_split_path": hotstart_split_path,
        "final_state_path": final_state_path,
        "split_time": split_time,
    }
