import os
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
import icechunk
import obstore
import xarray as xr
import rioxarray
import pyproj

from itzi.simulation_builder import SimulationBuilder
from itzi.data_containers import SimulationConfig, SurfaceFlowParameters
from itzi.const import TemporalType
from itzi.providers.xarray_input import XarrayRasterInputProvider
from itzi.providers.icechunk_output import IcechunkRasterOutputProvider
from itzi.providers.csv_output import CSVVectorOutputProvider


TEST8B_URL = "https://zenodo.org/api/records/15256842/files/Test8B_dataset_2010.zip/content"
TEST8B_MD5 = "84b865cedd28f8156cfe70b84004b62c"


@pytest.fixture(scope="session")
def ea8b_test_data(test_data_temp_path, helpers):
    file_name = "Test8B_dataset_2010.zip"
    file_path = os.path.join(test_data_temp_path, file_name)
    try:
        assert helpers.md5(file_path) == TEST8B_MD5
    except Exception:
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


@pytest.fixture(scope="package")
def ea8b_data(ea8b_test_data, test_data_temp_path):
    os.chdir(test_data_temp_path)

    with zipfile.ZipFile(ea8b_test_data, "r") as zip_ref:
        zip_ref.extractall()
    unzip_path = os.path.join(test_data_temp_path, "Test8B dataset 2010")

    west, south, east, north = 263976, 664408, 264940, 664808
    res = 2.0
    cols = int((east - west) / res)
    rows = int((north - south) / res)
    assert cols == 482
    assert rows == 200

    x_coords = np.linspace(west + res / 2, east - res / 2, cols)
    y_coords = np.linspace(north - res / 2, south + res / 2, rows)
    crs = pyproj.CRS.from_epsg(32633)

    dem_path = os.path.join(unzip_path, "Test8DEM.asc")
    dem_da = rioxarray.open_rasterio(dem_path).isel(band=0)
    dem_da = dem_da.interp(x=x_coords, y=y_coords, method="linear")
    dem_data = dem_da.values

    buildings_path = os.path.join(unzip_path, "Test8Buildings.asc")
    buildings_da = rioxarray.open_rasterio(buildings_path, mask_and_scale=True).isel(band=0)
    buildings_da = buildings_da.interp(x=x_coords, y=y_coords, method="nearest")
    buildings_data = buildings_da.values
    dem_with_buildings = np.where(np.isnan(buildings_data), dem_data, dem_data + 5.0)

    road_path = os.path.join(unzip_path, "Test8RoadPavement.asc")
    road_da = rioxarray.open_rasterio(road_path, mask_and_scale=True).isel(band=0)
    road_da = road_da.interp(x=x_coords, y=y_coords, method="nearest")
    road_data = road_da.values
    manning = np.where(np.isnan(road_data), 0.05, 0.02)

    if dem_with_buildings.ndim > 2:
        dem_with_buildings = np.squeeze(dem_with_buildings)
    if manning.ndim > 2:
        manning = np.squeeze(manning)

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


@pytest.fixture(scope="package")
def ea8b_simulation(ea8b_data, test_data_path, test_data_temp_path):
    os.chdir(test_data_temp_path)

    output_dir = Path(test_data_temp_path) / "spatialite_output"
    output_dir.mkdir(exist_ok=True)
    db_file = output_dir / "out_drainage.db"
    if db_file.exists():
        db_file.unlink()

    output_storage = icechunk.in_memory_storage()
    inp_file = os.path.join(test_data_path, "EA_test_8", "b", "test8b_drainage_ponding.inp")

    sim_start_time = datetime.min
    sim_end_time = sim_start_time + timedelta(hours=3, minutes=20)
    split_time = sim_start_time + timedelta(hours=1, minutes=40)

    arr_mask = np.zeros((ea8b_data["rows"], ea8b_data["cols"]), dtype=bool)
    surface_flow_params = SurfaceFlowParameters(cfl=0.5, theta=0.7)

    sim_config = SimulationConfig(
        start_time=sim_start_time,
        end_time=sim_end_time,
        record_step=timedelta(seconds=30),
        temporal_type=TemporalType.RELATIVE,
        input_map_names={"dem": "dem", "friction": "friction"},
        output_map_names={"water_depth": "test_water_depth"},
        drainage_output="out_drainage",
        swmm_inp=inp_file,
        stats_file="ea8b.csv",
        surface_flow_parameters=surface_flow_params,
        orifice_coeff=1.0,
    )

    raster_input_provider = XarrayRasterInputProvider(
        {
            "dataset": ea8b_data["dataset"],
            "input_map_names": sim_config.input_map_names,
            "simulation_start_time": sim_config.start_time,
            "simulation_end_time": sim_config.end_time,
        }
    )
    domain_data = raster_input_provider.get_domain_data()
    coords = domain_data.get_coordinates()
    x_coords = coords["x"]
    y_coords = coords["y"]
    crs = pyproj.CRS.from_wkt(domain_data.crs_wkt)

    raster_output_provider = IcechunkRasterOutputProvider(
        {
            "out_map_names": sim_config.output_map_names,
            "crs": crs,
            "x_coords": x_coords,
            "y_coords": y_coords,
            "icechunk_storage": output_storage,
        }
    )

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

    simulation = (
        SimulationBuilder(sim_config, arr_mask)
        .with_input_provider(raster_input_provider)
        .with_raster_output_provider(raster_output_provider)
        .with_vector_output_provider(vector_output_provider)
        .build()
    )

    simulation.initialize()
    while simulation.sim_time < split_time:
        simulation.update()

    hotstart_split = simulation.create_hotstart()
    hotstart_split_path = Path(test_data_temp_path) / "ea8b_hotstart_split.zip"
    with open(hotstart_split_path, "wb") as f:
        f.write(hotstart_split.getvalue())

    while simulation.sim_time < simulation.end_time:
        simulation.update()

    hotstart_end = simulation.create_hotstart()
    hotstart_end_path = Path(test_data_temp_path) / "ea8b_hotstart.zip"
    with open(hotstart_end_path, "wb") as f:
        f.write(hotstart_end.getvalue())

    simulation.finalize()

    final_state_path = Path(test_data_temp_path) / "ea8b_final_state.npz"
    final_state = {}
    for key in simulation.raster_domain.k_all:
        final_state[f"raster_{key}"] = simulation.raster_domain.get_array(key)
    np.savez(final_state_path, **final_state)

    return {
        "obj_store": obj_store,
        "output_storage": output_storage,
        "hotstart_split_path": hotstart_split_path,
        "hotstart_end_path": hotstart_end_path,
        "final_state_path": final_state_path,
        "split_time": split_time,
        "sim_start_time": sim_start_time,
        "data": ea8b_data,
    }


@pytest.fixture(scope="package")
def ea8b_drainage_results(ea8b_simulation):
    obj_store = ea8b_simulation["obj_store"]
    nodes_csv = StringIO(
        bytes(obstore.get(obj_store, "out_drainage_nodes.csv").bytes()).decode("utf-8")
    )
    df_results = pd.read_csv(nodes_csv)
    df_results["sim_time"] = pd.to_timedelta(df_results["sim_time"])
    df_results["start_time"] = df_results["sim_time"].dt.total_seconds().astype(int)
    df_results.set_index("start_time", inplace=True)
    df_results.drop(columns=["sim_time"], inplace=True)
    df_results = df_results[df_results.index >= 3000]
    df_results.index = pd.to_timedelta(df_results.index, unit="s")
    return df_results["coupling_flow"]


@pytest.fixture(scope="session")
def ea8b_reference(test_data_path):
    col_names = ["Time", "results"]
    file_path = os.path.join(test_data_path, "EA_test_8", "b", "xpstorm.csv")
    df_ref = pd.read_csv(file_path, index_col=0, names=col_names)
    df_ref.index *= 60.0
    df_ref.index = df_ref.index.round(decimals=2)
    df_ref.index = pd.to_timedelta(df_ref.index, unit="s")
    return df_ref.squeeze()
