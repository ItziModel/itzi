"""
Integration tests using the EA test case 8a with xarray and memory backends.
The results from itzi are compared with those from LISFLOOD-FP.
"""

from datetime import datetime, timedelta
import os
from pathlib import Path
import zipfile

import pandas as pd
import numpy as np
import requests
import pytest
import xarray as xr
import rioxarray

from itzi.profiler import profile_context
from itzi.simulation_builder import SimulationBuilder
from itzi.data_containers import SimulationConfig, SurfaceFlowParameters
from itzi.const import TemporalType, VerbosityLevel
from itzi.providers.xarray_input import XarrayRasterInputProvider
from itzi.providers.memory_output import MemoryRasterOutputProvider, MemoryVectorOutputProvider


TEST8A_URL = "https://zenodo.org/api/records/15256842/files/Test8A_dataset_2010.zip/content"
TEST8A_MD5 = "46b589ee000ff87c9077fcc51fa71e8e"

os.environ["ITZI_VERBOSE"] = str(VerbosityLevel.SUPER_QUIET)


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


@pytest.fixture(scope="module")
def ea_test8a_xarray_data(test8a_file, test_data_path, test_data_temp_path):
    """Create xarray dataset for EA test 8a."""
    # Keep all generated files in the test_data_temp_path
    os.chdir(test_data_temp_path)

    # Unzip the file
    with zipfile.ZipFile(test8a_file, "r") as zip_ref:
        zip_ref.extractall()
    unzip_path = os.path.join(test_data_temp_path, "Test8A dataset 2010")

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

    # Process DEM - import and aggregate from 50cm to 2m (matching GRASS r.resamp.stats)
    dem_path = os.path.join(unzip_path, "Test8DEM.asc")
    dem_da = rioxarray.open_rasterio(dem_path, masked=True).isel(band=0)

    # Aggregate using coarsen (4x4 blocks since 50cm to 2m is 4x factor)
    # This matches GRASS r.resamp.stats default (mean aggregation)
    dem_da_coarse = dem_da.coarsen(x=4, y=4, boundary="pad").mean()

    # Interpolate to exact target coordinates
    dem_da_resampled = dem_da_coarse.interp(x=x_coords, y=y_coords, method="nearest")
    dem_data = dem_da_resampled.values

    # Fill any remaining NaN values with nearest valid value
    if np.any(np.isnan(dem_data)):
        mask = np.isnan(dem_data)
        from scipy.ndimage import distance_transform_edt

        indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
        dem_data = dem_data[tuple(indices)]

    # Process road pavement for Manning coefficient
    road_path = os.path.join(unzip_path, "Test8RoadPavement.asc")
    road_da = rioxarray.open_rasterio(road_path, mask_and_scale=True).isel(band=0)
    road_da = road_da.interp(x=x_coords, y=y_coords, method="nearest")
    road_data = road_da.values
    # Create Manning coefficient: 0.02 where road exists, 0.05 elsewhere
    n_data = np.where(np.isnan(road_data), 0.05, 0.02)

    # Verify no null cells in DEM and Manning
    assert not np.any(np.isnan(dem_data))
    assert np.min(n_data) == 0.02
    assert np.max(n_data) == 0.05

    # Process point inflow location first
    point_path = os.path.join(unzip_path, "Test8A-inflow-location.csv")
    df_point = pd.read_csv(point_path, skiprows=1, names=["x", "y"])
    inflow_x = df_point["x"].values[0]
    inflow_y = df_point["y"].values[0]

    # Find the cell indices for the inflow point
    inflow_col = np.argmin(np.abs(x_coords - inflow_x))
    inflow_row = np.argmin(np.abs(y_coords - inflow_y))

    # Read the inflow time series
    inflow_path = os.path.join(test_data_path, "EA_test_8", "a", "point-inflow.csv")
    df_flow = pd.read_csv(inflow_path)

    # Create time series for inflow with linear interpolation
    inflow_times_list = []
    inflow_values_list = []
    for _, row in df_flow.iterrows():
        inflow_times_list.append(row["start"])
        inflow_values_list.append(row["flux"])
        inflow_times_list.append(row["end"])
        inflow_values_list.append(row["flux"])

    df_inflow = (
        pd.DataFrame({"time": inflow_times_list, "flux": inflow_values_list})
        .drop_duplicates(subset="time")
        .sort_values("time")
    )

    df_inflow["time"] = pd.to_timedelta(df_inflow["time"], unit="s")
    df_inflow = df_inflow.set_index("time")

    time_range = pd.timedelta_range(start="0s", end=f"{int(df_flow['end'].max())}s", freq="1s")
    df_inflow_interp = df_inflow.reindex(time_range).interpolate(method="linear").fillna(0)

    # Create UNIFIED time coordinate using Python timedelta objects
    # This matches the xarray provider unit tests and ensures proper precision
    max_time_seconds = int(df_flow["end"].max())
    unified_times_set = set()

    # Add rainfall critical times
    for t in [0, 60, 240]:
        unified_times_set.add(timedelta(seconds=t))

    # Add inflow times (every 10 seconds)
    for t in range(0, max_time_seconds + 1, 10):
        unified_times_set.add(timedelta(seconds=t))

    # Sort to get a list of Python timedelta objects
    time_coords = sorted(list(unified_times_set))

    # Create DataFrames for easier manipulation
    df_unified = pd.DataFrame(index=[pd.Timedelta(t) for t in time_coords])
    df_unified["inflow"] = 0.0
    df_unified["rainfall"] = 0.0

    # Fill inflow values
    for td in time_coords:
        pd_td = pd.Timedelta(td)
        if pd_td in df_inflow_interp.index:
            df_unified.loc[pd_td, "inflow"] = df_inflow_interp.loc[pd_td, "flux"]
        else:
            # Use ffill logic
            mask = df_inflow_interp.index <= pd_td
            if mask.any():
                df_unified.loc[pd_td, "inflow"] = df_inflow_interp.loc[mask].iloc[-1]["flux"]

    # Fill rainfall values (step function)
    for td in time_coords:
        td_seconds = td.total_seconds()
        pd_td = pd.Timedelta(td)
        if td_seconds < 60:
            df_unified.loc[pd_td, "rainfall"] = 0
        elif td_seconds < 240:
            df_unified.loc[pd_td, "rainfall"] = 400
        else:
            df_unified.loc[pd_td, "rainfall"] = 0

    # Create rainfall arrays
    rain_data_list = []
    for rain_val in df_unified["rainfall"].values:
        rain_arr = np.full((rows, cols), rain_val, dtype=np.float32)
        rain_data_list.append(rain_arr)
    rain_data = np.stack(rain_data_list, axis=0)

    # Create inflow arrays
    inflow_data_list = []
    for flux_val in df_unified["inflow"].values:
        inflow_arr = np.zeros((rows, cols), dtype=np.float32)
        inflow_arr[inflow_row, inflow_col] = flux_val
        inflow_data_list.append(inflow_arr)
    inflow_data = np.stack(inflow_data_list, axis=0)

    # Create output points for comparison
    stages_path = os.path.join(unzip_path, "Test8Output.csv")
    df_output = pd.read_csv(stages_path, skiprows=1, names=["cat", "x", "y"])
    output_points = df_output[["x", "y"]].values

    # Create xarray dataset with unified time coordinate
    rainfall_da = xr.DataArray(
        rain_data,
        coords={"time": time_coords, "y": y_coords, "x": x_coords},
        dims=["time", "y", "x"],
    )

    inflow_da = xr.DataArray(
        inflow_data,
        coords={"time": time_coords, "y": y_coords, "x": x_coords},
        dims=["time", "y", "x"],
    )

    ds = xr.Dataset(
        {
            "dem": (["y", "x"], dem_data.astype(np.float32)),
            "friction": (["y", "x"], n_data.astype(np.float32)),
            "rainfall": rainfall_da,
            "inflow": inflow_da,
        },
        coords={
            "x": x_coords,
            "y": y_coords,
        },
    )
    return ds, output_points


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


@pytest.fixture(scope="module")
def ea_test8a_sim(ea_test8a_xarray_data, test_data_path, test_data_temp_path):
    """Run EA test 8a simulation using xarray input and memory output."""
    ds, output_points = ea_test8a_xarray_data

    # Keep all generated files in the test_data_temp_path
    os.chdir(test_data_temp_path)

    # Simulation parameters
    sim_start_time = datetime(2000, 1, 1, 0, 0, 0)
    sim_duration = timedelta(seconds=4980)  # 1 hour 23 minutes
    sim_end_time = sim_start_time + sim_duration

    # Create input provider
    input_config = {
        "dataset": ds,
        "input_map_names": {
            "dem": "dem",
            "friction": "friction",
            "rain": "rainfall",
            "inflow": "inflow",
        },
        "simulation_start_time": sim_start_time,
        "simulation_end_time": sim_end_time,
    }
    input_provider = XarrayRasterInputProvider(input_config)

    # Create simulation configuration
    sim_config = SimulationConfig(
        start_time=sim_start_time,
        end_time=sim_end_time,
        record_step=timedelta(seconds=30),
        temporal_type=TemporalType.RELATIVE,
        input_map_names={
            "dem": "dem",
            "friction": "friction",
            "rain": "rainfall",
            "inflow": "inflow",
        },
        output_map_names={
            "water_depth": "out_water_depth",
            "water_surface_elevation": "out_wse",
            "qx": "out_qx",
            "qy": "out_qy",
        },
        surface_flow_parameters=SurfaceFlowParameters(
            dtmax=5.0,
            cfl=0.5,
            theta=0.7,
            vrouting=0.1,
            hmin=0.005,
        ),
        stats_file="stats_ea8a.csv",
    )

    # Create output providers
    raster_output = MemoryRasterOutputProvider({"out_map_names": sim_config.output_map_names})
    vector_output = MemoryVectorOutputProvider({})

    # Create simulation domain data from input provider
    domain_data = input_provider.get_domain_data()

    # Create mask (no mask, use whole domain)
    array_mask = np.full(
        shape=(domain_data.rows, domain_data.cols), fill_value=False, dtype=np.bool_
    )

    # Build and run simulation
    # Note: with_input_provider() automatically sets domain_data
    profile_path = Path(test_data_temp_path) / Path("test8a_profile.txt")
    with profile_context(profile_path):
        simulation, timed_arrays = (
            SimulationBuilder(sim_config, array_mask, np.float32)
            .with_input_provider(input_provider)
            .with_raster_output_provider(raster_output)
            .with_vector_output_provider(vector_output)
            .build()
        )

        # Helper function to update input arrays (similar to SimulationRunner.update_input_arrays)
        def update_input_arrays():
            """Get new arrays using TimedArray and update simulation"""
            for arr_key, ta in timed_arrays.items():
                if not ta.is_valid(simulation.sim_time):
                    # Convert mm/h to m/s for rainfall
                    if arr_key in ["rain"]:
                        new_arr = ta.get(simulation.sim_time) / (1000 * 3600)
                    else:
                        new_arr = ta.get(simulation.sim_time)
                    # update array
                    simulation.set_array(arr_key, new_arr)

        # Run the simulation
        update_input_arrays()  # Update arrays before initialize
        simulation.initialize()
        while simulation.sim_time < simulation.end_time:
            simulation.update()
            update_input_arrays()  # Update arrays after each step
        simulation.finalize()

    return simulation, output_points, domain_data


@pytest.mark.slow
def test_ea8a(ea_test8a_sim, ea_test8a_reference):
    """Compare results with LISFLOOD-FP"""
    simulation, output_points, domain_data = ea_test8a_sim

    # Get water depth results from memory output
    raster_results = simulation.report.raster_provider.output_maps_dict
    depth_results = raster_results["water_depth"]

    # Extract results at output points
    itzi_data = []
    for idx, (sim_time, depth_array) in enumerate(depth_results):
        # Convert simulation time to minutes
        time_minutes = (
            sim_time.total_seconds() / 60.0
            if isinstance(sim_time, timedelta)
            else (sim_time - simulation.start_time).total_seconds() / 60.0
        )

        # Extract values at each output point
        point_values = []
        for x, y in output_points:
            # Find nearest cell
            row, col = domain_data.coordinates_to_pixel(x=x, y=y)

            # Ensure indices are within bounds
            col = max(0, min(col, domain_data.cols - 1))
            row = max(0, min(row, domain_data.rows - 1))

            point_values.append(depth_array[row, col])

        itzi_data.append([time_minutes] + point_values)

    # Create dataframe
    col_names = ["Time (min)"] + list(range(1, len(output_points) + 1))
    df_itzi = pd.DataFrame(itzi_data, columns=col_names)
    df_itzi.set_index("Time (min)", inplace=True)
    df_itzi.index = np.round(df_itzi.index, 1)

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
