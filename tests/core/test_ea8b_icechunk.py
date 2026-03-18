"""
Integration tests using the EA test case 8b with icechunk and spatialite backends.
The results from itzi are compared with those from XPSTORM.
"""

from io import StringIO
from pathlib import Path
import zipfile
import json
import hashlib

import pandas as pd
import numpy as np
import pytest

# Skip entire module if optional dependencies are missing
pytest.importorskip("requests")
pytest.importorskip("xarray")
pytest.importorskip("rioxarray")
pytest.importorskip("pyproj")
pytest.importorskip("icechunk")
pytest.importorskip("obstore")

import xarray as xr
import icechunk
import icechunk.xarray
import obstore


# Mark all tests in this module as cloud tests
pytestmark = pytest.mark.cloud


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
# @pytest.mark.forked  # Avoid pyswmm.errors.MultiSimulationError: Multi-Simulation Error.
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
        assert np.allclose(
            df_stats["vol_change_ref"], df_stats["volume_change"], atol=1, rtol=0.01
        )

    # Check hotstart file
    hotstart_file = ea_test8b_sim_icechunk["hotstart_path"]
    with zipfile.ZipFile(hotstart_file, "r") as zip_ref:
        with zip_ref.open("metadata.json") as metadata_file:
            metadata_dict = json.load(metadata_file)
            # Check hashes
            ref_raster_hash = metadata_dict["simulation_state"]["raster_domain_hash"]
            hash_raster = hashlib.blake2b(zip_ref.read("raster_state.npz")).hexdigest()
            assert hash_raster == ref_raster_hash
            ref_swmm_hash = metadata_dict["simulation_state"]["swmm_hotstart_hash"]
            hash_swmm = hashlib.blake2b(zip_ref.read("swmm_hotstart.hsf")).hexdigest()
            assert hash_swmm == ref_swmm_hash
