"""
Integration tests for the EA test case 8b.
The results from itzi are compared with those from XPSTORM.
"""

import json
import zipfile
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import icechunk

from tests.core.ea8b.helpers import EA8B_REFERENCE_MAX_RSR, EA8B_REFERENCE_MIN_NSE


pytestmark = pytest.mark.cloud


@pytest.mark.slow
def test_ea8b_scenario(
    ea8b_reference,
    ea8b_drainage_results,
    ea8b_simulation,
    helpers,
    test_data_temp_path,
):
    ds_itzi_results = ea8b_drainage_results
    ds_ref = ea8b_reference

    if ds_itzi_results.empty:
        pytest.skip("No drainage results found - simulation may not have run properly")

    nse = helpers.get_nse(ds_itzi_results, ds_ref)
    rsr = helpers.get_rsr(ds_itzi_results, ds_ref)

    assert nse > EA8B_REFERENCE_MIN_NSE
    assert rsr < EA8B_REFERENCE_MAX_RSR

    output_storage = ea8b_simulation["output_storage"]
    repo = icechunk.Repository.open(output_storage)
    session = repo.readonly_session("main")
    output_dataset = xr.open_zarr(session.store)

    assert "test_water_depth" in output_dataset.variables

    water_depth_data = output_dataset["test_water_depth"]

    nan_count = np.sum(np.isnan(water_depth_data.values))
    assert nan_count == 0, (
        f"Found {nan_count} NaN values in water depth data - there should be none"
    )

    min_depth = np.min(water_depth_data.values)
    max_depth = np.max(water_depth_data.values)

    assert min_depth >= 0.0, f"Water depth values below 0 found: minimum = {min_depth}"
    assert max_depth <= 2.0, f"Water depth values above 2 found: maximum = {max_depth}"

    print(f"Water depth range: {min_depth:.3f} to {max_depth:.3f}")

    assert water_depth_data.ndim == 3

    expected_rows = ea8b_simulation["data"]["rows"]
    expected_cols = ea8b_simulation["data"]["cols"]
    spatial_shape = water_depth_data.shape[-2:]

    assert spatial_shape[0] == expected_rows
    assert spatial_shape[1] == expected_cols

    print(f"Water depth data shape: {water_depth_data.shape}")
    print(f"Water depth data dimensions: {water_depth_data.dims}")

    stat_file_path = Path(test_data_temp_path) / "ea8b.csv"
    if stat_file_path.exists():
        df_stats = pd.read_csv(stat_file_path, sep=",")
        df_stats["percent_error"] = (
            df_stats["percent_error"].str.rstrip("%").astype("float") / 100.0
        )
        df_stats["err_ref"] = np.where(
            df_stats["volume_change"] == 0,
            0.0,
            df_stats["volume_error"] / df_stats["volume_change"],
        )
        assert np.allclose(df_stats["percent_error"], df_stats["err_ref"], atol=0.0005)

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

    hotstart_end_path = ea8b_simulation["hotstart_end_path"]
    with zipfile.ZipFile(hotstart_end_path, "r") as zip_ref:
        with zip_ref.open("metadata.json") as metadata_file:
            metadata_dict = json.load(metadata_file)
            ref_raster_hash = metadata_dict["simulation_state"]["raster_domain_hash"]
            hash_raster = hashlib.blake2b(zip_ref.read("raster_state.npz")).hexdigest()
            assert hash_raster == ref_raster_hash
            ref_swmm_hash = metadata_dict["simulation_state"]["swmm_hotstart_hash"]
            hash_swmm = hashlib.blake2b(zip_ref.read("swmm_hotstart.hsf")).hexdigest()
            assert hash_swmm == ref_swmm_hash
