"""
Integration tests for hotstart functionality with the EA test case 8b.

This file implements Phase 6 from the hotstart testing plan:
- Phase 6: Scenario Coverage Tests (Drainage Scenario)

The test verifies that:
1. Hotstart with SWMM drainage state is correctly saved and restored
2. Resumed simulation produces identical final results to uninterrupted run

This test relies on the hotstart and final state files saved by test_ea8b_icechunk.py.
Run test_ea8b first to generate the required files.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import io
import os
from pathlib import Path
import zipfile
import json
import hashlib

import numpy as np
import pytest

# Skip entire module if optional dependencies are missing
pytest.importorskip("requests")
pytest.importorskip("xarray")
pytest.importorskip("rioxarray")
pytest.importorskip("pyproj")
pytest.importorskip("icechunk")
pytest.importorskip("obstore")

import pyproj
import icechunk
import icechunk.xarray
import obstore
import pandas as pd

from itzi.simulation_builder import SimulationBuilder
from itzi.data_containers import SimulationConfig, SurfaceFlowParameters
from itzi.const import TemporalType
from itzi.providers.xarray_input import XarrayRasterInputProvider
from itzi.providers.icechunk_output import IcechunkRasterOutputProvider
from itzi.providers.csv_output import CSVVectorOutputProvider
from itzi.hotstart import HotstartLoader


# Mark all tests in this module as cloud tests
pytestmark = pytest.mark.cloud


def build_simulation(
    sim_config: SimulationConfig,
    ea_test8b_icechunk_data: dict,
    test_data_path: str,
    hotstart_bytes: bytes | None = None,
):
    """Build a simulation with optional hotstart.

    Args:
        sim_config: Simulation configuration
        ea_test8b_icechunk_data: Domain data fixture
        test_data_path: Path to test data
        hotstart_bytes: Optional hotstart archive bytes

    Returns:
        Tuple of (simulation, obj_store, output_storage)
    """
    # Create icechunk output storage
    output_storage = icechunk.in_memory_storage()

    # Create mask array (no mask for this test)
    arr_mask = np.zeros(
        (ea_test8b_icechunk_data["rows"], ea_test8b_icechunk_data["cols"]), dtype=bool
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
    builder = (
        SimulationBuilder(sim_config, arr_mask)
        .with_input_provider(raster_input_provider)
        .with_raster_output_provider(raster_output_provider)
        .with_vector_output_provider(vector_output_provider)
    )

    if hotstart_bytes is not None:
        builder.with_hotstart(hotstart_bytes)

    simulation = builder.build()

    return simulation, obj_store, output_storage


@pytest.mark.slow
@pytest.mark.forked  # Avoid pyswmm.errors.MultiSimulationError
def test_ea8b_hotstart_roundtrip(
    ea_test8b_icechunk_data,
    test_data_path,
    test_data_temp_path,
    helpers,
):
    """Test hotstart roundtrip with EA8b drainage scenario.

    This test uses the hotstart file saved at split point by test_ea8b_icechunk.py
    and verifies that resuming from hotstart produces the same final results.

    Verifies:
    - State is correctly restored from hotstart
    - Final results match between uninterrupted and resumed runs
    - Hotstart archive is valid and contains expected data

    Note: This test requires test_ea8b to be run first to generate the hotstart files.
    """
    # Define paths to the files created by test_ea8b
    hotstart_split_path = Path(test_data_temp_path) / "ea8b_hotstart_split.zip"
    final_state_path = Path(test_data_temp_path) / "ea8b_final_state.npz"

    # Skip if hotstart split file doesn't exist (test_ea8b not run yet)
    if not hotstart_split_path.exists():
        pytest.skip("Hotstart split file not found - run test_ea8b first")

    # Load the hotstart file to get split_time
    hotstart_loader = HotstartLoader.from_file(hotstart_split_path)
    split_time = datetime.fromisoformat(hotstart_loader.get_simulation_state().sim_time)

    # Total simulation duration: 3 hours 20 minutes
    sim_start_time = datetime.min
    sim_end_time = sim_start_time + timedelta(hours=3, minutes=20)

    # Create simulation configuration (same as original test)
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
        swmm_inp=os.path.join(test_data_path, "EA_test_8", "b", "test8b_drainage_ponding.inp"),
        stats_file="ea8b_hotstart.csv",
        surface_flow_parameters=surface_flow_params,
        orifice_coeff=1.0,
    )

    # Change to temp directory for any file outputs
    os.chdir(test_data_temp_path)

    # Build simulation with hotstart
    with open(hotstart_split_path, "rb") as f:
        hotstart_bytes = f.read()

    simulation, obj_store, output_storage = build_simulation(
        sim_config,
        ea_test8b_icechunk_data,
        test_data_path,
        hotstart_bytes=hotstart_bytes,
    )

    # =========================================================================
    # Verify state restoration
    # =========================================================================

    # Verify simulation time was restored
    assert simulation.sim_time == split_time, (
        f"sim_time not restored: {simulation.sim_time} != {split_time}"
    )

    # Verify raster state was restored from hotstart
    raster_state_bytes = hotstart_loader.raster_state_bytes
    raster_state_buffer = np.load(io.BytesIO(raster_state_bytes), allow_pickle=False)

    for key in simulation.raster_domain.k_all:
        arr_restored = simulation.raster_domain.get_padded(key)
        arr_saved = raster_state_buffer[key]
        np.testing.assert_allclose(
            arr_restored,
            arr_saved,
            err_msg=f"Raster state {key} not restored correctly",
        )

    # =========================================================================
    # Run resumed simulation to end
    # =========================================================================

    # Skip initialize for hotstarted simulation
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()

    # =========================================================================
    # Diagnostic: inspect drainage node coupling flow in resumed simulation
    # =========================================================================

    nodes_csv_bytes = bytes(obstore.get(obj_store, "out_drainage_nodes.csv").bytes())
    df_resumed = pd.read_csv(io.StringIO(nodes_csv_bytes.decode("utf-8")))

    diagnostic_path = Path(test_data_temp_path) / "ea8b_hotstart_diagnostic.txt"
    with open(diagnostic_path, "w") as f:
        f.write("=== Resumed simulation: first 10 drainage node rows ===\n")
        f.write(df_resumed[["sim_time", "node_id", "coupling_flow", "depth"]].head(10).to_string())
        f.write("\n\n=== Resumed simulation: last 10 drainage node rows ===\n")
        f.write(df_resumed[["sim_time", "node_id", "coupling_flow", "depth"]].tail(10).to_string())
        f.write(f"\n\nTotal rows: {len(df_resumed)}, coupling_flow stats:\n")
        f.write(df_resumed["coupling_flow"].describe().to_string())

        # Load reference drainage CSV written by the uninterrupted simulation (LocalStore)
        # The LocalStore prefix is test_data_temp_path, and drainage_results_name = "out_drainage"
        ref_nodes_path = Path(test_data_temp_path) / "out_drainage_nodes.csv"
        if ref_nodes_path.exists():
            df_ref = pd.read_csv(ref_nodes_path)

            # Convert sim_time to timedelta for proper comparison
            # CSV uses ISO 8601 duration format (PT6000.0S)
            df_ref["sim_time_td"] = pd.to_timedelta(df_ref["sim_time"])
            df_resumed["sim_time_td"] = pd.to_timedelta(df_resumed["sim_time"])

            # Split time as timedelta
            split_time_td = pd.Timedelta(hours=1, minutes=40)

            # Filter to second half (>= split_time)
            df_ref_second_half = df_ref[df_ref["sim_time_td"] >= split_time_td]
            df_resumed_second_half = df_resumed[df_resumed["sim_time_td"] >= split_time_td]

            f.write("\n\n=== Uninterrupted simulation (second half): first 10 rows ===\n")
            f.write(
                df_ref_second_half[["sim_time", "node_id", "coupling_flow", "depth"]]
                .head(10)
                .to_string()
            )
            f.write("\n\nReference coupling_flow stats (second half):\n")
            f.write(df_ref_second_half["coupling_flow"].describe().to_string())

            # Compare coupling_flow between resumed and reference for second half
            f.write("\n\n=== Coupling flow comparison (resumed vs reference) ===\n")
            f.write(f"Resumed second half rows: {len(df_resumed_second_half)}\n")
            f.write(f"Reference second half rows: {len(df_ref_second_half)}\n")

            # Detailed comparison for J1 (the diverging node)
            j1_resumed = df_resumed_second_half[
                df_resumed_second_half["node_id"] == "J1"
            ].sort_values("sim_time_td")
            j1_ref = df_ref_second_half[df_ref_second_half["node_id"] == "J1"].sort_values(
                "sim_time_td"
            )
            if len(j1_resumed) > 0 and len(j1_ref) > 0:
                f.write("\n\n=== J1 detailed comparison (first 5 timesteps) ===\n")
                f.write("Resumed J1:\n")
                f.write(
                    j1_resumed[["sim_time", "coupling_flow", "depth", "head", "inflow"]]
                    .head(5)
                    .to_string()
                )
                f.write("\n\nReference J1:\n")
                f.write(
                    j1_ref[["sim_time", "coupling_flow", "depth", "head", "inflow"]]
                    .head(5)
                    .to_string()
                )
                # Also show depth and head differences
                if "depth" in j1_resumed.columns and "depth" in j1_ref.columns:
                    f.write("\n\n=== J1 depth/head comparison ===\n")
                    for i in range(min(5, len(j1_resumed))):
                        r = j1_resumed.iloc[i]
                        ref = j1_ref.iloc[i]
                        f.write(
                            f"  {r['sim_time']}: depth resumed={r['depth']:.6f} ref={ref['depth']:.6f} "
                            f"diff={r['depth'] - ref['depth']:.6f}\n"
                        )

            # Group by node_id and compare
            for node_id in df_resumed_second_half["node_id"].unique():
                resumed_node = df_resumed_second_half[
                    df_resumed_second_half["node_id"] == node_id
                ].sort_values("sim_time_td")
                ref_node = df_ref_second_half[
                    df_ref_second_half["node_id"] == node_id
                ].sort_values("sim_time_td")
                if len(resumed_node) > 0 and len(ref_node) > 0:
                    # Compare coupling_flow values
                    resumed_cf = resumed_node["coupling_flow"].values
                    ref_cf = ref_node["coupling_flow"].values
                    if len(resumed_cf) == len(ref_cf):
                        max_diff = np.max(np.abs(resumed_cf - ref_cf))
                        mean_diff = np.mean(np.abs(resumed_cf - ref_cf))
                        f.write(
                            f"Node {node_id}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}\n"
                        )
                    else:
                        f.write(
                            f"Node {node_id}: length mismatch (resumed={len(resumed_cf)}, ref={len(ref_cf)})\n"
                        )
        else:
            f.write(f"\n\nReference CSV not found at {ref_nodes_path}\n")

    # =========================================================================
    # Compare final results with uninterrupted simulation
    # =========================================================================

    # Load final state from uninterrupted simulation
    final_state = np.load(final_state_path, allow_pickle=False)

    # Compare key output arrays (water_depth, qe, qs)
    # Note: We use qe/qs (internal flow arrays) instead of qx/qy (output arrays
    # computed on-the-fly) because qx/qy are not stored in k_all.
    key_output_arrays = ["water_depth", "qe", "qs"]
    for key in key_output_arrays:
        arr_resumed = simulation.raster_domain.get_array(key)
        arr_uninterrupted = final_state[f"raster_{key}"]
        np.testing.assert_allclose(
            arr_resumed,
            arr_uninterrupted,
            err_msg=f"Final {key} mismatch between uninterrupted and resumed simulations",
        )


@pytest.mark.slow
@pytest.mark.forked
def test_ea8b_hotstart_archive_validity(test_data_temp_path):
    """Test that the created hotstart archive is valid and contains expected data.

    Verifies:
    - Archive is a valid ZIP file
    - Contains required metadata.json and raster_state.npz
    - Contains optional swmm_hotstart.hsf (drainage scenario)
    - Hashes in metadata match actual file contents
    """
    hotstart_split_path = Path(test_data_temp_path) / "ea8b_hotstart_split.zip"

    # Skip if hotstart split file doesn't exist
    if not hotstart_split_path.exists():
        pytest.skip("Hotstart split file not found")

    with zipfile.ZipFile(hotstart_split_path, "r") as zip_ref:
        members = zip_ref.namelist()

        # Check required members
        assert "metadata.json" in members, "Missing metadata.json"
        assert "raster_state.npz" in members, "Missing raster_state.npz"
        assert "swmm_hotstart.hsf" in members, "Missing swmm_hotstart.hsf (drainage scenario)"

        # Load and validate metadata
        with zip_ref.open("metadata.json") as metadata_file:
            metadata_dict = json.load(metadata_file)

        # Check hotstart version
        assert metadata_dict["hotstart_version"] == 1, "Wrong hotstart version"

        # Check that simulation state is present
        assert "simulation_state" in metadata_dict, "Missing simulation_state in metadata"
        sim_state = metadata_dict["simulation_state"]

        # Verify raster hash
        ref_raster_hash = sim_state["raster_domain_hash"]
        hash_raster = hashlib.blake2b(zip_ref.read("raster_state.npz")).hexdigest()
        assert hash_raster == ref_raster_hash, "Raster state hash mismatch"

        # Verify SWMM hash
        ref_swmm_hash = sim_state["swmm_hotstart_hash"]
        hash_swmm = hashlib.blake2b(zip_ref.read("swmm_hotstart.hsf")).hexdigest()
        assert hash_swmm == ref_swmm_hash, "SWMM hotstart hash mismatch"

        # Verify domain data is present
        assert "domain_data" in metadata_dict, "Missing domain_data in metadata"
        domain_data = metadata_dict["domain_data"]
        assert domain_data["rows"] == 200, "Wrong rows in domain_data"
        assert domain_data["cols"] == 482, "Wrong cols in domain_data"
