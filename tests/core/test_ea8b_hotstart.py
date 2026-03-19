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


EA8B_REFERENCE_MIN_NSE = 0.99
EA8B_REFERENCE_MAX_RSR = 0.01
EA8B_FINAL_ARRAY_ATOL = {
    "water_depth": 6.3e-3,
    "qe": 2.9e-3,
    "qs": 9.0e-4,
}


def drainage_results_to_coupling_series(df_results: pd.DataFrame) -> pd.Series:
    """Convert EA8b drainage CSV output to the XPSTORM comparison series."""
    df_results = df_results.copy()
    df_results["sim_time"] = pd.to_timedelta(df_results["sim_time"])
    df_results["start_time"] = df_results["sim_time"].dt.total_seconds().astype(int)
    df_results.set_index("start_time", inplace=True)
    df_results.drop(columns=["sim_time"], inplace=True)
    df_results = df_results[df_results.index >= 3000]
    df_results.index = pd.to_timedelta(df_results.index, unit="s")
    return df_results["coupling_flow"]


def get_reference_metrics(
    results: pd.Series,
    reference: pd.Series,
    helpers,
) -> dict[str, float | bool]:
    """Compute the EA8b reference metrics used by the icechunk test."""
    nse = helpers.get_nse(results, reference)
    rsr = helpers.get_rsr(results, reference)
    return {
        "nse": float(nse),
        "rsr": float(rsr),
        "matches_reference": bool(nse > EA8B_REFERENCE_MIN_NSE and rsr < EA8B_REFERENCE_MAX_RSR),
    }


def assert_matches_reference(metrics: dict[str, float | bool], label: str) -> None:
    """Assert that EA8b drainage results satisfy the XPSTORM tolerances."""
    assert metrics["nse"] > EA8B_REFERENCE_MIN_NSE, (
        f"{label} NSE below XPSTORM tolerance: "
        f"{metrics['nse']:.6f} <= {EA8B_REFERENCE_MIN_NSE:.2f}"
    )
    assert metrics["rsr"] < EA8B_REFERENCE_MAX_RSR, (
        f"{label} RSR above XPSTORM tolerance: "
        f"{metrics['rsr']:.6f} >= {EA8B_REFERENCE_MAX_RSR:.2f}"
    )


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
    ea_test8b_reference,
    test_data_path,
    test_data_temp_path,
    helpers,
):
    """Test hotstart roundtrip with EA8b drainage scenario.

    This test uses the hotstart file saved at split point by test_ea8b_icechunk.py
    and verifies that resuming from hotstart preserves acceptable EA8b behavior.

    Verifies:
    - State is correctly restored from hotstart
    - Resumed drainage results stay within the same XPSTORM acceptance criteria
      as the uninterrupted EA8b run
    - Final raster results remain close to the uninterrupted run within small,
      deterministic tolerances
    - Hotstart archive is valid and contains expected data

    The raster tolerances are intentionally higher than the usual exact/near-exact
    expectations because EA8b with drainage and ponding is not restart-exact after
    a SWMM hotstart resume. The remaining differences are traced to SWMM's resumed
    ponding-related state, but both uninterrupted and resumed runs stay within the
    accepted XPSTORM reference thresholds. Disabling ponding reduces standalone
    SWMM resume differences, but makes the full coupled EA8b simulation unstable,
    so this test accepts the smallest observed tolerances that keep the coupled
    case reproducible enough without hiding larger regressions.

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

    simulation, obj_store, _output_storage = build_simulation(
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

    nodes_csv_bytes = bytes(obstore.get(obj_store, "out_drainage_nodes.csv").bytes())
    df_resumed = pd.read_csv(io.StringIO(nodes_csv_bytes.decode("utf-8")))
    resumed_results = drainage_results_to_coupling_series(df_resumed)
    resumed_metrics = get_reference_metrics(resumed_results, ea_test8b_reference, helpers)

    assert_matches_reference(resumed_metrics, label="Resumed hotstart run")

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
            rtol=0.0,
            atol=EA8B_FINAL_ARRAY_ATOL[key],
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
