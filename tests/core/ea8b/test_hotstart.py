"""
Integration tests for hotstart functionality with the EA test case 8b.

The shared expensive simulation (EA8b with drainage, running from t=0 to t=3h20m)
is built once by the ``ea8b_simulation`` fixture in conftest.py.  Both test functions
below depend on it, so the simulation artifacts (hotstart at split point, final
raster state, hotstart at end) are guaranteed to exist before either test runs.
"""

from __future__ import annotations

import io
import json
import hashlib
import os
import zipfile
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import obstore

from itzi.data_containers import SimulationConfig, SurfaceFlowParameters
from itzi.const import TemporalType
from itzi.hotstart import HotstartLoader

from tests.core.ea8b.helpers import (
    EA8B_FINAL_ARRAY_ATOL,
    assert_matches_reference,
    build_resumed_simulation,
    drainage_results_to_coupling_series,
    get_reference_metrics,
)


pytestmark = pytest.mark.cloud


@pytest.mark.slow
def test_ea8b_hotstart_roundtrip(
    ea8b_simulation,
    ea8b_reference,
    ea8b_data,
    test_data_path,
    test_data_temp_path,
    helpers,
):
    """Test that resuming from hotstart at the split point reproduces the full run.

    Verifies that:
    - Simulation time and raster state are correctly restored from hotstart
    - Resumed drainage results satisfy the XPSTORM acceptance criteria
    - Final raster arrays are close to the uninterrupted reference within
      deterministic tolerances

    The raster tolerances are intentionally higher than the usual exact/near-exact
    expectations because EA8b with drainage and ponding is not restart-exact after
    a SWMM hotstart resume.  The remaining differences are traced to SWMM's
    ponding-related state, but both uninterrupted and resumed runs stay within the
    accepted XPSTORM reference thresholds.
    """
    hotstart_split_path = ea8b_simulation["hotstart_split_path"]
    final_state_path = ea8b_simulation["final_state_path"]
    split_time = ea8b_simulation["split_time"]
    sim_start_time = ea8b_simulation["sim_start_time"]
    sim_end_time = sim_start_time + timedelta(hours=3, minutes=20)

    surface_flow_params = SurfaceFlowParameters(cfl=0.5, theta=0.7)
    sim_config = SimulationConfig(
        start_time=sim_start_time,
        end_time=sim_end_time,
        record_step=timedelta(seconds=30),
        temporal_type=TemporalType.RELATIVE,
        input_map_names={"dem": "dem", "friction": "friction"},
        output_map_names={"water_depth": "test_water_depth"},
        drainage_output="out_drainage",
        swmm_inp=f"{test_data_path}/EA_test_8/b/test8b_drainage_ponding.inp",
        stats_file="ea8b_hotstart.csv",
        surface_flow_parameters=surface_flow_params,
        orifice_coeff=1.0,
    )

    os.chdir(test_data_temp_path)

    hotstart_loader = HotstartLoader.from_file(hotstart_split_path)

    with open(hotstart_split_path, "rb") as f:
        hotstart_bytes = f.read()

    simulation, obj_store = build_resumed_simulation(sim_config, ea8b_data, hotstart_bytes)

    assert simulation.sim_time == split_time

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

    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()

    nodes_csv_bytes = bytes(obstore.get(obj_store, "out_drainage_nodes.csv").bytes())
    df_resumed = pd.read_csv(io.StringIO(nodes_csv_bytes.decode("utf-8")))
    resumed_results = drainage_results_to_coupling_series(df_resumed)
    resumed_metrics = get_reference_metrics(resumed_results, ea8b_reference, helpers)

    assert_matches_reference(resumed_metrics, label="Resumed hotstart run")

    final_state = np.load(final_state_path, allow_pickle=False)

    for key in ["water_depth", "qe", "qs"]:
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
def test_ea8b_hotstart_archive_validity(ea8b_simulation):
    """Verify the hotstart archive at the split point is structurally valid.

    Checks:
    - Archive is a well-formed ZIP with required members (metadata.json,
      raster_state.npz, swmm_hotstart.hsf)
    - Hotstart version is correct
    - Simulation state and domain data are present
    - Hashes in metadata match the actual file contents
    """
    hotstart_split_path = ea8b_simulation["hotstart_split_path"]

    with zipfile.ZipFile(hotstart_split_path, "r") as zip_ref:
        members = zip_ref.namelist()

        assert "metadata.json" in members, "Missing metadata.json"
        assert "raster_state.npz" in members, "Missing raster_state.npz"
        assert "swmm_hotstart.hsf" in members, "Missing swmm_hotstart.hsf"

        with zip_ref.open("metadata.json") as metadata_file:
            metadata_dict = json.load(metadata_file)

        assert metadata_dict["hotstart_version"] == 1

        assert "simulation_state" in metadata_dict
        sim_state = metadata_dict["simulation_state"]

        ref_raster_hash = sim_state["raster_domain_hash"]
        hash_raster = hashlib.blake2b(zip_ref.read("raster_state.npz")).hexdigest()
        assert hash_raster == ref_raster_hash

        ref_swmm_hash = sim_state["swmm_hotstart_hash"]
        hash_swmm = hashlib.blake2b(zip_ref.read("swmm_hotstart.hsf")).hexdigest()
        assert hash_swmm == ref_swmm_hash

        assert "domain_data" in metadata_dict
        domain_data = metadata_dict["domain_data"]
        assert domain_data["rows"] == 200
        assert domain_data["cols"] == 482
