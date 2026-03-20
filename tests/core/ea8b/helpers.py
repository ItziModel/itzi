import numpy as np
import pandas as pd
import pyproj
import icechunk
import obstore

from itzi.data_containers import SimulationConfig
from itzi.simulation_builder import SimulationBuilder
from itzi.providers.csv_output import CSVVectorOutputProvider
from itzi.providers.icechunk_output import IcechunkRasterOutputProvider
from itzi.providers.xarray_input import XarrayRasterInputProvider


EA8B_REFERENCE_MIN_NSE = 0.99
EA8B_REFERENCE_MAX_RSR = 0.01
EA8B_FINAL_ARRAY_ATOL: dict[str, float] = {
    "water_depth": 6.3e-3,
    "qe": 2.9e-3,
    "qs": 9.0e-4,
}


def drainage_results_to_coupling_series(df_results: pd.DataFrame) -> pd.Series:
    df_results = df_results.copy()
    df_results["sim_time"] = pd.to_timedelta(df_results["sim_time"])
    df_results["start_time"] = df_results["sim_time"].dt.total_seconds().astype(int)
    df_results.set_index("start_time", inplace=True)
    df_results.drop(columns=["sim_time"], inplace=True)
    df_results = df_results[df_results.index >= 3000]
    df_results.index = pd.to_timedelta(df_results.index, unit="s")
    return df_results["coupling_flow"]


def get_reference_metrics(
    results: pd.Series, reference: pd.Series, helpers
) -> dict[str, float | bool]:
    nse = helpers.get_nse(results, reference)
    rsr = helpers.get_rsr(results, reference)
    return {
        "nse": float(nse),
        "rsr": float(rsr),
        "matches_reference": bool(nse > EA8B_REFERENCE_MIN_NSE and rsr < EA8B_REFERENCE_MAX_RSR),
    }


def assert_matches_reference(metrics: dict[str, float | bool], label: str) -> None:
    assert metrics["nse"] > EA8B_REFERENCE_MIN_NSE, (
        f"{label} NSE below XPSTORM tolerance: "
        f"{metrics['nse']:.6f} <= {EA8B_REFERENCE_MIN_NSE:.2f}"
    )
    assert metrics["rsr"] < EA8B_REFERENCE_MAX_RSR, (
        f"{label} RSR above XPSTORM tolerance: "
        f"{metrics['rsr']:.6f} >= {EA8B_REFERENCE_MAX_RSR:.2f}"
    )


def build_resumed_simulation(
    sim_config: SimulationConfig,
    ea8b_data: dict,
    hotstart_bytes: bytes,
):
    arr_mask = np.zeros((ea8b_data["rows"], ea8b_data["cols"]), dtype=bool)

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

    output_storage = icechunk.in_memory_storage()
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
        .with_hotstart(hotstart_bytes)
        .build()
    )

    return simulation, obj_store
