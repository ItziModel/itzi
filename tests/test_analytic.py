#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test itzi against analytic solutions to the shallow water equation.
"""

import os
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest


from itzi.simulation_factories import create_memory_simulation
from itzi.configreader import ConfigReader
from itzi.rasterdomain import DomainData

ASCIIMetadata = namedtuple(
    "ASCIIMetadata", ["ncols", "nrows", "xllcorner", "yllcorner", "cellsize"]
)


def read_ascii_grid(filepath):
    filepath = Path(filepath)
    with filepath.open("r") as f:
        # Read header
        ncols = int(f.readline().split()[1])
        nrows = int(f.readline().split()[1])
        xllcorner = float(f.readline().split()[1])
        yllcorner = float(f.readline().split()[1])
        cellsize = float(f.readline().split()[1])
        # Read the grid data
        data = np.loadtxt(f)
    assert data.shape == (nrows, ncols)
    metadata = ASCIIMetadata(ncols, nrows, xllcorner, yllcorner, cellsize)
    return data, metadata


def metadata_to_domaindata(metadata: ASCIIMetadata) -> DomainData:
    rows = metadata.nrows
    cols = metadata.ncols
    west = metadata.xllcorner
    east = west + cols * metadata.cellsize
    south = metadata.yllcorner
    north = south + rows * metadata.cellsize
    return DomainData(north, south, east, west, rows, cols)


@pytest.fixture(scope="module")
def mcdo_norain_sim(test_data_path, test_data_temp_path):
    """Run a simulation for MacDonald 1D solution long channel without rain.
    Delestre, O., Lucas, C., Ksinant, P.-A., Darboux, F., Laguerre, C., Vo, T.-N.-T., … Cordier, S. (2013).
    SWASHES: a compilation of shallow water analytic solutions for hydraulic and environmental studies.
    International Journal for Numerical Methods in Fluids, 72(3), 269–300. https://doi.org/10.1002/fld.3741
    """
    test_data_path = Path(test_data_path)
    data_dir = test_data_path / Path("analytic")
    reference_path = data_dir / Path("mcdo_norain.csv")
    reference = pd.read_csv(reference_path)
    arr_topo = reference["topo"].values
    # Create DEM
    arr_dem = np.tile(arr_topo, (3, 1))
    assert arr_dem.shape == (3, 200)
    domain_data = DomainData(north=5 * 200, south=0, east=5 * 200, west=0, rows=3, cols=200)
    # Manning
    arr_n = np.full_like(arr_dem, fill_value=0.033)
    # Inflow at westmost boundary
    arr_inflow = np.zeros_like(arr_dem)
    arr_inflow[:, 0] = 0.4
    # free eastmost boundary
    arr_bctype = np.ones_like(arr_dem)
    arr_bctype[:, -1] = 2
    # No mask. Whole domain.
    array_mask = np.full(shape=arr_dem.shape, fill_value=False, dtype=np.bool_)

    # Run the simulation in the temp dir
    os.chdir(test_data_temp_path)
    config_file = data_dir / Path("mcdo_norain.ini")
    config = ConfigReader(config_file)
    simulation = create_memory_simulation(
        sim_times=config.sim_times,
        output_maps=config.output_map_names,
        sim_param=config.sim_param,
        drainage_params=config.drainage_params,
        domain_data=domain_data,
        arr_mask=array_mask,
        dtype=np.float32,
        stats_file=config.stats_file,
    )
    # Set the input arrays
    simulation.set_array("dem", arr_dem)
    simulation.set_array("bctype", arr_bctype)
    simulation.set_array("inflow", arr_inflow)
    simulation.set_array("friction", arr_n)
    # run the simulation
    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()
    return simulation, reference


class TestMcdo_norain:
    def test_mcdo_norain(self, mcdo_norain_sim):
        simulation, reference = mcdo_norain_sim
        raster_results = simulation.report.raster_provider.output_maps_dict
        wse_time, wse_array = raster_results["water_surface_elevation"][-1]
        wse_centerline = pd.DataFrame({"wse_model": wse_array[1, :]})
        df_results = reference.join(wse_centerline)
        df_results["abs_error"] = np.abs(df_results["wse_model"] - df_results["wse"])
        mae = np.mean(df_results["abs_error"])
        assert mae < 0.03

    def test_flow_is_unidimensional(self, mcdo_norain_sim):
        simulation, reference = mcdo_norain_sim
        """In the MacDonald 1D test, flow should be unidimensional in the X dimension"""
        qy_array_list = simulation.report.raster_provider.output_maps_dict["qy"]
        for _, qy_array in qy_array_list:
            print(qy_array)
            # univar = gscript.parse_command("r.univar", map=raster, flags="g")
            assert np.min(qy_array) == 0
            assert np.max(qy_array) == 0

    def test_stat_file_is_coherent(self, test_data_temp_path):
        stat_file_path = Path(test_data_temp_path) / Path("mcdo_norain.csv")
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
        assert np.allclose(df_stats["percent_error"], df_stats["err_ref"], atol=0.0001)

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


@pytest.fixture(scope="module")
def mcdo_rain_sim(test_data_path, test_data_temp_path):
    """Run a simulation for MacDonald 1D solution long channel with rain.
    Delestre, O., Lucas, C., Ksinant, P.-A., Darboux, F., Laguerre, C., Vo, T.-N.-T., … Cordier, S. (2013).
    SWASHES: a compilation of shallow water analytic solutions for hydraulic and environmental studies.
    International Journal for Numerical Methods in Fluids, 72(3), 269–300. https://doi.org/10.1002/fld.3741
    """
    data_dir = Path(test_data_path) / Path("analytic")

    # Create DEM
    reference_path = data_dir / Path("mcdo_rain.csv")
    reference = pd.read_csv(reference_path)
    arr_topo = reference["topo"].values
    arr_dem = np.tile(arr_topo, (3, 1))
    assert arr_dem.shape == (3, 200)
    domain_data = DomainData(north=5 * 200, south=0, east=5 * 200, west=0, rows=3, cols=200)
    # Create Manning map
    arr_n = np.full_like(arr_dem, fill_value=0.033)
    # Inflow at westmost boundary
    arr_inflow = np.zeros_like(arr_dem)
    arr_inflow[:, 0] = 0.2
    # free eastmost boundary
    arr_bctype = np.ones_like(arr_dem)
    arr_bctype[:, -1] = 2
    # Simulation object takes rainfall input in m/s.
    arr_rain = np.full_like(arr_dem, fill_value=0.001)
    # No mask. Whole domain.
    array_mask = np.full(shape=arr_dem.shape, fill_value=False, dtype=np.bool_)

    # Run the simulation in the temp dir
    os.chdir(test_data_temp_path)
    config_file = data_dir / Path("mcdo_rain.ini")
    config = ConfigReader(config_file)
    simulation = create_memory_simulation(
        sim_times=config.sim_times,
        output_maps=config.output_map_names,
        sim_param=config.sim_param,
        drainage_params=config.drainage_params,
        domain_data=domain_data,
        arr_mask=array_mask,
        dtype=np.float32,
        stats_file=config.stats_file,
    )
    # Set the input arrays
    simulation.set_array("dem", arr_dem)
    simulation.set_array("bctype", arr_bctype)
    simulation.set_array("inflow", arr_inflow)
    simulation.set_array("friction", arr_n)
    simulation.set_array("rain", arr_rain)
    # run the simulation
    simulation.initialize()
    while simulation.sim_time < simulation.end_time:
        simulation.update()
    simulation.finalize()
    return simulation, reference


def test_mcdo_rain(mcdo_rain_sim):
    simulation, reference = mcdo_rain_sim
    raster_results = simulation.report.raster_provider.output_maps_dict
    wse_time, wse_array = raster_results["water_surface_elevation"][-1]
    wse_centerline = pd.DataFrame({"wse_model": wse_array[1, :]})
    df_results = reference.join(wse_centerline)
    df_results["abs_error"] = np.abs(df_results["wse_model"] - df_results["wse"])
    mae = np.mean(df_results["abs_error"])
    print(mae)
    assert mae < 0.035
