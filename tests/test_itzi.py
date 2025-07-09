#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" """

import os
from io import StringIO

import pandas as pd
import numpy as np
import grass.script as gscript

from itzi import SimulationRunner


def test_number_of_output(grass_5by5_sim):
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    assert current_mapset == "5by5"

    h_map_list = gscript.list_grouped("raster", pattern="*out_5by5_h_*")[current_mapset]
    assert len(h_map_list) == 4

    wse_map_list = gscript.list_grouped("raster", pattern="*out_5by5_wse_*")[current_mapset]
    assert len(wse_map_list) == 3

    fr_map_list = gscript.list_grouped("raster", pattern="*out_5by5_froude_*")[current_mapset]
    assert len(fr_map_list) == 3

    v_map_list = gscript.list_grouped("raster", pattern="*out_5by5_v_*")[current_mapset]
    assert len(v_map_list) == 4

    vdir_map_list = gscript.list_grouped("raster", pattern="*out_5by5_vdir_*")[current_mapset]
    assert len(vdir_map_list) == 3

    qx_map_list = gscript.list_grouped("raster", pattern="*out_5by5_qx_*")[current_mapset]
    assert len(qx_map_list) == 3

    qy_map_list = gscript.list_grouped("raster", pattern="*out_5by5_qy_*")[current_mapset]
    assert len(qy_map_list) == 3

    verr_map_list = gscript.list_grouped("raster", pattern="*out_5by5_verror_*")[current_mapset]
    assert len(verr_map_list) == 3


def test_flow_symmetry(grass_5by5_sim):
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    assert current_mapset == "5by5"
    h_values = gscript.read_command(
        "v.what.rast", map="control_points", flags="p", raster="out_5by5_h_0002"
    )
    s_h = pd.read_csv(StringIO(h_values), sep="|", names=["h"], usecols=[1]).squeeze()
    assert np.all(np.isclose(s_h[:-1], s_h[1:]))


def test_region_mask(grass_5by5, test_data_path):
    """Check if temporary mask and region are set and teared down."""
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    assert current_mapset == "5by5"
    # Get data from initial region and mask
    init_ncells = int(gscript.parse_command("g.region", flags="pg")["cells"])
    init_nulls = int(gscript.parse_command("r.univar", map="z", flags="g")["null_cells"])
    # Set simulation (should set region and mask)
    config_file = os.path.join(test_data_path, "5by5", "5by5_mask.ini")
    sim_runner = SimulationRunner()
    sim_runner.initialize(config_file)
    # Run simulation
    sim_runner.run().finalize()
    # Check temporary mask and region
    assert int(gscript.parse_command("r.univar", map="out_5by5_v_max", flags="g")["n"]) == 9
    # Check tear down
    assert int(gscript.parse_command("g.region", flags="pg")["cells"]) == init_ncells
    assert int(gscript.parse_command("r.univar", map="z", flags="g")["null_cells"]) == init_nulls
    return sim_runner


# TODO: Add test for asymmetrical cells


def test_max_values(grass_5by5_max_values_sim):
    """Check if the maximum values of h and v are properly calculated."""
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    assert current_mapset == "5by5"
    h_maps = gscript.list_grouped("raster", pattern="*out_5by5_max_values_h_*")[current_mapset]
    v_maps = gscript.list_grouped("raster", pattern="*out_5by5_max_values_v_*")[current_mapset]
    gscript.run_command(
        "r.series",
        input=h_maps,
        output="h_max_test",
        method="maximum",
        overwrite=True,
    )
    gscript.run_command(
        "r.series",
        input=v_maps,
        output="v_max_test",
        method="maximum",
        overwrite=True,
    )
    h_max = gscript.raster_info("out_5by5_max_values_h_max")
    h_max_test = gscript.raster_info("h_max_test")
    v_max = gscript.raster_info("out_5by5_max_values_v_max")
    v_max_test = gscript.raster_info("v_max_test")
    assert np.isclose(h_max["max"], h_max_test["max"])
    assert np.isclose(v_max["max"], v_max_test["max"])


def test_stats_file(grass_5by5_stats_sim, test_data_temp_path):
    """Check if the statistics are accurate"""
    stats_path = os.path.join(test_data_temp_path, "5by5_stats.csv")
    assert os.path.exists(stats_path)
    df = pd.read_csv(stats_path)

    expected_cols = [
        "sim_time",
        "avg_timestep",
        "#timesteps",
        "boundary_vol",
        "rain_vol",
        "inf_vol",
        "inflow_vol",
        "losses_vol",
        "drain_net_vol",
        "domain_vol",
        "created_vol",
        "%error",
    ]
    assert df.columns.to_list() == expected_cols

    # Domain area in m2
    g_region = gscript.parse_command("g.region", flags="pg")
    area = float(g_region["nsres"]) * float(g_region["ewres"]) * float(g_region["cells"])

    # All rates are in mm/h, except inflow (m/s) and losses (m/s)
    expected_rain_vol = 10.0 / (1000 * 3600) * area
    expected_inf_vol = -2.0 / (1000 * 3600) * area
    expected_losses_vol = -1.5 / (1000 * 3600) * area
    expected_inflow_vol = 0.1 * area
    assert np.all(np.isclose(df["rain_vol"], expected_rain_vol))
    assert np.all(np.isclose(df["inf_vol"], expected_inf_vol))
    assert np.all(np.isclose(df["inflow_vol"], expected_inflow_vol))
    assert np.all(np.isclose(df["losses_vol"], expected_losses_vol))


def test_stats_maps(grass_5by5_stats_sim):
    """Check if the maps statistics are accurate"""
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    for map_name in ["rainfall", "inflow", "infiltration", "losses"]:
        raster_maps = gscript.list_grouped("raster", pattern=f"*_{map_name}_*")[current_mapset]
        # initial stage + 5 time steps
        assert len(raster_maps) == 6
        for raster_map in raster_maps:
            stats = gscript.parse_command("r.univar", flags="g", map=raster_map)
            minimum = float(stats["min"])
            maximum = float(stats["max"])
            # Initial maps are expected to be zero (simulation has not started yet)
            print(f"{raster_map=}, {minimum=}, {maximum=}")
            if raster_map.endswith('0000'):
                assert np.isclose(minimum, 0)
                assert np.isclose(maximum, 0)
                continue
            # Ignore the very first time step
            if not raster_map.endswith('0001'):
                if map_name == "rainfall":
                    assert np.isclose(minimum, 10.0)
                    assert np.isclose(maximum, 10.0)
                elif map_name == "inflow":
                    assert np.isclose(minimum, 0.1)
                    assert np.isclose(maximum, 0.1)
                elif map_name == "infiltration":
                    assert np.isclose(minimum, 2.0)
                    assert np.isclose(maximum, 2.0)
                elif map_name == "losses":
                    assert np.isclose(minimum, 1.5 / 3600 / 1000)
                    assert np.isclose(maximum, 1.5 / 3600 / 1000)
    # water depth
    first_depth_map = gscript.list_grouped("raster", pattern=f"*_h_0000")[current_mapset]
    depth_stats = gscript.parse_command("r.univar", flags="g", map=first_depth_map)
    print(f"{depth_stats=}")
    depth_n = int(depth_stats["n"])
    depth_null_cells = int(depth_stats["null_cells"])
    depth_mean = float(depth_stats["mean"])
    depth_min = float(depth_stats["min"])
    depth_max = float(depth_stats["max"])
    # Only one cell with value. All values below a threshold are set to NaN
    assert depth_n == 1
    assert depth_null_cells == 24
    assert np.isclose(depth_mean, 0.2)
    assert np.isclose(depth_min, 0.2)
    assert np.isclose(depth_max, 0.2)
