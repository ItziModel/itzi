#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" """

import os
from io import StringIO
import dataclasses

import pandas as pd
import numpy as np
import grass.script as gscript
import pytest

from itzi import SimulationRunner
from itzi.data_containers import MassBalanceData


@pytest.mark.usefixtures("grass_5by5_sim")
def test_number_of_output():
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    assert current_mapset == "5by5"

    h_map_list = gscript.list_grouped("raster", pattern="*out_5by5_water_depth_*")[current_mapset]
    assert len(h_map_list) == 4

    wse_map_list = gscript.list_grouped("raster", pattern="*out_5by5_water_surface_elevation_*")[
        current_mapset
    ]
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

    verr_map_list = gscript.list_grouped("raster", pattern="*out_5by5_volume_error_*")[
        current_mapset
    ]
    assert len(verr_map_list) == 3


@pytest.mark.usefixtures("grass_5by5_sim")
def test_flow_symmetry():
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    assert current_mapset == "5by5"
    h_values = gscript.read_command(
        "v.what.rast", map="control_points", flags="p", raster="out_5by5_water_depth_0002"
    )
    s_h = pd.read_csv(StringIO(h_values), sep="|", names=["h"], usecols=[1]).squeeze()
    assert np.all(np.isclose(s_h[:-1], s_h[1:]))


@pytest.mark.usefixtures("grass_5by5")
def test_region_mask(test_data_path):
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


@pytest.mark.usefixtures("grass_5by5_max_values_sim")
def test_max_values():
    """Check if the maximum values of h and v are properly calculated."""
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    assert current_mapset == "5by5"
    h_maps = gscript.list_grouped("raster", pattern="*out_5by5_max_values_water_depth_*")[
        current_mapset
    ]
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
    h_max = gscript.raster_info("out_5by5_max_values_water_depth_max")
    h_max_test = gscript.raster_info("h_max_test")
    v_max = gscript.raster_info("out_5by5_max_values_v_max")
    v_max_test = gscript.raster_info("v_max_test")
    assert np.isclose(h_max["max"], h_max_test["max"])
    assert np.isclose(v_max["max"], v_max_test["max"])


@pytest.mark.usefixtures("grass_5by5_stats_sim")
def test_stats_file(test_data_temp_path):
    """Check if the statistics are accurate"""
    stats_path = os.path.join(test_data_temp_path, "5by5_stats.csv")
    assert os.path.exists(stats_path)
    df = pd.read_csv(stats_path)

    expected_cols = [f.name for f in dataclasses.fields(MassBalanceData)]
    assert df.columns.to_list() == expected_cols

    # Domain area in m2
    g_region = gscript.parse_command("g.region", flags="pg")
    area = float(g_region["nsres"]) * float(g_region["ewres"]) * float(g_region["cells"])

    # All rates are in mm/h, except inflow (m/s) and losses (m/s)
    expected_rain_vol = 10.0 / (1000 * 3600) * area
    expected_inf_vol = -2.0 / (1000 * 3600) * area
    expected_losses_vol = -1.5 / (1000 * 3600) * area
    expected_inflow_vol = 0.1 * area
    # Ignore first values as they are initial state, before time-stepping
    assert np.all(np.isclose(df["rainfall_volume"][1:], expected_rain_vol, atol=0.001))
    assert np.all(np.isclose(df["infiltration_volume"][1:], expected_inf_vol, atol=0.001))
    assert np.all(np.isclose(df["inflow_volume"][1:], expected_inflow_vol, atol=0.001))
    assert np.all(np.isclose(df["losses_volume"][1:], expected_losses_vol, atol=0.001))
    # Check if the volume change is coherent with the rest of the volumes
    df["vol_change_ref"] = (
        df["boundary_volume"]
        + df["rainfall_volume"]
        + df["infiltration_volume"]
        + df["inflow_volume"]
        + df["losses_volume"]
        + df["drainage_network_volume"]
        + df["volume_error"]
    )
    print(df.to_string())
    assert np.allclose(df["vol_change_ref"], df["volume_change"], atol=1, rtol=0.01)


@pytest.mark.usefixtures("grass_5by5_stats_sim")
def test_stats_maps():
    """Check if the maps statistics are accurate"""
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    for map_name in ["mean_rainfall", "mean_inflow", "mean_infiltration", "mean_losses"]:
        raster_maps = gscript.list_grouped("raster", pattern=f"*_{map_name}_*")[current_mapset]
        # initial stage + 5 time steps
        assert len(raster_maps) == 6
        for raster_map in raster_maps:
            stats = gscript.parse_command("r.univar", flags="g", map=raster_map)
            minimum = float(stats["min"])
            maximum = float(stats["max"])
            # Initial maps are expected to be zero (simulation has not started yet)
            if raster_map.endswith("0000"):
                assert np.isclose(minimum, 0)
                assert np.isclose(maximum, 0)
                continue
            else:
                if map_name == "mean_rainfall":
                    assert np.isclose(minimum, 10.0)
                    assert np.isclose(maximum, 10.0)
                elif map_name == "mean_inflow":
                    assert np.isclose(minimum, 0.1)
                    assert np.isclose(maximum, 0.1)
                elif map_name == "mean_infiltration":
                    assert np.isclose(minimum, 2.0)
                    assert np.isclose(maximum, 2.0)
                elif map_name == "mean_losses":
                    assert np.isclose(minimum, 1.5 / 3600 / 1000)
                    assert np.isclose(maximum, 1.5 / 3600 / 1000)
    # water depth
    first_depth_map = gscript.list_grouped("raster", pattern="*_water_depth_0000")[current_mapset]
    depth_stats = gscript.parse_command("r.univar", flags="g", map=first_depth_map)
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


@pytest.mark.usefixtures("grass_5by5_open_boundaries_sim")
def test_open_boundaries():
    """Check if the open boundary condition works on all sides"""
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    g_region = gscript.parse_command("g.region", flags="pg")
    # Works only if the the south-west corner is [0,0]
    max_x = float(g_region["e"])
    max_y = float(g_region["n"])
    boundaries_coords = gscript.read_command(
        "v.out.ascii", input="boundary_points", type="point", flags="c"
    )
    df_coords = pd.read_csv(
        StringIO(boundaries_coords), sep="|", header=0, names=["x", "y", "id"], index_col=2
    )

    boundaries_map_list = gscript.list_grouped(
        "raster", pattern="out_5by5_open_boundaries_boundaries_*"
    )[current_mapset]
    for boundary_map in boundaries_map_list:
        boundaries_values = gscript.read_command(
            "v.what.rast", map="boundary_points", flags="p", raster=boundary_map
        )
        if not boundaries_values.strip():
            continue
        df_values = pd.read_csv(
            StringIO(boundaries_values), sep="|", names=["id", "flow_rate"], index_col=0
        )
        df_merged = df_coords.join(df_values).dropna()

        coord_to_flow = df_merged.set_index(["x", "y"])["flow_rate"]
        for _, row in df_merged.iterrows():
            sym_x = max_x - row["x"]
            sym_y = max_y - row["y"]
            sym_flow = coord_to_flow.get((sym_x, sym_y))
            if sym_flow is not None:
                assert np.isclose(row["flow_rate"], sym_flow)


@pytest.mark.usefixtures("grass_5by5_wse_sim")
def test_wse():
    """Test if Water surface elevation is properly applied"""
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()

    h_maps = gscript.list_grouped("raster", pattern="*out_5by5_wse_water_depth_*")[current_mapset]
    gscript.run_command(
        "r.series",
        input=h_maps,
        output="h_test_max",
        method="maximum",
        overwrite=True,
    )

    wse_maps = gscript.list_grouped("raster", pattern="*out_5by5_wse_water_surface_elevation_*")[
        current_mapset
    ]
    gscript.run_command(
        "r.series",
        input=wse_maps,
        output="wse_test_max",
        method="maximum",
        overwrite=True,
    )
    h_test_max_univar = gscript.parse_command("r.univar", flags="g", map="h_test_max")
    wse_test_max_univar = gscript.parse_command("r.univar", flags="g", map="wse_test_max")

    print(f"{h_test_max_univar=}")
    print(f"{wse_test_max_univar=}")
    h_max = float(h_test_max_univar["max"])
    wse_max = float(wse_test_max_univar["max"])
    wse_min = float(wse_test_max_univar["min"])
    assert np.isclose(h_max, 0.2, atol=0.000005)
    assert np.isclose(wse_max, 132.2)
    assert np.isclose(wse_min, 132)
