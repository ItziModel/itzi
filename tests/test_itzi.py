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
    wse_map_list = gscript.list_grouped("raster", pattern="*out_5by5_wse_*")[
        current_mapset
    ]
    assert len(wse_map_list) == 3
    fr_map_list = gscript.list_grouped("raster", pattern="*out_5by5_fr_*")[
        current_mapset
    ]
    assert len(fr_map_list) == 3
    v_map_list = gscript.list_grouped("raster", pattern="*out_5by5_v_*")[current_mapset]
    assert len(v_map_list) == 4
    vdir_map_list = gscript.list_grouped("raster", pattern="*out_5by5_vdir_*")[
        current_mapset
    ]
    assert len(vdir_map_list) == 3
    qx_map_list = gscript.list_grouped("raster", pattern="*out_5by5_qx_*")[
        current_mapset
    ]
    assert len(qx_map_list) == 3
    qy_map_list = gscript.list_grouped("raster", pattern="*out_5by5_qy_*")[
        current_mapset
    ]
    assert len(qy_map_list) == 3
    verr_map_list = gscript.list_grouped("raster", pattern="*out_5by5_verror_*")[
        current_mapset
    ]
    assert len(verr_map_list) == 3


def test_flow_symmetry(grass_5by5_sim):
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    assert current_mapset == "5by5"
    h_values = gscript.read_command(
        "v.what.rast", map="control_points", flags="p", raster="out_5by5_h_0001"
    )
    s_h = pd.read_csv(StringIO(h_values), sep="|", names=["h"], usecols=[1]).squeeze()
    assert np.all(np.isclose(s_h[:-1], s_h[1:]))


def test_region_mask(grass_5by5, test_data_path):
    """Check if temporary mask and region are set and teared down."""
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    assert current_mapset == "5by5"
    # Get data from initial region and mask
    init_ncells = int(gscript.parse_command("g.region", flags="pg")["cells"])
    init_nulls = int(
        gscript.parse_command("r.univar", map="z", flags="g")["null_cells"]
    )
    # Set simulation (should set region and mask)
    config_file = os.path.join(test_data_path, "5by5", "5by5_mask.ini")
    sim_runner = SimulationRunner()
    sim_runner.initialize(config_file)
    # Run simulation
    sim_runner.run().finalize()
    # Check temporary mask and region
    assert (
        int(gscript.parse_command("r.univar", map="out_5by5_v_max", flags="g")["n"])
        == 9
    )
    # Check tear down
    assert int(gscript.parse_command("g.region", flags="pg")["cells"]) == init_ncells
    assert (
        int(gscript.parse_command("r.univar", map="z", flags="g")["null_cells"])
        == init_nulls
    )
    return sim_runner


# TODO: Add test for asymmetrical cells
