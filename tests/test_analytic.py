#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test itzi against analytic solutions to the shallow water equation.
"""

import os
from io import StringIO

import numpy as np
import pandas as pd
import pytest
import grass.script as gscript

from itzi import SimulationRunner


@pytest.fixture(scope="session")
def mcdo_norain_reference(test_data_path):
    """ """
    file_path = os.path.join(
        test_data_path, "McDonald_long_channel_wo_rain", "mcdo_norain.csv"
    )
    return pd.read_csv(file_path)


@pytest.fixture(scope="class")
def grass_mcdo_norain(grass_xy_session, test_data_path):
    """Create a domain for MacDonald 1D solution long channel without rain.
    Delestre, O., Lucas, C., Ksinant, P.-A., Darboux, F., Laguerre, C., Vo, T.-N.-T., … Cordier, S. (2013).
    SWASHES: a compilation of shallow water analytic solutions for hydraulic and environmental studies.
    International Journal for Numerical Methods in Fluids, 72(3), 269–300. https://doi.org/10.1002/fld.3741
    """
    data_dir = os.path.join(test_data_path, "McDonald_long_channel_wo_rain")
    dem_path = os.path.join(data_dir, "dem.asc")
    bctype_path = os.path.join(data_dir, "bctype.asc")
    inflow_path = os.path.join(data_dir, "q.asc")
    points_path = os.path.join(data_dir, "axis_points.json")
    # Create new mapset
    gscript.run_command("g.mapset", mapset="mcdo_norain", flags="c")
    # Load raster data
    gscript.run_command("r.in.gdal", input=dem_path, output="dem")
    gscript.run_command("r.in.gdal", input=bctype_path, output="bctype")
    gscript.run_command("r.in.gdal", input=inflow_path, output="inflow")
    # Generate Manning map
    gscript.run_command("g.region", raster="dem", flags="o")
    region = gscript.parse_command("g.region", flags="pg")
    assert int(region["cells"]) == 600
    gscript.mapcalc("n=0.033")
    # Load axis points vector
    gscript.run_command("v.in.ogr", input=points_path, output="axis_points", flags="o")
    return None


@pytest.fixture(scope="class")
def grass_mcdo_norain_sim(grass_mcdo_norain, test_data_path):
    """ """
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    assert current_mapset == "mcdo_norain"
    # accessible_mapsets = gscript.read_command('g.mapsets', flags='p').rstrip()
    config_file = os.path.join(
        test_data_path, "McDonald_long_channel_wo_rain", "mcdo_norain.ini"
    )
    sim_runner = SimulationRunner()
    assert isinstance(sim_runner, SimulationRunner)
    sim_runner.initialize(config_file)
    sim_runner.run().finalize()
    return sim_runner


@pytest.fixture(scope="session")
def mcdo_rain_reference(test_data_path):
    """ """
    file_path = os.path.join(
        test_data_path, "McDonald_long_channel_rain", "mcdo_rain.csv"
    )
    return pd.read_csv(file_path)


@pytest.fixture(scope="function")
def grass_mcdo_rain(grass_xy_session, test_data_path):
    """Create a domain for MacDonald 1D solution long channel with rain.
    Delestre, O., Lucas, C., Ksinant, P.-A., Darboux, F., Laguerre, C., Vo, T.-N.-T., … Cordier, S. (2013).
    SWASHES: a compilation of shallow water analytic solutions for hydraulic and environmental studies.
    International Journal for Numerical Methods in Fluids, 72(3), 269–300. https://doi.org/10.1002/fld.3741
    """
    data_dir = os.path.join(test_data_path, "McDonald_long_channel_rain")
    dem_path = os.path.join(data_dir, "dem.asc")
    bctype_path = os.path.join(data_dir, "bctype.asc")
    inflow_path = os.path.join(data_dir, "q.asc")
    points_path = os.path.join(data_dir, "axis_points.json")
    # Create new mapset
    gscript.run_command("g.mapset", mapset="mcdo_rain", flags="c")
    # Load raster data
    gscript.run_command("r.in.gdal", input=dem_path, output="dem")
    gscript.run_command("r.in.gdal", input=bctype_path, output="bctype")
    gscript.run_command("r.in.gdal", input=inflow_path, output="inflow")
    # Create Manning map
    gscript.run_command("g.region", raster="dem", flags="o")
    region = gscript.parse_command("g.region", flags="pg")
    assert int(region["cells"]) == 600
    gscript.mapcalc("n=0.033")
    # Create rain map
    gscript.mapcalc("rain=3600")
    # Load axis points vector
    gscript.run_command("v.in.ogr", input=points_path, output="axis_points", flags="o")
    return None


@pytest.fixture(scope="function")
def grass_mcdo_rain_sim(grass_mcdo_rain, test_data_path):
    """ """
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    assert current_mapset == "mcdo_rain"
    config_file = os.path.join(
        test_data_path, "McDonald_long_channel_rain", "mcdo_rain.ini"
    )
    sim_runner = SimulationRunner()
    sim_runner.initialize(config_file)
    sim_runner.run().finalize()
    return sim_runner


@pytest.mark.usefixtures("mcdo_norain_reference", "grass_mcdo_norain_sim")
class TestMcdo_norain:
    def test_mcdo_norain(self, mcdo_norain_reference):
        current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
        assert current_mapset == "mcdo_norain"

        wse = gscript.read_command(
            "v.what.rast",
            map="axis_points",
            raster="out_mcdo_norain_wse_0004",
            flags="p",
        )
        df_wse = pd.read_csv(StringIO(wse), sep="|", names=["wse_model"], usecols=[1])
        df_results = mcdo_norain_reference.join(df_wse)
        df_results["abs_error"] = np.abs(df_results["wse_model"] - df_results["wse"])
        mae = np.mean(df_results["abs_error"])
        assert mae < 0.03

    def test_flow_is_unidimensional(self):
        """In the MacDonald 1D test, flow should be unidimensional in x dimension"""
        current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
        assert current_mapset == "mcdo_norain"

        map_list = gscript.list_grouped("raster", pattern="out_mcdo_norain_qy*")[
            current_mapset
        ]
        for raster in map_list:
            univar = gscript.parse_command("r.univar", map=raster, flags="g")
            assert float(univar["min"]) == 0
            assert float(univar["max"]) == 0


def test_mcdo_rain(grass_mcdo_rain_sim, mcdo_rain_reference):
    current_mapset = gscript.read_command("g.mapset", flags="p").rstrip()
    assert current_mapset == "mcdo_rain"
    wse = gscript.read_command(
        "v.what.rast", map="axis_points", raster="out_mcdo_rain_wse_0001", flags="p"
    )
    df_wse = pd.read_csv(StringIO(wse), sep="|", names=["wse_model"], usecols=[1])
    df_results = mcdo_rain_reference.join(df_wse)
    df_results["abs_error"] = np.abs(df_results["wse_model"] - df_results["wse"])
    mae = np.mean(df_results["abs_error"])
    assert mae < 0.04
