#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Define pytest fixture common to various test modules.
"""

import os
import sys
import hashlib
import tempfile
from pathlib import Path
import subprocess

import pytest
import numpy as np

# need to set the path to the GRASS Python library
grass_python_path = subprocess.check_output(
    ["grass", "--config", "python_path"], text=True
).strip()
sys.path.append(grass_python_path)
import grass.script as gscript  # noqa: E402

from itzi import SimulationRunner  # noqa: E402

TESTS_ROOT = Path.cwd()


class Helpers:
    @staticmethod
    def get_rmse(model, ref):
        """return root mean square error"""
        return np.sqrt(np.mean((model - ref) ** 2))

    @staticmethod
    def get_nse(model, ref):
        """Nash-Sutcliffe Efficiency"""
        noise = np.mean((ref - model) ** 2)
        information = np.mean((ref - np.mean(ref)) ** 2)
        return 1 - (noise / information)

    @staticmethod
    def get_rsr(model, ref):
        """RMSE/StdDev ratio"""
        rmse = Helpers.get_rmse(model, ref)
        return rmse / np.std(ref)

    @staticmethod
    def roughness(timeseries):
        """Sum of the squared difference of
        the normalized differences.
        """
        f = timeseries.diff()
        normed_f = (f - f.mean()) / f.std()
        return (normed_f.diff() ** 2).sum()

    @staticmethod
    def sha256(file_path):
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    @staticmethod
    def md5(file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


@pytest.fixture(scope="session")
def helpers():
    return Helpers


@pytest.fixture(scope="session")
def test_data_path():
    """Path to the permanent test data directory."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, "test_data")


@pytest.fixture(scope="session")
def test_data_temp_path():
    """Directory where generated test data resides."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    temp_path = os.path.join(dir_path, "test_data_temp")
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    return temp_path


@pytest.fixture(scope="session")
def grass_xy_session(test_data_temp_path):
    """Create a GRASS session in a new XY location and PERMANENT mapset"""
    # Keep all generated files in the test_data_temp_path
    os.chdir(test_data_temp_path)
    tmpdir = tempfile.TemporaryDirectory(prefix="tests_itzi_")
    gscript.set_raise_on_error(True)
    # create a new location
    location_name = "xy"
    gscript.create_project(tmpdir.name, name=location_name)
    # set up session
    grass_session = gscript.setup.init(
        path=tmpdir.name, location=location_name, mapset="PERMANENT", grass_path="grass"
    )
    os.environ["GRASS_VERBOSE"] = "1"
    # os.environ['ITZI_VERBOSE'] = '4'
    # os.environ['GRASS_OVERWRITE'] = '1'
    yield grass_session
    grass_session.finish()
    tmpdir.cleanup()


@pytest.fixture(scope="class")
def grass_5by5(grass_xy_session, test_data_path):
    """Create a square, 5 by 5 domain."""
    resolution = 10
    # Create new mapset
    gscript.run_command("g.mapset", mapset="5by5", flags="c")
    # Create 3by5 named region
    gscript.run_command("g.region", res=resolution, s=10, n=40, w=0, e=50, save="3by5", flags="o")
    region = gscript.parse_command("g.region", flags="pg")
    assert int(region["cells"]) == 15
    # Create raster for mask (do not apply mask)
    gscript.run_command("g.region", res=resolution, s=0, n=50, w=10, e=40)
    region = gscript.parse_command("g.region", flags="pg")
    assert int(region["cells"]) == 15
    gscript.mapcalc("5by3=1")
    # Set a 5x5 region
    gscript.run_command("g.region", res=resolution, s=0, w=0, e=50, n=50)
    region = gscript.parse_command("g.region", flags="pg")
    assert int(region["cells"]) == 25
    # DEM
    gscript.mapcalc("z=0")
    univar_z = gscript.parse_command("r.univar", map="z", flags="g")
    assert int(univar_z["min"]) == 0
    assert int(univar_z["max"]) == 0
    # Manning
    gscript.mapcalc("n=0.05")
    univar_n = gscript.parse_command("r.univar", map="n", flags="g")
    assert float(univar_n["min"]) == 0.05
    assert float(univar_n["max"]) == 0.05
    # Start depth
    gscript.write_command("v.in.ascii", input="-", stdin="25|25", output="start_h")
    gscript.run_command(
        "v.to.rast",
        input="start_h",
        output="start_h",
        type="point",
        use="val",
        value=0.2,
    )
    # Set null values to 0
    gscript.run_command("r.null", map="start_h", null=0)
    univar_start_h = gscript.parse_command("r.univar", map="start_h", flags="g")
    assert float(univar_start_h["min"]) == 0
    assert float(univar_start_h["max"]) == 0.2
    # Symmetry control points
    control_points = os.path.join(test_data_path, "5by5", "control_points.csv")
    gscript.run_command(
        "v.in.ascii", input=control_points, output="control_points", separator="comma"
    )
    # Rate maps
    gscript.mapcalc("rainfall=10")
    gscript.mapcalc("infiltration_rate=2")
    gscript.mapcalc("loss_rate=1.5")
    gscript.mapcalc("inflow_rate=0.1")
    return None


@pytest.fixture(scope="class")
def grass_5by5_sim(grass_5by5, test_data_path):
    """ """
    config_file = os.path.join(test_data_path, "5by5", "5by5.ini")
    sim_runner = SimulationRunner()
    assert isinstance(sim_runner, SimulationRunner)
    sim_runner.initialize(config_file)
    sim_runner.run().finalize()
    return sim_runner


@pytest.fixture(scope="class")
def grass_5by5_max_values_sim(grass_5by5, test_data_path):
    """ """
    config_file = os.path.join(test_data_path, "5by5", "5by5_max_values.ini")
    sim_runner = SimulationRunner()
    assert isinstance(sim_runner, SimulationRunner)
    sim_runner.initialize(config_file)
    sim_runner.run().finalize()
    return sim_runner


@pytest.fixture(scope="class")
def grass_5by5_stats_sim(grass_5by5, test_data_path):
    """ """
    config_file = os.path.join(test_data_path, "5by5", "5by5_stats.ini")
    sim_runner = SimulationRunner()
    assert isinstance(sim_runner, SimulationRunner)
    sim_runner.initialize(config_file)
    sim_runner.run().finalize()
    return sim_runner
