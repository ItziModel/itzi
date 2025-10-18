"""
Define pytest fixture common to various test modules.
"""

import os
import sys
import tempfile
from pathlib import Path
import subprocess

import pytest

# need to set the path to the GRASS Python library
grass_python_path = subprocess.check_output(
    ["grass", "--config", "python_path"], text=True
).strip()
sys.path.append(grass_python_path)
import grass.script as gscript  # noqa: E402

from itzi import SimulationRunner  # noqa: E402

TESTS_ROOT = Path.cwd()


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
    # DEMs
    gscript.mapcalc("z=0")
    univar_z = gscript.parse_command("r.univar", map="z", flags="g")
    assert int(univar_z["min"]) == 0
    assert int(univar_z["max"]) == 0
    z_high_value = 132
    gscript.mapcalc(f"z_high={z_high_value}")
    univar_z = gscript.parse_command("r.univar", map="z_high", flags="g")
    assert int(univar_z["min"]) == z_high_value
    assert int(univar_z["max"]) == z_high_value
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
    # Start Water Surface Elevation
    gscript.run_command(
        "v.to.rast",
        input="start_h",
        output="start_wse",
        type="point",
        use="val",
        value=z_high_value + 0.2,
    )
    # Set null values to 0
    gscript.run_command("r.null", map="start_h", null=0)
    gscript.run_command(
        "r.null", map="start_wse", null=0
    )  # WSE will be lower than DEM, on purpose
    univar_start_h = gscript.parse_command("r.univar", map="start_h", flags="g")
    univar_start_wse = gscript.parse_command("r.univar", map="start_wse", flags="g")
    assert float(univar_start_h["null_cells"]) == 0
    assert float(univar_start_h["min"]) == 0
    assert float(univar_start_h["max"]) == 0.2
    assert float(univar_start_wse["null_cells"]) == 0
    assert float(univar_start_wse["min"]) == 0
    assert float(univar_start_wse["max"]) == z_high_value + 0.2
    # Symmetry control points
    control_points = os.path.join(test_data_path, "5by5", "control_points.csv")
    gscript.run_command(
        "v.in.ascii", input=control_points, output="control_points", separator="comma"
    )
    # Boundary condition map
    boundary_points = f"""
    {resolution / 2},{resolution / 2}
    {resolution / 2},{resolution / 2 + resolution * 1}
    {resolution / 2},{resolution / 2 + resolution * 2}
    {resolution / 2},{resolution / 2 + resolution * 3}
    {resolution / 2},{resolution / 2 + resolution * 4}
    {resolution / 2 + resolution * 1},{resolution / 2}
    {resolution / 2 + resolution * 2},{resolution / 2}
    {resolution / 2 + resolution * 3},{resolution / 2}
    {resolution / 2 + resolution * 4},{resolution / 2}
    {resolution / 2 + resolution * 4},{resolution / 2 + resolution * 1}
    {resolution / 2 + resolution * 4},{resolution / 2 + resolution * 2}
    {resolution / 2 + resolution * 4},{resolution / 2 + resolution * 3}
    {resolution / 2 + resolution * 4},{resolution / 2 + resolution * 4}
    {resolution / 2 + resolution * 3},{resolution / 2 + resolution * 4}
    {resolution / 2 + resolution * 2},{resolution / 2 + resolution * 4}
    {resolution / 2 + resolution * 1},{resolution / 2 + resolution * 4}"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=True) as boundary_points_file:
        boundary_points_file.write(boundary_points)
        boundary_points_file.flush()  # Ensure data is written
        gscript.run_command(
            "v.in.ascii",
            input=boundary_points_file.name,
            output="boundary_points",
            separator="comma",
        )
    gscript.run_command(
        "v.to.rast",
        input="boundary_points",
        output="open_boundaries",
        type="point",
        use="val",
        value=2,
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


@pytest.fixture(scope="class")
def grass_5by5_open_boundaries_sim(grass_5by5, test_data_path):
    """ """
    config_file = os.path.join(test_data_path, "5by5", "5by5_open_boundaries.ini")
    sim_runner = SimulationRunner()
    assert isinstance(sim_runner, SimulationRunner)
    sim_runner.initialize(config_file)
    sim_runner.run().finalize()
    return sim_runner


@pytest.fixture(scope="class")
def grass_5by5_wse_sim(grass_5by5, test_data_path):
    """ """
    config_file = os.path.join(test_data_path, "5by5", "5by5_wse.ini")
    sim_runner = SimulationRunner()
    assert isinstance(sim_runner, SimulationRunner)
    sim_runner.initialize(config_file)
    sim_runner.run().finalize()
    return sim_runner
