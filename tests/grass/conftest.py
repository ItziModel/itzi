"""
Define pytest fixture common to GRASS-based test modules.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

# need to set the path to the GRASS Python library
grass_python_path = subprocess.check_output(
    ["grass", "--config", "python_path"], text=True
).strip()
sys.path.append(grass_python_path)
import grass.script as gscript  # noqa: E402
from grass.script.setup import SessionHandle  # noqa: E402

from itzi import (  # noqa: E402
    SimulationRunner,  # noqa: E402
)
from itzi.configreader import ConfigReader  # noqa: E402
from itzi.grass import session  # noqa: E402


@pytest.fixture(autouse=True)
def grass_test_environment() -> Iterator[None]:
    """Restore process state after GRASS tests change it."""
    environment = os.environ.copy()
    workdir = Path.cwd()
    try:
        yield
    finally:
        os.chdir(workdir)
        os.environ.clear()
        os.environ.update(environment)


@pytest.fixture(scope="session")
def grass_xy_session(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SessionHandle]:
    """Keep one GRASS session and its temporal helpers for the test suite."""
    tmp_path = tmp_path_factory.mktemp("grass")
    environment = os.environ.copy()
    gscript.set_raise_on_error(True)
    location_name = "xy"
    gscript.create_project(tmp_path, name=location_name)
    with gscript.setup.init(
        path=tmp_path, location=location_name, mapset="PERMANENT", grass_path="grass"
    ) as grass_session:
        import grass.temporal as tgis
        from grass.lib import gis as libgis, raster as libraster

        # GRASS C libraries cache session paths and windows across Python sessions.
        libgis.G__read_gisrc_path()
        libgis.G__read_gisrc_env()
        libgis.G_reset_mapsets()
        libgis.G_unset_window()
        libraster.Rast_unset_window()
        os.environ["GRASS_VERBOSE"] = "1"
        try:
            yield grass_session
        finally:
            tgis.stop_subprocesses()
            session._initialized_temporal_session = None
    os.environ.clear()
    os.environ.update(environment)


@pytest.fixture(scope="session")
def grass_5by5_template(
    grass_xy_session: SessionHandle, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, Path]:
    """Build the 5 by 5 input data once and save a pristine mapset copy."""
    from grass.pygrass.gis import make_mapset, set_current_mapset

    resolution = 10
    # Create new mapset
    make_mapset("5by5")
    set_current_mapset("5by5")
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
    session.GrassSessionManager.ensure_temporal_initialized()
    mapset_path = Path(gscript.gisenv()["GISDBASE"]) / "xy" / "5by5"
    template_path = tmp_path_factory.mktemp("grass_template") / "5by5"
    shutil.copytree(mapset_path, template_path)
    return mapset_path, template_path


@pytest.fixture
def grass_5by5(grass_5by5_template: tuple[Path, Path], tmp_path: Path) -> Iterator[None]:
    """Restore fresh data while reusing the same GRASS session and helpers."""
    from grass.lib import gis as libgis, raster as libraster
    from grass.pygrass.gis import set_current_mapset

    mapset_path, template_path = grass_5by5_template
    set_current_mapset("PERMANENT", "xy", str(mapset_path.parent.parent))
    shutil.rmtree(mapset_path)
    shutil.copytree(template_path, mapset_path)
    set_current_mapset("5by5")
    libgis.G_reset_mapsets()
    libgis.G_unset_window()
    libraster.Rast_unset_window()
    os.chdir(tmp_path)
    session.GrassSessionManager.ensure_temporal_initialized()
    yield


@pytest.fixture
def grass_5by5_sim(grass_5by5: None, test_data_path: str) -> SimulationRunner:
    config_file = os.path.join(test_data_path, "5by5", "5by5.ini")
    conf_data = ConfigReader(config_file)
    with session.GrassSessionManager(conf_data.grass_params):
        sim_runner = SimulationRunner(conf_data.sim_config).initialize()
        assert isinstance(sim_runner, SimulationRunner)
        try:
            sim_runner.run()
        finally:
            sim_runner.finalize()
    return sim_runner
