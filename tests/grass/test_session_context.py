"""Isolate session transitions which can otherwise exit through the GRASS C API."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_bootstrap_outside_grass_discovers_python_path_and_cleans_session(tmp_path: Path) -> None:
    import grass.script as gscript

    gscript.setup.setup_runtime_env()
    gscript.create_project(tmp_path, name="bootstrap")
    with gscript.setup.init(
        tmp_path,
        location="bootstrap",
        mapset="PERMANENT",
        grass_path="grass",
        env=os.environ.copy(),
    ) as runtime:
        gscript.run_command("g.region", n=30, s=0, e=50, w=0, res=10, env=runtime.env)
    environment = os.environ | {"PYTHONPATH": str(Path(__file__).parents[2] / "src")}
    environment.pop("GISRC", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
import sys
from pathlib import Path
from itzi.grass.session import GrassParams, GrassSessionManager

assert "grass.script" not in sys.modules
assert GrassSessionManager.current_context() is None
with GrassSessionManager(GrassParams(sys.argv[1], "bootstrap", "PERMANENT")) as manager:
    from itzi.grass.utils import read_domain
    assert (read_domain().rows, read_domain().cols) == (3, 5)
    gisrc = Path(os.environ["GISRC"])
assert "GISRC" not in os.environ
assert not gisrc.exists()
""",
            str(tmp_path),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("inherited", ["none", "WIND_OVERRIDE", "GRASS_REGION", "both"])
def test_mapset_switches_restore_caller_and_owned_sessions(tmp_path: Path, inherited: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from itzi.grass.session import GrassParams, GrassSessionManager

assert "grass.pygrass" not in sys.modules
assert "itzi.grass.interface" not in sys.modules
assert "itzi.grass.utils" not in sys.modules
import grass.script as gs

database = Path(sys.argv[1])
gs.setup.setup_runtime_env()
gs.create_project(database, name="project")
gs.create_project(database, name="second")
with gs.setup.init(database, location="project", mapset="PERMANENT", grass_path="grass"):
    gs.run_command("g.region", n=50, s=0, e=50, w=0, res=10)
    gs.mapcalc("gradient=float(row()*10+col())")
    gs.mapcalc("MASK=if(col()==1, 0, 1)")
    gs.run_command("g.region", n=40, s=10, e=50, w=0, res=10, flags="u", save="outer")
    gs.run_command("g.mapset", flags="c", mapset="target")
    gs.run_command("g.region", n=30, s=0, e=40, w=0, res=10)
    gs.mapcalc("gradient=float(100+row()*10+col())")
    gs.mapcalc("MASK=if(col()==4, 0, 1)")
    gs.run_command("g.mapset", mapset="PERMANENT")
    if sys.argv[2] in ("WIND_OVERRIDE", "both"):
        os.environ["WIND_OVERRIDE"] = "outer"
    if sys.argv[2] in ("GRASS_REGION", "both"):
        os.environ["GRASS_REGION"] = gs.region_env(region="outer")
    os.environ["GRASS_MASK"] = "MASK@PERMANENT"
    previous = {key: os.environ.get(key) for key in ("GISRC", "GRASS_REGION", "WIND_OVERRIDE", "GRASS_MASK")}
    gisrc_contents = Path(previous["GISRC"]).read_bytes()
    original_region = gs.region()
    saved_regions = gs.list_strings("region")
    params = GrassParams(str(database), "project", "target")
    start = datetime(2020, 1, 1)
    for index in range(2):
        with GrassSessionManager(params) as manager:
            from grass.lib import raster as lr
            import grass.temporal as tgis
            from itzi.grass.interface import GrassInterface
            assert gs.gisenv()["MAPSET"] == "target"
            assert manager.current_context()[2] == "target"
            assert tgis.get_current_mapset() == "target"
            assert manager.effective_mask.source == "MASK@target"
            assert (lr.Rast_window_rows(), lr.Rast_window_cols()) == (3, 4)
            interface = GrassInterface(start, start + timedelta(seconds=1), np.float32)
            values = interface.read_raster_map("gradient@target")
            expected = 100 + np.arange(10, 40, 10)[:, None] + np.arange(1, 5)
            np.testing.assert_allclose(values[:, :3], expected[:, :3])
            assert np.isnan(values[:, 3]).all()
            interface.write_raster_map(values, f"roundtrip_{index}@target", "friction", 0)
            np.testing.assert_array_equal(interface.read_raster_map(f"roundtrip_{index}@target"), values)
            with GrassSessionManager(GrassParams()):
                assert gs.region()["rows"] == 3
                assert gs.gisenv()["MAPSET"] == "target"
        assert {key: os.environ.get(key) for key in previous} == previous
        assert gs.region() == original_region
        assert manager.current_context()[2] == "PERMANENT"
        assert tgis.get_current_mapset() == "PERMANENT"
        assert gs.list_strings("region") == saved_regions
        assert Path(previous["GISRC"]).read_bytes() == gisrc_contents
        with GrassSessionManager(GrassParams()):
            restored = GrassInterface(start, start + timedelta(seconds=1), np.float32)
            assert restored.yr == original_region["rows"]
            assert restored.get_npmask()[:, 0].all()
        assert not gs.find_file(f"roundtrip_{index}", mapset="PERMANENT")["name"]
    os.environ.pop("GRASS_REGION", None)
    os.environ.pop("WIND_OVERRIDE", None)
    os.environ.pop("GRASS_MASK", None)

# Cached modules and C libraries must follow newly acquired underlying sessions.
for project, rows in (("second", 3), ("project", 5), ("second", 4)):
    with gs.setup.init(database, location=project, mapset="PERMANENT", grass_path="grass"):
        gs.run_command("g.region", n=rows*10, s=0, e=50, w=0, res=10)
        gs.mapcalc("owned_gradient=float(row()*10+col())", overwrite=True)
        with GrassSessionManager(GrassParams()):
            assert (lr.Rast_window_rows(), lr.Rast_window_cols()) == (rows, 5)
        assert (lr.Rast_window_rows(), lr.Rast_window_cols()) == (rows, 5)
    with GrassSessionManager(GrassParams(str(database), project, "PERMANENT")) as manager:
        job_gisrc = os.environ["GISRC"]
        interface = GrassInterface(start, start + timedelta(seconds=1), np.float32)
        assert interface.yr == rows
        assert manager.current_context()[1] == project
        assert tgis.get_current_location() == project
        assert interface.read_raster_map("owned_gradient@PERMANENT").shape == (rows, 5)
    manager.close()
    assert "GISRC" not in os.environ
    assert not Path(job_gisrc).exists()
""",
            str(tmp_path),
            inherited,
        ],
        env=os.environ | {"PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
