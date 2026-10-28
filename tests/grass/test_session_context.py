"""Check warm context restoration and isolate cold session lifecycle checks."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import grass.script as gscript
import numpy as np
import pytest
from grass.script.setup import SessionHandle

from itzi.grass.session import GrassParams, GrassSessionManager


@pytest.fixture(scope="module")
def grass_context_session(
    grass_xy_session: SessionHandle, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Path]:
    """Prepare both mapsets once and restore the suite's session after this module."""
    database = tmp_path_factory.mktemp("grass_context")
    gscript.create_project(database, name="project")
    environment = os.environ.copy()
    for name in ("GRASS_REGION", "WIND_OVERRIDE", "GRASS_MASK"):
        environment.pop(name, None)
    with gscript.setup.init(
        database,
        location="project",
        mapset="PERMANENT",
        grass_path="grass",
        env=environment,
    ) as runtime:
        gscript.run_command("g.region", n=50, s=0, e=50, w=0, res=10, env=runtime.env)
        gscript.mapcalc("gradient=float(row()*10+col())", env=runtime.env)
        gscript.mapcalc("MASK=if(col()==1, 0, 1)", env=runtime.env)
        gscript.run_command(
            "g.region", n=40, s=10, e=50, w=0, res=10, flags="u", save="outer", env=runtime.env
        )
        gscript.run_command("g.mapset", flags="c", mapset="target", env=runtime.env)
        gscript.run_command("g.region", n=30, s=0, e=40, w=0, res=10, env=runtime.env)
        gscript.mapcalc("gradient=float(100+row()*10+col())", env=runtime.env)
        gscript.mapcalc("MASK=if(col()==4, 0, 1)", env=runtime.env)
    with GrassSessionManager(GrassParams(str(database), "project", "PERMANENT")):
        yield database


def test_bootstrap_outside_grass_discovers_python_path_and_cleans_session(tmp_path: Path) -> None:
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
    for name in ("GISRC", "GRASS_REGION", "WIND_OVERRIDE", "GRASS_MASK"):
        environment.pop(name, None)
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
def test_mapset_switches_restore_caller(
    grass_context_session: Path, monkeypatch: pytest.MonkeyPatch, inherited: str
) -> None:
    import grass.temporal as tgis
    from grass.lib import gis as libgis, raster as libraster

    from itzi.grass.interface import GrassInterface

    monkeypatch.delenv("GRASS_REGION", raising=False)
    monkeypatch.delenv("WIND_OVERRIDE", raising=False)
    if inherited in ("WIND_OVERRIDE", "both"):
        monkeypatch.setenv("WIND_OVERRIDE", "outer")
    if inherited in ("GRASS_REGION", "both"):
        monkeypatch.setenv("GRASS_REGION", gscript.region_env(region="outer"))
    monkeypatch.setenv("GRASS_MASK", "MASK@PERMANENT")
    libgis.G_unset_window()
    libraster.Rast_unset_window()
    previous = {
        key: os.environ.get(key)
        for key in ("GISRC", "GRASS_REGION", "WIND_OVERRIDE", "GRASS_MASK")
    }
    gisrc = Path(os.environ["GISRC"])
    gisrc_contents = gisrc.read_bytes()
    original_region = gscript.region()
    saved_regions = gscript.list_strings("region")
    params = GrassParams(str(grass_context_session), "project", "target")
    start = datetime(2020, 1, 1)
    for index in range(2):
        with GrassSessionManager(params) as manager:
            assert gscript.gisenv()["MAPSET"] == "target"
            assert manager.current_context()[2] == "target"
            assert tgis.get_current_mapset() == "target"
            assert manager.effective_mask.source == "MASK@target"
            assert (libraster.Rast_window_rows(), libraster.Rast_window_cols()) == (3, 4)
            interface = GrassInterface(start, start + timedelta(seconds=1), np.float32)
            values = interface.read_raster_map("gradient@target")
            expected = 100 + np.arange(10, 40, 10)[:, None] + np.arange(1, 5)
            np.testing.assert_allclose(values[:, :3], expected[:, :3])
            assert np.isnan(values[:, 3]).all()
            output_name = f"roundtrip_{inherited}_{index}"
            output_id = f"{output_name}@target"
            interface.write_raster_map(values, output_id, "friction", 0)
            np.testing.assert_array_equal(interface.read_raster_map(output_id), values)
            with GrassSessionManager(GrassParams()):
                assert gscript.region()["rows"] == 3
                assert gscript.gisenv()["MAPSET"] == "target"
        assert {key: os.environ.get(key) for key in previous} == previous
        assert gscript.region() == original_region
        assert manager.current_context()[2] == "PERMANENT"
        assert tgis.get_current_mapset() == "PERMANENT"
        assert gscript.list_strings("region") == saved_regions
        assert gisrc.read_bytes() == gisrc_contents
        with GrassSessionManager(GrassParams()):
            restored = GrassInterface(start, start + timedelta(seconds=1), np.float32)
            assert restored.yr == original_region["rows"]
            assert restored.get_npmask()[:, 0].all()
        assert not gscript.find_file(output_name, mapset="PERMANENT")["name"]


def test_owned_sessions_refresh_cached_libraries_and_clean_up(tmp_path: Path) -> None:
    environment = os.environ | {"PYTHONPATH": os.pathsep.join(sys.path)}
    for name in ("GISRC", "GRASS_REGION", "WIND_OVERRIDE", "GRASS_MASK"):
        environment.pop(name, None)
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

# Cached modules and C libraries must follow newly acquired underlying sessions.
start = datetime(2020, 1, 1)
for project, rows in (("second", 3), ("project", 5), ("second", 4)):
    with gs.setup.init(database, location=project, mapset="PERMANENT", grass_path="grass"):
        gs.run_command("g.region", n=rows*10, s=0, e=50, w=0, res=10)
        gs.mapcalc("owned_gradient=float(row()*10+col())", overwrite=True)
        with GrassSessionManager(GrassParams()):
            from grass.lib import raster as lr
            import grass.temporal as tgis
            from itzi.grass.interface import GrassInterface
            assert (lr.Rast_window_rows(), lr.Rast_window_cols()) == (rows, 5)
        assert (lr.Rast_window_rows(), lr.Rast_window_cols()) == (rows, 5)
    with GrassSessionManager(GrassParams(str(database), project, "PERMANENT")) as manager:
        job_gisrc = Path(os.environ["GISRC"])
        interface = GrassInterface(start, start + timedelta(seconds=1), np.float32)
        assert interface.yr == rows
        assert manager.current_context()[1] == project
        assert tgis.get_current_location() == project
        assert interface.read_raster_map("owned_gradient@PERMANENT").shape == (rows, 5)
    manager.close()
    assert "GISRC" not in os.environ
    assert not job_gisrc.exists()
""",
            str(tmp_path),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
