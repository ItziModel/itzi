"""Temporary regions must agree across GRASS commands and in-process raster access."""

from __future__ import annotations

import os
from ctypes import byref
from datetime import datetime, timedelta

import grass.script as gscript
import numpy as np
import pytest

from itzi.grass.session import GrassParams, GrassSessionManager


def working_windows() -> tuple[bytes, bytes, bytes]:
    from grass.lib import gis as libgis, raster as libraster

    gis = libgis.Cell_head()
    raster_input = libgis.Cell_head()
    raster_output = libgis.Cell_head()
    libgis.G_get_set_window(byref(gis))
    libraster.Rast_get_input_window(byref(raster_input))
    libraster.Rast_get_output_window(byref(raster_output))
    return bytes(gis), bytes(raster_input), bytes(raster_output)


@pytest.mark.usefixtures("grass_5by5")
@pytest.mark.parametrize("inherited", ["none", "WIND_OVERRIDE", "GRASS_REGION", "both"])
def test_successive_regions_restore_overrides_windows_and_saved_regions(
    monkeypatch: pytest.MonkeyPatch, inherited: str
) -> None:
    from grass.lib import raster as libraster
    from grass.pygrass.gis.region import Region

    from itzi.grass.interface import GrassInterface

    gscript.run_command("g.region", n=40, s=0, e=50, w=0, res=10, save="outer")
    if inherited in ("WIND_OVERRIDE", "both"):
        monkeypatch.setenv("WIND_OVERRIDE", "outer")
    if inherited in ("GRASS_REGION", "both"):
        monkeypatch.setenv("GRASS_REGION", gscript.region_env(region="outer"))
    Region()
    libraster.Rast_unset_window()
    previous_windows = working_windows()
    previous_region = gscript.region()
    previous_overrides = (os.environ.get("WIND_OVERRIDE"), os.environ.get("GRASS_REGION"))
    previous_saved_regions = gscript.list_strings("region")
    gscript.mapcalc("gradient=float(row()*10+col())")

    start = datetime(2020, 1, 1)
    for index, region_id in enumerate(("3by5", None, "3by5")):
        rows = 3 if region_id else int(previous_region["rows"])
        with GrassSessionManager(GrassParams(region=region_id)) as manager:
            interface = GrassInterface(start, start + timedelta(seconds=1), np.float32)
            assert (interface.yr, interface.xr) == (rows, 5)
            assert (interface.get_domain_data().rows, interface.get_domain_data().cols) == (
                rows,
                5,
            )
            assert (gscript.region()["rows"], gscript.region()["cols"]) == (rows, 5)
            values = interface.read_raster_map("gradient@5by5")
            assert values.shape == (rows, 5)
            expected = (
                np.arange(1, 6)
                + np.arange(10, (int(previous_region["rows"]) + 1) * 10, 10)[:, None]
            )
            if region_id:
                expected = expected[1:4] if previous_region["rows"] == 5 else expected[:3]
            np.testing.assert_allclose(values, expected)
            output = f"region_output_{index}@5by5"
            interface.write_raster_map(values, output, "friction", 0)
            info = gscript.raster_info(output)
            assert (int(info["rows"]), int(info["cols"])) == (rows, 5)
            np.testing.assert_array_equal(interface.read_raster_map(output), values)
        manager.close()
        assert gscript.region() == previous_region
        assert working_windows() == previous_windows
        assert (
            os.environ.get("WIND_OVERRIDE"),
            os.environ.get("GRASS_REGION"),
        ) == previous_overrides
        assert gscript.list_strings("region") == previous_saved_regions


@pytest.mark.usefixtures("grass_5by5")
@pytest.mark.parametrize("failure", ["missing_region", "small_region", "missing_mask"])
def test_constructor_failure_restores_region_immediately(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from grass.exceptions import CalledModuleError
    from grass.lib import raster as libraster
    from grass.pygrass.gis.region import Region

    from itzi.ensemble.models import EnsembleError
    from itzi.messenger import FatalError

    gscript.run_command("g.region", n=20, s=0, e=20, w=0, res=10, save="small_region")
    monkeypatch.setenv("WIND_OVERRIDE", "3by5")
    Region()
    libraster.Rast_unset_window()
    previous_windows = working_windows()
    previous_saved_regions = gscript.list_strings("region")
    region = failure if failure != "missing_mask" else "3by5"
    mask = "missing_mask" if failure == "missing_mask" else None
    with pytest.raises((CalledModuleError, FatalError, EnsembleError)):
        GrassSessionManager(GrassParams(region=region, mask=mask)).open()
    assert os.environ["WIND_OVERRIDE"] == "3by5"
    assert working_windows() == previous_windows
    assert gscript.list_strings("region") == previous_saved_regions
    with GrassSessionManager(GrassParams()):
        assert gscript.region()["rows"] == 3


@pytest.mark.usefixtures("grass_5by5")
def test_nested_jobs_restore_the_outer_work_window() -> None:
    from grass.lib import raster as libraster

    from itzi.grass.interface import GrassInterface

    start = datetime(2020, 1, 1)
    gscript.run_command("g.region", n=40, s=10, e=40, w=10, res=10, flags="u", save="3by3")
    with GrassSessionManager(GrassParams(region="3by5", mask="5by3")):
        previous_windows = working_windows()
        previous_region = gscript.region()
        previous_mask = os.environ["GRASS_MASK"]
        with GrassSessionManager(GrassParams(region="3by3", mask="n")):
            interface = GrassInterface(start, start + timedelta(seconds=1), np.float32)
            assert interface.read_raster_map("n@5by5").shape == (3, 3)
            assert (libraster.Rast_window_rows(), libraster.Rast_window_cols()) == (3, 3)
        assert working_windows() == previous_windows
        assert gscript.region() == previous_region
        assert os.environ["GRASS_MASK"] == previous_mask
        with GrassSessionManager(GrassParams()):
            assert working_windows() == previous_windows
            assert os.environ["GRASS_MASK"] == previous_mask


@pytest.mark.usefixtures("grass_5by5")
def test_domain_lookup_reads_the_active_window_without_reloading() -> None:
    from grass.pygrass.raster import RasterRow

    from itzi.grass.utils import read_domain

    with GrassSessionManager(GrassParams(region="3by5")):
        with RasterRow("n", mapset="5by5", mode="r") as source:
            previous_windows = working_windows()
            domain = read_domain()
            assert (domain.rows, domain.cols) == (3, 5)
            assert working_windows() == previous_windows
            np.testing.assert_allclose(np.asarray(source.get_row(0)), 0.05)


@pytest.mark.usefixtures("grass_5by5")
def test_default_region_snapshot_preserves_external_changes() -> None:
    from itzi.grass.interface import GrassInterface

    start = datetime(2020, 1, 1)
    with GrassSessionManager(GrassParams()):
        interface = GrassInterface(start, start + timedelta(seconds=1), np.float32)
        environment = os.environ.copy()
        environment.pop("GRASS_REGION", None)
        environment.pop("WIND_OVERRIDE", None)
        gscript.run_command("g.region", n=40, s=0, e=50, w=0, res=10, env=environment)
        assert gscript.region()["rows"] == 5
        assert interface.read_raster_map("n@5by5").shape == (5, 5)
        with GrassSessionManager(GrassParams()):
            assert gscript.region()["rows"] == 5
    assert gscript.region()["rows"] == 4
    with GrassSessionManager(GrassParams()):
        next_interface = GrassInterface(start, start + timedelta(seconds=1), np.float32)
        assert next_interface.read_raster_map("n@5by5").shape == (4, 5)
