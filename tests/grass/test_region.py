"""Temporary regions must agree across GRASS commands and in-process raster access."""

from __future__ import annotations

import copy
import os
from ctypes import byref
from datetime import datetime, timedelta

import grass.script as gscript
import numpy as np
import pytest


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

    start = datetime(2020, 1, 1)
    for index, region_id in enumerate(("3by5", None, "3by5")):
        rows = 3 if region_id else int(previous_region["rows"])
        with GrassInterface(
            start, start + timedelta(seconds=1), np.float32, region_id, None
        ) as interface:
            assert (interface.yr, interface.xr) == (rows, 5)
            assert (interface.get_domain_data().rows, interface.get_domain_data().cols) == (
                rows,
                5,
            )
            assert (gscript.region()["rows"], gscript.region()["cols"]) == (rows, 5)
            values = interface.read_raster_map("n@5by5")
            assert values.shape == (rows, 5)
            np.testing.assert_allclose(values, 0.05)
            output = f"region_output_{index}@5by5"
            interface.write_raster_map(values, output, "friction", 0)
            info = gscript.raster_info(output)
            assert (int(info["rows"]), int(info["cols"])) == (rows, 5)
        interface.cleanup()
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
    from itzi.grass.interface import GrassInterface
    from itzi.messenger import FatalError

    gscript.run_command("g.region", n=20, s=0, e=20, w=0, res=10, save="small_region")
    monkeypatch.setenv("WIND_OVERRIDE", "3by5")
    Region()
    libraster.Rast_unset_window()
    previous_windows = working_windows()
    previous_saved_regions = gscript.list_strings("region")
    start = datetime(2020, 1, 1)
    region = failure if failure != "missing_mask" else "3by5"
    mask = "missing_mask" if failure == "missing_mask" else None
    with pytest.raises((CalledModuleError, FatalError, EnsembleError)):
        GrassInterface(start, start + timedelta(seconds=1), np.float32, region, mask)
    assert os.environ["WIND_OVERRIDE"] == "3by5"
    assert working_windows() == previous_windows
    assert gscript.list_strings("region") == previous_saved_regions


@pytest.mark.usefixtures("grass_5by5")
def test_custom_gis_and_split_raster_windows_are_restored() -> None:
    from grass.lib import raster as libraster
    from grass.pygrass.gis.region import Region

    from itzi.grass.interface import GrassInterface

    configured = Region()
    libraster.Rast_unset_window()
    gis = copy.deepcopy(configured)
    gis.north = 40
    gis.adjust()
    raster_input = copy.deepcopy(configured)
    raster_input.north = 30
    raster_input.adjust()
    raster_output = copy.deepcopy(configured)
    raster_output.east = 40
    raster_output.adjust()
    try:
        raster_input.set_current()
        libraster.Rast_unset_window()
        libraster.Rast_get_input_window(raster_input.byref())
        libraster.Rast_set_output_window(raster_output.byref())
        gis.set_current()
        previous_windows = working_windows()
        start = datetime(2020, 1, 1)
        with GrassInterface(
            start, start + timedelta(seconds=1), np.float32, "3by5", None
        ) as interface:
            assert interface.read_raster_map("n@5by5").shape == (3, 5)
        assert working_windows() == previous_windows
    finally:
        configured.set_current()
        libraster.Rast_unset_window()


@pytest.mark.usefixtures("grass_5by5")
def test_domain_lookup_preserves_working_windows_and_open_raster() -> None:
    from grass.lib import raster as libraster
    from grass.pygrass.gis.region import Region
    from grass.pygrass.raster import RasterRow

    from itzi.grass.utils import read_domain

    configured = Region()
    libraster.Rast_unset_window()
    libraster.Rast_get_input_window(configured.byref())
    gis = copy.deepcopy(configured)
    gis.north = 40
    gis.adjust()
    gis.set_current()
    try:
        with RasterRow("n", mapset="5by5", mode="r") as source:
            previous_windows = working_windows()
            domain = read_domain("3by5")
            assert (domain.rows, domain.cols) == (3, 5)
            assert working_windows() == previous_windows
            np.testing.assert_allclose(np.asarray(source.get_row(0)), 0.05)
    finally:
        configured.set_current()
