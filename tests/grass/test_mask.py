"""Configured and inherited masks must agree with GRASS raster reads."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import grass.script as gscript
import numpy as np
import pytest

from itzi.grass.session import GrassParams, GrassSessionManager

if TYPE_CHECKING:
    from itzi_core import Simulation


@pytest.mark.usefixtures("grass_5by5")
@pytest.mark.parametrize("selection", ["explicit", "MASK", "GRASS_MASK"])
@pytest.mark.parametrize(
    ("expression", "quant_rules", "excluded"),
    [
        (
            "if(col()==1, null(), if(col()==2, 0, if(col()==3, -2, 1)))",
            None,
            (True, True, False, False, False),
        ),
        (
            "float(if(col()==1, null(), if(col()==2, 0.25, "
            "if(col()==3, -0.25, if(col()==4, 0.75, -0.75)))))",
            None,
            (True, True, True, False, False),
        ),
        (
            "if(col()==1, null(), if(col()==2, 0.25, "
            "if(col()==3, -0.25, if(col()==4, 0.75, -0.75))))",
            None,
            (True, True, True, False, False),
        ),
        (
            "if(col()==1, null(), if(col()==2, 0.75, "
            "if(col()==3, 1.25, if(col()==4, 1.75, 0.0))))",
            "0:1:0\n1:2:1\n",
            (True, True, False, False, True),
        ),
    ],
    ids=["CELL", "FCELL", "DCELL", "custom-quantisation"],
)
def test_mask_selection_has_identical_rules_and_preserves_rasters(
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
    expression: str,
    quant_rules: str | None,
    excluded: tuple[bool, ...],
) -> None:
    from itzi.ensemble.models import EffectiveMask
    from itzi.grass.interface import GrassInterface
    from itzi.grass.utils import resolve_effective_mask

    gscript.mapcalc(f"mask_values={expression}")
    if quant_rules is not None:
        gscript.write_command("r.quant", input="mask_values", rules="-", stdin=quant_rules)
    if selection == "MASK":
        gscript.run_command("g.copy", raster="mask_values,MASK")
        monkeypatch.delenv("GRASS_MASK", raising=False)
        source = "MASK@5by5"
    else:
        gscript.mapcalc("MASK=if(col()==5, 1, 0)")
        source = "mask_values@5by5"
        monkeypatch.setenv("GRASS_MASK", source if selection == "GRASS_MASK" else "MASK@5by5")

    original_mask = os.environ.get("GRASS_MASK")
    original_rasters = gscript.list_strings("raster")
    expected = np.tile(excluded, (5, 1))
    start = datetime(2020, 1, 1)
    with GrassSessionManager(
        GrassParams(mask=source if selection == "explicit" else None)
    ) as manager:
        interface = GrassInterface(start, start + timedelta(seconds=1), np.float32)
        assert manager.effective_mask.mode == ("explicit" if selection == "explicit" else "active")
        assert os.environ["GRASS_MASK"] == source
        assert resolve_effective_mask(None) == EffectiveMask("active", source)
        np.testing.assert_array_equal(interface.get_npmask(), expected)
        friction = interface.read_raster_map("n@5by5")
        np.testing.assert_array_equal(np.isnan(friction), expected)
        np.testing.assert_allclose(friction[~expected], 0.05)
        np.testing.assert_array_equal(np.isnan(interface.read_raster_map(source)), expected)
        assert os.environ["GRASS_MASK"] == source
        np.testing.assert_array_equal(interface.get_npmask(), expected)
        stats = gscript.parse_command("r.univar", map="n", flags="g")
        assert int(stats["n"]) == np.count_nonzero(~expected)

    assert os.environ.get("GRASS_MASK") == original_mask
    manager.close()
    assert os.environ.get("GRASS_MASK") == original_mask
    assert gscript.list_strings("raster") == original_rasters


@pytest.mark.usefixtures("grass_5by5")
def test_missing_grass_mask_does_not_fall_back_to_mapset_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from itzi.ensemble.models import EffectiveMask
    from itzi.grass.interface import GrassInterface
    from itzi.grass.utils import resolve_effective_mask

    gscript.mapcalc("MASK=0")
    monkeypatch.setenv("GRASS_MASK", "missing_mask")
    assert resolve_effective_mask(None) == EffectiveMask("none", None)
    start = datetime(2020, 1, 1)
    with GrassSessionManager(GrassParams()):
        interface = GrassInterface(start, start + timedelta(seconds=1), np.float32)
        assert not interface.get_npmask().any()
        np.testing.assert_allclose(interface.read_raster_map("n@5by5"), 0.05)
    assert os.environ["GRASS_MASK"] == "missing_mask"


@pytest.mark.usefixtures("grass_5by5")
def test_qualified_mask_in_another_mapset_and_exception_restoration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from grass.exceptions import OpenError
    from grass.pygrass.gis import set_current_mapset

    from itzi.ensemble.models import EffectiveMask
    from itzi.grass.interface import GrassInterface
    from itzi.grass.utils import resolve_effective_mask

    set_current_mapset("PERMANENT")
    gscript.run_command("g.region", n=50, s=0, e=50, w=0, res=10)
    gscript.mapcalc("shared_mask=if(col()==1, 0, 1)", overwrite=True)
    set_current_mapset("5by5")
    monkeypatch.setenv("GRASS_MASK", "shared_mask@PERMANENT")
    assert resolve_effective_mask(None) == EffectiveMask("active", "shared_mask@PERMANENT")

    start = datetime(2020, 1, 1)
    with pytest.raises(OpenError):
        with GrassSessionManager(GrassParams(mask="5by3")):
            interface = GrassInterface(start, start + timedelta(seconds=1), np.float32)
            assert os.environ["GRASS_MASK"] == "5by3@5by5"
            with pytest.raises(OpenError):
                interface.read_raster_map("missing_input@5by5")
            assert os.environ["GRASS_MASK"] == "5by3@5by5"
            interface.read_raster_map("missing_input@5by5")
    assert os.environ["GRASS_MASK"] == "shared_mask@PERMANENT"

    with GrassSessionManager(GrassParams()):
        interface = GrassInterface(start, start + timedelta(seconds=1), np.float32)
        mask = interface.get_npmask()
        assert mask[:, 0].all()
        assert not mask[:, 1:].any()


@pytest.mark.usefixtures("grass_5by5")
@pytest.mark.parametrize("phase", ["initialize", "finalize"])
def test_runner_restores_mask_when_lifecycle_fails(
    monkeypatch: pytest.MonkeyPatch,
    test_data_path: str,
    phase: str,
) -> None:
    from grass.lib import raster as libraster

    from itzi.configreader import ConfigReader
    from itzi.grass.session import GrassParams
    from itzi.simulation_runner import SimulationRunner

    gscript.mapcalc("mask_values=if(col()==1, 0, 1)")
    monkeypatch.setenv("GRASS_MASK", "previous_mask")
    previous_region = gscript.region()
    previous_shape = (libraster.Rast_window_rows(), libraster.Rast_window_cols())
    reader = ConfigReader(str(Path(test_data_path) / "5by5" / "5by5.ini"))
    config = reader.sim_config.model_copy(update={"output_map_names": {}})
    if phase == "initialize":
        gscript.mapcalc("bad_dem=null()")
        config = config.model_copy(
            update={
                "input_map_names": dict(config.input_map_names) | {"ground_elevation": "bad_dem"}
            }
        )
    with pytest.raises(
        RuntimeError,
        match="only NULL/NaN cells" if phase == "initialize" else "finalization failed",
    ):
        with GrassSessionManager(GrassParams(region="3by5", mask="mask_values")):
            runner = SimulationRunner(config).initialize()
            assert os.environ["GRASS_MASK"] == "mask_values@5by5"

            def fail_finalize(self: Simulation) -> None:
                raise RuntimeError("finalization failed")

            monkeypatch.setattr(type(runner.sim), "finalize", fail_finalize)
            runner.finalize()
    assert os.environ["GRASS_MASK"] == "previous_mask"
    assert gscript.region() == previous_region
    assert "WIND_OVERRIDE" not in os.environ
    assert (libraster.Rast_window_rows(), libraster.Rast_window_cols()) == previous_shape


@pytest.mark.usefixtures("grass_5by5")
@pytest.mark.parametrize("explicit_mask", [False, True])
@pytest.mark.parametrize("previous_mask", [False, True])
@pytest.mark.parametrize("region", [None, "3by3"])
def test_work_context_restores_fresh_raster_reads(
    monkeypatch: pytest.MonkeyPatch,
    test_data_path: str,
    explicit_mask: bool,
    previous_mask: bool,
    region: str | None,
) -> None:
    gscript.mapcalc("gradient=float(row()*10+col())")
    gscript.run_command("g.region", n=40, s=10, e=40, w=10, res=10, save="3by3")
    gscript.mapcalc("previous_mask=if(col()==1, 0, 1)")
    if previous_mask:
        monkeypatch.setenv("GRASS_MASK", "previous_mask@5by5")
    else:
        monkeypatch.delenv("GRASS_MASK", raising=False)

    # A GRASS C-level exit must fail this test, not terminate the pytest process.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
import sys
from pathlib import Path

import numpy as np
import grass.script as gscript
from grass.lib import raster as libraster
from grass.pygrass.raster import RasterRow

from itzi.configreader import ConfigReader
from itzi.grass.session import GrassParams, GrassSessionManager
from itzi.simulation_runner import SimulationRunner

reader = ConfigReader(str(Path(sys.argv[1]) / "5by5" / "5by5.ini"))
config = reader.sim_config.model_copy(update={"output_map_names": {}})
original_mask = os.environ.get("GRASS_MASK")
mask = "5by3" if sys.argv[2] == "True" else None
region = sys.argv[3] or None
original_region = gscript.region()
with GrassSessionManager(GrassParams(region=region, mask=mask)):
    runner = SimulationRunner(config).initialize()
    values = runner.g_interface.read_raster_map("gradient@5by5")
    expected = (np.arange(20, 50, 10)[:, None] + np.arange(2, 5) if region
                else np.arange(10, 60, 10)[:, None] + np.arange(1, 6)).astype(np.float32)
    excluded = runner.g_interface.get_npmask()
    expected[excluded] = np.nan
    np.testing.assert_array_equal(values, expected)
    runner.g_interface.write_raster_map(values, "roundtrip@5by5", "friction", 0)
    np.testing.assert_array_equal(runner.g_interface.read_raster_map("roundtrip@5by5"), expected)
    runner.finalize()
assert os.environ.get("GRASS_MASK") == original_mask
assert gscript.region() == original_region
assert (libraster.Rast_window_rows(), libraster.Rast_window_cols()) == (5, 5)
with RasterRow("gradient", mapset="5by5", mode="r") as source:
    restored_row = np.asarray(source.get_row(0))
    np.testing.assert_array_equal(np.isnan(restored_row), [original_mask is not None, False, False, False, False])
    np.testing.assert_allclose(restored_row[1:], np.arange(12, 16))
with GrassSessionManager(GrassParams(region="3by5", mask=mask)):
    next_runner = SimulationRunner(config).initialize()
    assert next_runner.g_interface.read_raster_map("n@5by5").shape == (3, 5)
    next_runner.finalize()
assert os.environ.get("GRASS_MASK") == original_mask
""",
            test_data_path,
            str(explicit_mask),
            region or "",
        ],
        env=os.environ | {"PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
