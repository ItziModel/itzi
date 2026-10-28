"""Configured and inherited masks must agree with GRASS raster reads."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import grass.script as gscript
import numpy as np
import pytest

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
    with GrassInterface(
        start,
        start + timedelta(seconds=1),
        np.float32,
        None,
        source if selection == "explicit" else None,
    ) as interface:
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
    interface.cleanup()
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
    with GrassInterface(start, start + timedelta(seconds=1), np.float32, None, None) as interface:
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
        with GrassInterface(
            start, start + timedelta(seconds=1), np.float32, None, "5by3"
        ) as interface:
            assert os.environ["GRASS_MASK"] == "5by3@5by5"
            with pytest.raises(OpenError):
                interface.read_raster_map("missing_input@5by5")
            assert os.environ["GRASS_MASK"] == "5by3@5by5"
            interface.read_raster_map("missing_input@5by5")
    assert os.environ["GRASS_MASK"] == "shared_mask@PERMANENT"

    with GrassInterface(start, start + timedelta(seconds=1), np.float32, None, None) as interface:
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
    from itzi.configreader import ConfigReader
    from itzi.grass.session import GrassParams
    from itzi.simulation_runner import SimulationRunner

    gscript.mapcalc("mask_values=if(col()==1, 0, 1)")
    monkeypatch.setenv("GRASS_MASK", "previous_mask")
    reader = ConfigReader(str(Path(test_data_path) / "5by5" / "5by5.ini"))
    config = reader.sim_config.model_copy(update={"output_map_names": {}})
    if phase == "initialize":
        gscript.mapcalc("bad_dem=null()")
        config = config.model_copy(
            update={
                "input_map_names": dict(config.input_map_names) | {"ground_elevation": "bad_dem"}
            }
        )
    runner = SimulationRunner(config, GrassParams(mask="mask_values"))
    if phase == "initialize":
        with pytest.raises(RuntimeError, match="only NULL/NaN cells"):
            runner.initialize()
    else:
        runner.initialize()
        assert os.environ["GRASS_MASK"] == "mask_values@5by5"

        def fail_finalize(self: Simulation) -> None:
            raise RuntimeError("finalization failed")

        monkeypatch.setattr(type(runner.sim), "finalize", fail_finalize)
        with pytest.raises(RuntimeError, match="finalization failed"):
            runner.finalize()
    assert os.environ["GRASS_MASK"] == "previous_mask"
