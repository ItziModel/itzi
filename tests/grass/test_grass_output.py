from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import grass.script as gscript
import numpy as np
import pytest

from itzi.grass.session import GrassParams, GrassSessionManager


@pytest.mark.usefixtures("grass_5by5")
@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_native_raster_colors_match_r_colors(
    monkeypatch: pytest.MonkeyPatch, dtype: type[np.generic]
) -> None:
    from grass.pygrass import utils as gutils

    from itzi.grass.interface import GrassInterface, colors_rules_dict

    start = datetime(2020, 1, 1)
    values = np.linspace(-4, 400, 25, dtype=dtype).reshape(5, 5)
    if np.issubdtype(dtype, np.floating):
        values[0, 0] = np.nan
    original = values.copy()

    with GrassSessionManager(GrassParams()):
        interface = GrassInterface(start, start + timedelta(seconds=1), dtype)
        mapset = gutils.getenv("MAPSET")
        color_directory = (
            Path(gutils.getenv("GISDBASE")) / gutils.getenv("LOCATION_NAME") / mapset / "colr"
        )
        reference = f"reference_colors@{mapset}"
        interface.write_raster_map(values, reference, "ground_elevation", 0)

        for map_key, rules in colors_rules_dict.items():
            name = f"native_{map_key}"
            with monkeypatch.context() as native:
                native.setattr(
                    gscript,
                    "run_command",
                    lambda *_args, **_kwargs: pytest.fail("Native color writing launched a tool"),
                )
                interface.write_raster_map(values, f"{name}@{mapset}", map_key, 0)
            gscript.run_command("r.colors", map=reference, rules=rules, quiet=True)
            assert (color_directory / name).read_bytes() == (
                color_directory / "reference_colors"
            ).read_bytes(), map_key
            np.testing.assert_array_equal(interface.read_raster_map(f"{name}@{mapset}"), values)

    np.testing.assert_array_equal(values, original)


@pytest.mark.usefixtures("grass_5by5")
def test_native_raster_colors_report_missing_rules(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from itzi.grass.interface import GrassInterface, colors_rules_dict

    missing_rules = tmp_path / "missing_colors.txt"
    monkeypatch.setitem(colors_rules_dict, "flow_speed", str(missing_rules))
    start = datetime(2020, 1, 1)
    with GrassSessionManager(GrassParams()):
        interface = GrassInterface(start, start + timedelta(seconds=1), np.float32)
        with pytest.raises(RuntimeError, match="Unable to load color rules") as error:
            interface.write_raster_map(
                np.ones((5, 5), dtype=np.float32), "missing_colors@5by5", "flow_speed", 0
            )
        assert str(missing_rules) in str(error.value)
