from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    from itzi.grass.utils import MapData


@pytest.fixture(scope="module", autouse=True)
def grass_runtime_env() -> None:
    import grass.script as gscript

    gscript.setup.setup_runtime_env()


class FakeGrassInterface:
    def __init__(self) -> None:
        self.temporal_checks: list[str] = []

    def stds_temporal_sanity(self, strds_id: str) -> bool:
        self.temporal_checks.append(strds_id)
        return True

    def raster_list_from_strds(self, strds_id: str) -> list[MapData]:
        from itzi.grass.utils import MapData

        self.stds_temporal_sanity(strds_id)
        start = datetime(2020, 1, 1)
        return [MapData(strds_id, start, start + timedelta(hours=1))]


def test_resolved_input_kinds_share_raster_and_strds_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from itzi.grass import utils
    from itzi.grass.interface import GrassInterface
    from itzi.providers.grass_input import GrassRasterInputProvider
    from itzi.grass.utils import MapData

    monkeypatch.setattr(utils, "name_is_stds", lambda map_id: map_id == "series@test")
    monkeypatch.setattr(utils, "name_is_map", lambda map_id: map_id == "raster@test")

    start = datetime(2020, 1, 1)
    end = start + timedelta(hours=1)
    interface = FakeGrassInterface()

    provider = GrassRasterInputProvider(
        cast(GrassInterface, interface),
        {"ground_elevation": "raster@test", "rainfall_rate": "series@test"},
        start,
        end,
        {"ground_elevation": "raster", "rainfall_rate": "strds"},
    )

    assert provider.map_lists["ground_elevation"] == [MapData("raster@test", start, end)]
    assert provider.map_lists["rainfall_rate"] == [MapData("series@test", start, end)]
    assert interface.temporal_checks == ["series@test"]


@pytest.mark.forked
def test_visible_input_and_current_output_with_same_name(grass_xy_session) -> None:
    import grass.script as gscript

    from itzi.grass.utils import (
        output_name,
        qualify_output_id,
        write_raster_map_blocking,
    )
    from itzi.providers.grass_output import derived_record_name

    gscript.run_command("g.region", n=3, s=0, e=3, w=0, res=1)
    gscript.mapcalc("visible_output=1")
    gscript.run_command("g.mapset", mapset="child", flags="c")
    output_id = qualify_output_id("visible_output", "child")
    assert output_id == "visible_output@child"
    assert derived_record_name(output_id, 0) == "visible_output_0000@child"
    assert output_name(output_id) == "visible_output"
    with pytest.raises(ValueError, match="current mapset"):
        qualify_output_id("visible_output@PERMANENT", "child")

    gscript.run_command("g.region", n=3, s=0, e=3, w=0, res=1)
    write_raster_map_blocking(
        np.ones((3, 3), dtype=np.int32), "visible_output", "CELL", "ground_elevation", 0, False
    )


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
def test_direct_runner_qualifies_ids_before_creating_providers(test_data_path: str) -> None:
    from pathlib import Path

    from itzi.configreader import ConfigReader
    from itzi.grass.session import GrassSessionManager
    from itzi.simulation_runner import SimulationRunner

    GrassSessionManager.ensure_temporal_initialized()
    reader = ConfigReader(str(Path(test_data_path) / "5by5" / "5by5.ini"))
    config = reader.sim_config.model_copy(
        update={"output_map_names": {"water_depth": "direct_id_check"}}
    )
    runner = SimulationRunner(config, reader.grass_params)

    assert runner.sim_config.input_map_names["ground_elevation"] == "z@5by5"
    assert runner.sim_config.output_map_names["water_depth"] == "direct_id_check@5by5"


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
def test_vector_writer_uses_bare_name_from_qualified_output_id() -> None:
    from grass.pygrass.utils import get_mapset_vector
    from itzi_core.data_containers import DrainageNetworkAttributes, DrainageNetworkTopology

    from itzi.grass.interface import GrassInterface

    start = datetime(2020, 1, 1)
    with GrassInterface(start, start + timedelta(seconds=1), np.float32, None, None) as interface:
        interface.write_vector_map(
            DrainageNetworkTopology(nodes=(), links=()),
            DrainageNetworkAttributes(nodes=(), links=()),
            "vector_id_check@5by5",
        )
    assert get_mapset_vector("vector_id_check", "5by5") == "5by5"
