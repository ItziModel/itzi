from datetime import datetime, timedelta
from typing import Literal

import pytest

from itzi.providers.grass_input import GrassRasterInputProvider
from itzi.providers.grass_interface import MapData


class FakeGrassInterface:
    def __init__(self) -> None:
        self.temporal_checks: list[str] = []

    @staticmethod
    def format_id(name: str) -> str:
        return f"{name}@test"

    @staticmethod
    def name_is_stds(map_id: str, *, initialize: bool = True) -> bool:
        return map_id == "series@test"

    @staticmethod
    def name_is_map(map_id: str) -> bool:
        return map_id == "raster@test"

    def stds_temporal_sanity(self, strds_id: str) -> bool:
        self.temporal_checks.append(strds_id)
        return True

    def raster_list_from_strds(self, strds_id: str) -> list[MapData]:
        self.stds_temporal_sanity(strds_id)
        start = datetime(2020, 1, 1)
        return [MapData(strds_id, start, start + timedelta(hours=1))]


@pytest.mark.parametrize(
    "input_kinds", [None, {"ground_elevation": "raster", "rainfall_rate": "strds"}]
)
def test_input_kinds_share_raster_and_strds_construction(
    input_kinds: dict[str, Literal["raster", "strds"]] | None,
) -> None:
    start = datetime(2020, 1, 1)
    end = start + timedelta(hours=1)
    interface = FakeGrassInterface()

    provider = GrassRasterInputProvider(
        interface,
        {"ground_elevation": "raster", "rainfall_rate": "series"},
        start,
        end,
        input_kinds,
    )

    assert provider.map_lists["ground_elevation"] == [MapData("raster@test", start, end)]
    assert provider.map_lists["rainfall_rate"] == [MapData("series@test", start, end)]
    assert interface.temporal_checks == ["series@test"]
