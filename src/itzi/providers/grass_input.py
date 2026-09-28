"""
Copyright (C) 2025-2026 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

import numpy as np
from itzi_core import INPUT_ARRAY_KEYS, DomainData
from itzi_core.providers import RasterInputProvider

import itzi.messenger as msgr
from itzi.providers.grass_interface import MapData

if TYPE_CHECKING:
    from datetime import datetime

    from itzi.providers.grass_interface import GrassInterface


class GrassRasterInputProvider(RasterInputProvider):
    def __init__(
        self,
        grass_interface: GrassInterface,
        input_map_names: Mapping[str, str | None],
        default_start_time: datetime,
        default_end_time: datetime,
        input_kinds: Mapping[str, Literal["raster", "strds"]] | None,
    ) -> None:
        self.grass_interface = grass_interface
        self.start_time = default_start_time
        self.end_time = default_end_time
        self.map_lists = resolve_input_map_lists(
            grass_interface,
            input_map_names,
            default_start_time,
            default_end_time,
            input_kinds,
        )

    def get_domain_data(self) -> DomainData:
        """Return a DomainData object"""
        return DomainData(
            north=self.grass_interface.region.north,
            south=self.grass_interface.region.south,
            east=self.grass_interface.region.east,
            west=self.grass_interface.region.west,
            rows=self.grass_interface.region.rows,
            cols=self.grass_interface.region.cols,
            crs_wkt=self.grass_interface.get_crs_wkt(),
        )

    def get_array(
        self, map_key: str, current_time: datetime
    ) -> tuple[np.ndarray | None, datetime, datetime]:
        """Return the array and its half-open validity window for a given time.
        The input series is expected to cover the simulation timeline continuously.
        Reaching the final `else` therefore means the map set was
        modified after validation or the provider logic became inconsistent.
        """
        if map_key not in self.map_lists:
            raise ValueError(f"Unknown map key: {map_key}")
        map_list = self.map_lists[map_key]
        if map_list is None:
            return None, self.start_time, self.end_time
        for map_name in map_list:
            map_start: datetime = max(self.start_time, map_name.start_time)
            map_end: datetime = min(self.end_time, map_name.end_time)
            if map_start <= current_time < map_end:
                arr = self.grass_interface.read_raster_map(map_name.id)
                return arr, map_start, map_end

        # No gap is expected here: GRASS temporal sanity is checked at init.
        # An in-range lookup should always hit one map.
        raise ValueError(f"No map found for {map_key} at time {current_time}")


def resolve_input_map_lists(
    grass_interface: GrassInterface,
    map_names: Mapping[str, str | None],
    start_time: datetime,
    end_time: datetime,
    input_kinds: Mapping[str, Literal["raster", "strds"]] | None,
) -> Mapping[str, list[MapData] | None]:
    """Resolve provider-ready raster lists without creating an output provider."""

    invalid_input_keys = sorted(set(map_names) - INPUT_ARRAY_KEYS)
    if invalid_input_keys:
        raise ValueError(f"Invalid input keys found: {', '.join(invalid_input_keys)}")

    map_lists: dict[str, list[MapData] | None] = {arr_key: None for arr_key in INPUT_ARRAY_KEYS}
    for key, map_name in map_names.items():
        if not map_name:
            continue
        map_id = grass_interface.format_id(map_name)
        kind = input_kinds.get(key) if input_kinds is not None else None
        if kind == "strds":
            if not grass_interface.name_is_stds(map_id):
                msgr.fatal(f"STRDS input <{map_id}> is no longer available")
        elif kind == "raster":
            if not grass_interface.name_is_map(map_id):
                msgr.fatal(f"raster input <{map_id}> is no longer available")
        elif grass_interface.name_is_stds(map_id):
            kind = "strds"
        elif grass_interface.name_is_map(map_id):
            kind = "raster"
        else:
            msgr.fatal(f"{map_name} not found!")
        if kind == "strds":
            map_list = grass_interface.raster_list_from_strds(map_id)
        else:
            map_list = [MapData(id=map_id, start_time=start_time, end_time=end_time)]
        map_lists[key] = map_list
    return map_lists
