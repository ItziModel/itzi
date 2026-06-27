"""
Copyright (C) 2026 Laurent Courty

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

from datetime import datetime
from typing import Mapping, NotRequired, Sequence, TypedDict

import numpy as np
from pydantic import BaseModel, ConfigDict

from itzi.array_definitions import ARRAY_DEFINITIONS, ArrayCategory
from itzi.providers.base import RasterInputProvider
from itzi.providers.domain_data import DomainData


class TimedRasterSlice(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    start_time: datetime
    end_time: datetime
    array: np.ndarray


class MemoryRasterInputConfig(TypedDict):
    domain_data: DomainData
    simulation_start_time: datetime
    simulation_end_time: datetime
    static_arrays: NotRequired[Mapping[str, np.ndarray]]
    timed_arrays: NotRequired[Mapping[str, Sequence[TimedRasterSlice]]]


VALID_INPUT_KEYS: frozenset[str] = frozenset(
    arr_def.key for arr_def in ARRAY_DEFINITIONS if ArrayCategory.INPUT in arr_def.category
)


class MemoryRasterInputProvider(RasterInputProvider):
    """Provide raster inputs directly from in-memory numpy arrays."""

    def __init__(self, config: MemoryRasterInputConfig) -> None:
        self.domain_data = config["domain_data"]
        self.start_time = config["simulation_start_time"]
        self.end_time = config["simulation_end_time"]

        if self.start_time >= self.end_time:
            raise ValueError("simulation_start_time must be inferior to simulation_end_time")

        static_arrays = config.get("static_arrays", {})
        timed_arrays = config.get("timed_arrays", {})

        self._validate_input_keys(static_arrays, config_name="static_arrays")
        self._validate_input_keys(timed_arrays, config_name="timed_arrays")

        duplicate_keys = sorted(set(static_arrays.keys()) & set(timed_arrays.keys()))
        if duplicate_keys:
            raise ValueError(
                "static_arrays and timed_arrays contain duplicate keys: "
                f"{', '.join(duplicate_keys)}"
            )

        self.static_arrays: dict[str, np.ndarray] = {
            key: self._copy_and_validate_array(key, array) for key, array in static_arrays.items()
        }
        self.timed_arrays: dict[str, tuple[TimedRasterSlice, ...]] = {
            key: self._normalize_timed_slices(key, slices) for key, slices in timed_arrays.items()
        }

    def _validate_input_keys(self, arrays: Mapping[str, object], *, config_name: str) -> None:
        invalid_keys = sorted(set(arrays.keys()) - VALID_INPUT_KEYS)
        if invalid_keys:
            raise ValueError(f"invalid array keys in {config_name}: {', '.join(invalid_keys)}")

    def _copy_and_validate_array(self, map_key: str, arr: np.ndarray) -> np.ndarray:
        array = np.array(arr, copy=True)
        if array.ndim != 2:
            raise ValueError(
                f"input array '{map_key}' must be two-dimensional, got {array.ndim} dimensions"
            )
        if array.shape != self.domain_data.shape:
            raise ValueError(
                f"input array '{map_key}' shape {array.shape} does not match domain shape "
                f"{self.domain_data.shape}"
            )
        return array

    def _normalize_timed_slices(
        self,
        map_key: str,
        slices: Sequence[TimedRasterSlice],
    ) -> tuple[TimedRasterSlice, ...]:
        normalized_slices: list[TimedRasterSlice] = []
        previous_slice: TimedRasterSlice | None = None

        for timed_slice in slices:
            if timed_slice.start_time > timed_slice.end_time:
                raise ValueError(
                    f"timed slice for '{map_key}' has start_time after end_time: "
                    f"{timed_slice.start_time} > {timed_slice.end_time}"
                )

            if previous_slice is not None and timed_slice.start_time < previous_slice.start_time:
                raise ValueError(f"timed slices for '{map_key}' must be sorted by start_time")

            if previous_slice is not None and timed_slice.start_time <= previous_slice.end_time:
                raise ValueError(f"timed slices for '{map_key}' must not overlap")

            normalized_slice = TimedRasterSlice(
                start_time=timed_slice.start_time,
                end_time=timed_slice.end_time,
                array=self._copy_and_validate_array(map_key, timed_slice.array),
            )
            normalized_slices.append(normalized_slice)
            previous_slice = normalized_slice

        return tuple(normalized_slices)

    def get_domain_data(self) -> DomainData:
        """Return a DomainData object."""
        return self.domain_data

    def get_array(
        self, map_key: str, current_time: datetime
    ) -> tuple[np.ndarray | None, datetime, datetime]:
        """Return the array and its validity interval for a given key and time."""
        static_array: np.ndarray | None = self.static_arrays.get(map_key)
        if static_array is not None:
            return static_array.copy(), self.start_time, self.end_time

        for timed_slice in self.timed_arrays.get(map_key, ()):
            if timed_slice.start_time <= current_time <= timed_slice.end_time:
                return timed_slice.array.copy(), timed_slice.start_time, timed_slice.end_time

        return None, self.start_time, self.end_time
