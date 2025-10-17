# coding=utf8
"""
Copyright (C) 2025 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

from typing import Tuple, Mapping, TypedDict, NotRequired, Union, TYPE_CHECKING

import numpy as np

try:
    import xarray as xr
    import icechunk
    import pandas as pd
except ImportError:
    raise ImportError(
        "To use the Icechunk backend, install itzi with: "
        "'uv tool install itzi[cloud]' "
        "or 'pip install itzi[cloud]'"
    )

from itzi.providers.base import RasterInputProvider
from itzi.providers.domain_data import DomainData
from itzi.const import TemporalType

if TYPE_CHECKING:
    from datetime import datetime


class IcechunkRasterInputConfig(TypedDict):
    icechunk_storage: icechunk.Storage
    icechunk_group: str
    # A dict of key: input names
    input_map_names: Mapping[str, str]
    # A Mapping of dimensions names
    # {"time": "start_time", "x": "longitude"}
    dimension_names: NotRequired[Mapping[str, str]]
    simulation_start_time: "datetime"
    simulation_end_time: "datetime"


class IcechunkRasterInputProvider(RasterInputProvider):
    """Abstract base class for handling raster simulation inputs."""

    def __init__(self, config: IcechunkRasterInputConfig) -> None:
        self.sim_start_time = config["simulation_start_time"]
        self.sim_end_time = config["simulation_end_time"]
        self.input_map_names = config["input_map_names"]
        # Set default if dimension_names are not given
        try:
            dim_names = config["dimension_names"]
        except KeyError:
            dim_names = {k: k for k in ["x", "y", "time"]}
        self.x_dim = dim_names["x"]
        self.y_dim = dim_names["y"]
        self.time_dim = dim_names["time"]
        # Open dataset
        repo = icechunk.Repository.open(config["icechunk_storage"])
        session = repo.readonly_session(config["icechunk_group"])
        self.dataset = xr.open_zarr(session.store)
        try:
            self.crs_wkt = self.dataset.attrs["crs_wkt"]
        except KeyError:
            self.crs_wkt = ""

        if not self.is_dataset_sorted():
            raise ValueError("Coordinates must be sorted")
        if not self.is_equal_spacing():
            raise ValueError("Spatial coordinates must be equally spaced")
        self.temporal_type = self.detect_temporal_type()
        self.origin = self.get_origin()

    def is_equal_spacing(self) -> bool:
        """
        Check if spatial coordinates are equally spaced.
        """
        is_equal = []
        for dim_name in [self.x_dim, self.y_dim]:
            coord = self.dataset[dim_name]
            diffs = np.diff(coord.values if hasattr(coord, "values") else coord)
            if len(diffs) == 0:
                return True, None
            is_equal.append(np.allclose(diffs, diffs[0]))
        return all(is_equal)

    def detect_temporal_type(self) -> str:
        """Detect if time coordinates are relative (timedelta) or absolute (datetime)."""
        try:
            time_coords = self.dataset[self.time_dim]
            # Check the dtype of time coordinates
            if np.issubdtype(time_coords.dtype, np.timedelta64):
                return TemporalType.RELATIVE
            elif np.issubdtype(time_coords.dtype, np.datetime64):
                return TemporalType.ABSOLUTE
            else:
                raise ValueError(f"Unsupported temporal type: {time_coords.dtype}")
        except KeyError:
            # No time dimension
            return TemporalType.ABSOLUTE

    def get_domain_data(self) -> DomainData:
        """Return a DomainData object."""
        # Coordinates are at the center of the cells
        y_coords = self.dataset[self.y_dim]
        rows = len(y_coords)
        northernmost_cell = y_coords.values.max()
        southernmost_cell = y_coords.values.min()
        nsres = (northernmost_cell - southernmost_cell) / (rows - 1)
        north = northernmost_cell + nsres / 2
        south = southernmost_cell - nsres / 2

        x_coords = self.dataset[self.x_dim]
        cols = len(x_coords)
        easternmost_cell = x_coords.values.max()
        westernmost_cell = x_coords.values.min()
        ewres = (easternmost_cell - westernmost_cell) / (cols - 1)
        east = easternmost_cell + ewres / 2
        west = westernmost_cell - ewres / 2

        return DomainData(
            north=north,
            south=south,
            east=east,
            west=west,
            rows=rows,
            cols=cols,
            crs_wkt=self.crs_wkt,
        )

    def is_dataset_sorted(self) -> bool:
        """Check if all coordinates are sorted"""
        dim_is_sorted = []
        x_coords = self.dataset[self.x_dim].values
        y_coords = self.dataset[self.y_dim].values
        try:  # Check time if exists
            time_coords = self.dataset[self.time_dim].values
            time_ascending = self.is_array_sorted(time_coords, ascending=True)
            time_descending = self.is_array_sorted(time_coords, ascending=False)
            dim_is_sorted.append(time_ascending or time_descending)
        except KeyError:
            pass
        for coord in [x_coords, y_coords]:
            coord_ascending = self.is_array_sorted(coord, ascending=True)
            coord_descending = self.is_array_sorted(coord, ascending=False)
            dim_is_sorted.append(coord_ascending or coord_descending)
        return all(dim_is_sorted)

    @staticmethod
    def is_array_sorted(arr: np.ndarray, ascending: bool = True) -> bool:
        if ascending:
            return bool(np.all(np.diff(arr) >= 0))
        else:
            return bool(np.all(np.diff(arr) <= 0))

    @staticmethod
    def get_bracket_values(coord_array, target_value):
        """
        Get the lower and upper bound values that bracket a target value.

        Returns:
            tuple: (lower_value, upper_value)
        """
        insert_idx = np.searchsorted(coord_array, target_value)

        lower_idx = max(0, insert_idx - 1)
        upper_idx = min(len(coord_array) - 1, insert_idx)

        # Handle edge cases
        if insert_idx == 0:
            upper_idx = 0
        elif insert_idx >= len(coord_array):
            lower_idx = len(coord_array) - 1

        return (coord_array[lower_idx], coord_array[upper_idx])

    def get_array(
        self, map_key: str, current_time: "datetime"
    ) -> Tuple[Union["np.ndarray", None], "datetime", "datetime"]:
        """Take a given map key and current time
        return a numpy array associated with its start and end time
        if no map is found, return None instead of an array
        and a default start_time and end_time."""
        # Return None if no variable with requested name
        try:
            var_name = self.input_map_names[map_key]
        except KeyError:
            return None, self.sim_start_time, self.sim_end_time

        da = self.dataset[var_name]
        try:
            da_time = da[self.time_dim]
            if self.temporal_type == TemporalType.RELATIVE:
                # convert datetime to timedelta for the search
                np_time = np.timedelta64(current_time - self.sim_start_time)
                da_selected = da.sel({self.time_dim: np_time}, method="ffill")
                start_np, end_np = self.get_bracket_values(da_time.values, np_time)
                # convert back to datetime
                start_time = self.sim_start_time + pd.to_timedelta(start_np).to_pytimedelta()
                end_time = self.sim_start_time + pd.to_timedelta(end_np).to_pytimedelta()
                pass
            else:  # absolute time
                np_time = np.datetime64(current_time)
                da_selected = da.sel({self.time_dim: np_time}, method="ffill")
                start_np, end_np = self.get_bracket_values(da_time.values, np_time)
                start_time = pd.to_datetime(start_np).to_pydatetime()
                end_time = pd.to_datetime(end_np).to_pydatetime()
        # If no time dimension, send back the whole array
        except KeyError:
            start_time = self.sim_start_time
            end_time = self.sim_end_time
            da_selected = da
        assert len(da_selected.shape) == 2
        return da_selected.values, start_time, end_time
