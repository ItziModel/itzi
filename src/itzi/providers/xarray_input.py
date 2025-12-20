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

from __future__ import annotations

from typing import Iterable, Mapping, TypedDict, NotRequired, TYPE_CHECKING

import numpy as np

try:
    import xarray as xr
    import pandas as pd
except ImportError:
    raise ImportError(
        "To use the xarray input backend, install itzi with: "
        "'uv tool install itzi[cloud]' "
        "or 'pip install itzi[cloud]'"
    )

from itzi.providers.base import RasterInputProvider
from itzi.providers.domain_data import DomainData
from itzi.const import TemporalType

if TYPE_CHECKING:
    from datetime import datetime


type DimensionsDict = dict[str, dict[str, str]]


class XarrayRasterInputConfig(TypedDict):
    dataset: xr.Dataset
    # A dict of key: input names
    input_map_names: Mapping[str, str]
    # A Mapping of dimensions names
    # {"time": "start_time", "x": "longitude"}
    dimension_names: NotRequired[DimensionsDict]
    simulation_start_time: datetime
    simulation_end_time: datetime


class DimensionsDictFormatter:
    """Populate the dimension mapping based on default values and user-provided ones."""

    # Default dimension names
    default_dims: dict[str, str] = {
        "time": "time",
        "y": "y",
        "x": "x",
    }

    def __init__(self, input_var_names: Iterable[str], input_dims: DimensionsDict | None):
        self.input_var_names = input_var_names
        self.input_dims = input_dims
        self._dataset_dims: DimensionsDict = {}

        # Instantiate the dimensions with default values
        for var_name in input_var_names:
            self._dataset_dims[var_name] = (
                # copy avoids shared state
                self.default_dims.copy()
            )

        self._check_input_dims()

    def _check_input_dims(self):
        """Check conformity of provided dims dict."""
        if self.input_dims is None:
            return
        if not isinstance(self.input_dims, dict):
            raise TypeError("The 'dims' parameter must be of type dict[str, dict[str, str]].")
        # Check that all values are dicts
        for var_name, dims in self.input_dims.items():
            if not isinstance(dims, dict):
                raise TypeError(
                    f"The 'dims' parameter must be of type dict[str, dict[str, str]]. "
                    f"Value for key '{var_name}' is not a dict."
                )
            if var_name not in self.input_var_names:
                raise ValueError(
                    f"Variable {var_name} not found in the input dataset. "
                    f"Variables found: {self.input_var_names}"
                )

    def _fill_dims(self):
        """Replace the default values with those given by the user."""
        if self.input_dims is None:
            return
        for var_name, dims in self.input_dims.items():
            for dim_key, dim_value in dims.items():
                self._dataset_dims[var_name][dim_key] = dim_value

    def get_formatted_dims(self) -> DimensionsDict:
        self._fill_dims()
        return self._dataset_dims


class XarrayRasterInputProvider(RasterInputProvider):
    """Abstract base class for handling raster simulation inputs."""

    def __init__(self, config: XarrayRasterInputConfig) -> None:
        self.sim_start_time = config["simulation_start_time"]
        self.sim_end_time = config["simulation_end_time"]
        self.input_map_names = config["input_map_names"]

        # Open dataset
        self.dataset = config["dataset"]
        self.crs_wkt: str = self.dataset.attrs.get("crs_wkt", "")

        # Set dimensions Mapping
        dim_names = config.get("dimension_names")
        input_var_names: list[str] = [str(var_name) for var_name in self.dataset.data_vars.keys()]
        dim_formatter = DimensionsDictFormatter(input_var_names, dim_names)
        self.dataset_dims = dim_formatter.get_formatted_dims()

        # Data validation
        self._validate_map_names_are_variables()
        self._validate_dimensions()
        self._validate_variables_dimensionality()
        self._validate_equal_spacing_of_spatial_dims()
        self._validate_coordinates_are_sorted()
        self._validate_equality_of_spatial_dims()

        self.temporal_types = self.detect_temporal_type()
        self.origin: tuple[float, float] = self.get_origin()

    def _validate_map_names_are_variables(self):
        """Make sure that the provided maps names exist as variables in the provided dataset."""
        var_names: list[str] = [str(var_name) for var_name in self.dataset.data_vars.keys()]
        for map_key, map_name in self.input_map_names.items():
            if map_name not in var_names:
                raise ValueError(
                    f"provided input map name {map_name} "
                    f"for map key {map_key} not found in dataset."
                )

    def _validate_dimensions(self) -> None:
        """Validate that:
        - the specified spatial dimensions exist in the dataset.
        - The dimensions are one-dimensional."""
        for var_name in self.dataset.data_vars.keys():
            da_var: xr.DataArray = self.dataset[var_name]
            var_dims: set[str] = set(da_var.dims)
            for dim_type in ["x", "y"]:
                dim_name: str = self.dataset_dims[var_name][dim_type]
                if dim_name not in var_dims:
                    raise ValueError(
                        f"{dim_type} dimension '{dim_name}' not found in variable {var_name}. "
                        f"Available dimensions: {var_dims}"
                    )
                da_dim: xr.DataArray = self.dataset[dim_name]
                if da_dim.ndim != 1:
                    raise ValueError(
                        f"Dimension '{dim_name}' not one-dimensional. "
                        f"Found {da_dim.ndim} dimensions with shape {da_dim.shape}"
                    )

    def _validate_variables_dimensionality(self) -> None:
        """Validate that all variables are either 2D[y, x] or 3D[time, y, x]."""
        for var_name in self.input_map_names.values():
            da_var: xr.DataArray = self.dataset[var_name]
            var_dims: set[str] = set(da_var.dims)
            num_dims = len(da_var.dims)
            x_dim: str = self.dataset_dims[var_name]["x"]
            y_dim: str = self.dataset_dims[var_name]["y"]

            # Check if variable is 2D or 3D
            if num_dims == 2:
                # Must have x and y dimensions
                if x_dim not in var_dims or y_dim not in var_dims:
                    raise ValueError(
                        f"2D variable '{var_name}' must have dimensions [{y_dim}, {x_dim}]. "
                        f"Found: {list(da_var.dims)}"
                    )
            elif num_dims == 3:
                time_dim: str = self.dataset_dims[var_name]["time"]

                # Must have x, y, and time dimensions
                if time_dim not in var_dims:
                    raise ValueError(
                        f"3D variable '{var_name}' must have time dimension '{time_dim}'. "
                        f"Found: {list(da_var.dims)}"
                    )
                if x_dim not in var_dims or y_dim not in var_dims:
                    raise ValueError(
                        f"3D variable '{var_name}' must have "
                        f"dimensions [{time_dim}, {y_dim}, {x_dim}]. "
                        f"Found: {list(da_var.dims)}"
                    )
            else:
                raise ValueError(
                    f"Variable '{var_name}' must be either 2D[y, x] or 3D[time, y, x]. "
                    f"Found {num_dims}D: {list(da_var.dims)}"
                )

    def _validate_equal_spacing_of_spatial_dims(self) -> None:
        """Check if spatial coordinates are equally spaced."""
        for var_name in self.input_map_names.values():
            x_dim: str = self.dataset_dims[var_name]["x"]
            y_dim: str = self.dataset_dims[var_name]["y"]
            for dim_name in [x_dim, y_dim]:
                coord: xr.DataArray = self.dataset[dim_name]
                diffs = np.diff(coord.values if hasattr(coord, "values") else coord)
                # no coordinates present
                if len(diffs) == 0:
                    pass
                if not np.allclose(diffs, diffs[0]):
                    raise ValueError(
                        f"Dimension {dim_name} of variable {var_name} not equally spaced."
                    )

    def _validate_equality_of_spatial_dims(self):
        """Check that all spatial dimensions are equals.
        ⚠️ Must check first that they are all one-dimensional,
        if not, the check might wrongly pass because of numpy broadcasting.
        """
        for dim_type in ["x", "y"]:
            dim_names: set[str] = {
                self.dataset_dims[var_name][dim_type] for var_name in self.input_map_names.values()
            }
            if len(dim_names) == 0:
                continue
            da_list: list[xr.DataArray] = [self.dataset[dim_name] for dim_name in dim_names]
            ref_da: np.ndarray = da_list[0]
            for da in da_list:
                if not np.allclose(ref_da.values, da.values):
                    raise ValueError(
                        f"{dim_type} dimensions {ref_da.name} and {da.name} are not equals"
                    )

    def _validate_coordinates_are_sorted(self):
        """Check if all coordinates are sorted."""
        for coord_name, da_coord in self.dataset.coords.items():
            arr_coord: np.ndarray = da_coord.values
            coord_ascending = self.is_array_sorted(arr_coord, ascending=True)
            coord_descending = self.is_array_sorted(arr_coord, ascending=False)
            dim_is_sorted = coord_ascending or coord_descending
            if not dim_is_sorted:
                raise ValueError(f"Coordinates array {coord_name} is not sorted.")

    def detect_temporal_type(self) -> dict[str, TemporalType]:
        """Detect if time coordinates are relative (timedelta) or absolute (datetime)."""
        temporal_type_dict: dict[str, TemporalType] = {}
        try:
            time_dim_names: set[str] = {
                self.dataset_dims[var_name]["time"] for var_name in self.input_map_names.values()
            }
        # If no time dimension, return empty dict
        except KeyError:
            return temporal_type_dict

        for time_dim_name in time_dim_names:
            time_coords = self.dataset[time_dim_name]
            # Check the dtype of time coordinates
            if np.issubdtype(time_coords.dtype, np.timedelta64):
                temporal_type_dict[time_dim_name] = TemporalType.RELATIVE
            elif np.issubdtype(time_coords.dtype, np.datetime64):
                temporal_type_dict[time_dim_name] = TemporalType.ABSOLUTE
            else:
                raise ValueError(f"Unsupported temporal type: {time_coords.dtype}")
        return temporal_type_dict

    def get_domain_data(self) -> DomainData:
        """Return a DomainData object."""
        # get the first coords. They are all the same (checked at init).
        y_dim_name: str = [
            self.dataset_dims[var_name]["y"] for var_name in self.input_map_names.values()
        ][0]
        x_dim_name: str = [
            self.dataset_dims[var_name]["x"] for var_name in self.input_map_names.values()
        ][0]
        # Coordinates are at the center of the cells
        y_coords: xr.DataArray = self.dataset[y_dim_name]
        rows: int = len(y_coords)
        northernmost_cell = y_coords.values.max()
        southernmost_cell = y_coords.values.min()
        nsres = float((northernmost_cell - southernmost_cell) / (rows - 1))
        north = float(northernmost_cell + nsres / 2)
        south = float(southernmost_cell - nsres / 2)

        x_coords: xr.DataArray = self.dataset[x_dim_name]
        cols: int = len(x_coords)
        easternmost_cell = x_coords.values.max()
        westernmost_cell = x_coords.values.min()
        ewres = float((easternmost_cell - westernmost_cell) / (cols - 1))
        east = float(easternmost_cell + ewres / 2)
        west = float(westernmost_cell - ewres / 2)

        return DomainData(
            north=north,
            south=south,
            east=east,
            west=west,
            rows=rows,
            cols=cols,
            crs_wkt=self.crs_wkt,
        )

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
        self, map_key: str, current_time: datetime
    ) -> tuple[np.ndarray | None, datetime, datetime]:
        """Take a given map key and current time
        return a numpy array associated with its start and end time
        if no map is found, return None instead of an array
        and a default start_time and end_time."""
        # Return None if no variable with requested name
        try:
            var_name: str = self.input_map_names[map_key]
        except KeyError:
            return None, self.sim_start_time, self.sim_end_time

        da: xr.DataArray = self.dataset[var_name]
        time_dim: str = self.dataset_dims[var_name]["time"]

        if time_dim in da.dims:
            da_time: xr.DataArray = da[time_dim]
            if self.temporal_types[time_dim] == TemporalType.RELATIVE:
                # convert datetime to timedelta for the search
                np_time = np.timedelta64(current_time - self.sim_start_time)
                da_selected: xr.DataArray = da.sel({time_dim: np_time}, method="ffill")
                start_np, end_np = self.get_bracket_values(da_time.values, np_time)
                # convert back to datetime
                start_time = self.sim_start_time + pd.to_timedelta(start_np).to_pytimedelta()
                end_time = self.sim_start_time + pd.to_timedelta(end_np).to_pytimedelta()
            else:  # absolute time
                np_time = np.datetime64(current_time)
                da_selected: xr.DataArray = da.sel({time_dim: np_time}, method="ffill")
                start_np, end_np = self.get_bracket_values(da_time.values, np_time)
                start_time = pd.to_datetime(start_np).to_pydatetime()
                end_time = pd.to_datetime(end_np).to_pydatetime()
        else:
            # If no time dimension, send back the whole array
            start_time: datetime = self.sim_start_time
            end_time: datetime = self.sim_end_time
            da_selected: xr.DataArray = da

        assert len(da_selected.shape) == 2
        return da_selected.values, start_time, end_time
