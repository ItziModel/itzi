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

from typing import Dict, Self
from datetime import datetime, timedelta

import numpy as np

try:
    import xarray as xr
    import icechunk
    import zarr
except ImportError:
    raise ImportError(
        "To use the Icechunk backend, install itzi with: "
        "'uv tool install itzi[cloud]' "
        "or 'pip install itzi[cloud]'"
    )

from itzi.providers.base import RasterOutputProvider
from itzi.data_containers import SimulationData
from itzi.array_definitions import ARRAY_DEFINITIONS


class IcechunkRasterOutputProvider(RasterOutputProvider):
    """Save raster results in an Icechunk store."""

    def initialize(self, config: Dict) -> Self:
        """Create a repo in case it does not exists already"""
        # A dict of key:output_name
        self.out_map_names = config["out_map_names"]
        # A pyproj.CRS object
        self.crs = config["crs"]
        # An np.ndarray
        self.x_coords = config["x_coords"]
        # An np.ndarray
        self.y_coords = config["y_coords"]
        # An Icechunk storage object (local, S3, etc.)
        storage = config["icechunk_storage"]
        try:
            self.repo = icechunk.Repository.open(storage)
        except icechunk.IcechunkError as e:
            if "repository doesn't exist" in str(e):
                self.repo = icechunk.Repository.create(storage)
            else:
                raise
        self.cf_units = {arr_def.key: arr_def.cf_unit for arr_def in ARRAY_DEFINITIONS}
        self.cf_names = {arr_def.key: arr_def.cf_name for arr_def in ARRAY_DEFINITIONS}
        self.descriptions = {arr_def.key: arr_def.description for arr_def in ARRAY_DEFINITIONS}
        self.num_records = 0
        self.spatial_coordinates = self._get_spatial_coordinates()

    def _get_spatial_coordinates(self) -> list[tuple[str, np.ndarray, Dict]]:
        # Assume both axis have the same unit
        unit_name = self.crs.axis_info[0].unit_name
        y_attrs = {
            "axis": "Y",
            "units": unit_name,
            "long_name": "y coordinate of projection",
            "standard_name": "projection_y_coordinate",
        }
        x_attrs = {
            "axis": "X",
            "units": unit_name,
            "long_name": "x coordinate of projection",
            "standard_name": "projection_x_coordinate",
        }
        spatial_coordinates = [
            ("y", self.y_coords, y_attrs),
            ("x", self.x_coords, x_attrs),
        ]
        return spatial_coordinates

    def write_array(self, array: np.ndarray, map_key: str, sim_time: datetime | timedelta) -> None:
        """Write one array for the current time step."""
        pass

    def write_arrays(
        self, array_dict: Dict[str, np.ndarray], sim_time: datetime | timedelta
    ) -> None:
        """Write results to an icechunk repository"""
        # prepare the data
        data_vars = {}
        if isinstance(sim_time, datetime):
            time_dtype = "datetime64[ms]"
            sim_time_np = np.datetime64(sim_time, "ms")
            time_unit = "milliseconds since 1970-01-01T00:00:00"
        elif isinstance(sim_time, timedelta):
            time_dtype = "timedelta64[ms]"
            sim_time_np = np.timedelta64(sim_time, "ms")
            time_unit = "milliseconds"
        else:
            raise ValueError(f"Unknown temporal type: {type(sim_time)}")

        time_coordinate = np.array([sim_time_np], dtype=time_dtype)
        # assert time_coordinate.dtype == time_dtype
        time_attrs = {}
        time_encoding = {
            "units": time_unit,
            "dtype": time_dtype,
            # 'calendar': 'proleptic_gregorian',
        }

        coordinates = [("time", time_coordinate, time_attrs)] + self.spatial_coordinates
        for key, arr in array_dict.items():
            coords_shape = (len(self.y_coords), len(self.x_coords))
            if arr.shape != coords_shape:
                raise ValueError(
                    f"Array shape {arr.shape} incompatible with coordinates shape {coords_shape}"
                )

            variable_name = self.out_map_names[key]
            var_attributes = {
                "units": self.cf_units[key],
                "standard_name": self.cf_names[key],
                "long_name": self.descriptions[key],
            }
            data_array = xr.DataArray(
                data=np.expand_dims(arr, axis=0),
                coords=coordinates,
                name=variable_name,
                attrs=var_attributes,
            )
            assert data_array["time"].dtype == time_dtype

            data_array.encoding["time"] = time_encoding
            data_vars[variable_name] = data_array
        dataset_attributes = {
            "crs_wkt": self.crs.to_wkt(),
        }
        dataset = xr.Dataset(data_vars, attrs=dataset_attributes)
        # Commit to the repo
        commit_message = f"itzi results for simulation time {sim_time}"
        with self.repo.transaction("main", message=commit_message) as store:
            # /!\ this will overwrite an existing store
            if self.num_records == 0:
                dataset.to_zarr(
                    store,
                    zarr_format=3,
                    encoding={"time": time_encoding},
                    consolidated=False,
                    mode="w",
                )
            else:
                # Use zarr directly to append data while preserving time encoding
                self._append_to_zarr_store(store, dataset, sim_time_np)
        self.num_records += 1

    def _append_to_zarr_store(
        self, store, dataset: xr.Dataset, sim_time_np: np.datetime64 | np.timedelta64
    ) -> None:
        """Append data to zarr store using zarr directly to preserve time encoding."""
        # Open the zarr group
        z_group = zarr.open_group(store, mode="r+")

        # Get current time array and append new time
        current_time = z_group["time"][:]
        new_time_array = np.append(current_time, sim_time_np)

        # Resize and update time coordinate
        z_group["time"].resize(len(new_time_array))
        z_group["time"][:] = new_time_array

        # Append data for each variable
        for var_name, data_array in dataset.data_vars.items():
            current_data = z_group[var_name][:]
            new_data = data_array.values
            # Concatenate along time dimension (axis 0)
            combined_data = np.concatenate([current_data, new_data], axis=0)
            # Resize and update the variable
            z_group[var_name].resize(combined_data.shape)
            z_group[var_name][:] = combined_data

    def finalize(self, final_data: SimulationData) -> None:
        """Finalize outputs and cleanup."""
        pass
