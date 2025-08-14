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
    import pyproj
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
        # A list of var names to be written
        self.out_var_names = config["out_var_names"]
        # A pyproj.CRS object
        self.crs = config["crs"]
        # An np.ndarray
        self.x_coords = config["x_coords"]
        # An np.ndarray
        self.y_coords = config["y_coords"]
        # An Icechunk storage object (local, S3, etc.)
        storage = config["icechunk_storage"]
        is_existing_repo = True
        try:
            self.repo = icechunk.Repository.open(storage)
        except icechunk.IcechunkError as e:
            if "repository doesn't exist" in str(e):
                self.repo = icechunk.Repository.create(storage)
                is_existing_repo = False
            else:
                raise
        self.spatial_coordinates = self._get_spatial_coordinates()
        # Make sure new data matches existing one
        if is_existing_repo:
            self.check_repo_match()

        self.cf_units = {arr_def.key: arr_def.cf_unit for arr_def in ARRAY_DEFINITIONS}
        self.cf_names = {arr_def.key: arr_def.cf_name for arr_def in ARRAY_DEFINITIONS}
        self.descriptions = {arr_def.key: arr_def.description for arr_def in ARRAY_DEFINITIONS}
        self.num_records = 0
        return self

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

    def check_repo_match(self):
        """Raises ValueError if entry data does not match the existing repo."""
        crs_match = False
        vars_match = False
        repo_match = False

        # Get crs, variables, dims and coordinates from existing repo
        session = self.repo.readonly_session("main")
        try:
            existing_ds = xr.open_zarr(session.store)
        except Exception:
            raise ValueError(f"Existing {session.store} is not a valid zarr store")
        try:
            existing_crs = pyproj.CRS.from_wkt(existing_ds.attrs["crs_wkt"])
        except Exception:
            crs_match = False

        existing_vars = existing_ds.data_vars
        existing_num_vars = len(existing_vars)
        # Check if existing variables coordinates match the new ones
        # This implies coherence in variables names and dims
        if existing_crs == self.crs:
            crs_match = True
        new_num_vars = len([self.out_var_names])
        vars_match_dict = {}
        for existing_var_name in existing_vars:
            existing_var = existing_ds[existing_var_name]
            try:
                # incompatible dimension names will fail here
                existing_var_x_coords = existing_var.coords["x"].values
                existing_var_y_coords = existing_var.coords["y"].values
            except Exception:
                vars_match_dict[existing_var_name] = False
                continue
            try:  # allclose raises ValueError is not same len
                var_x_coords_match = np.allclose(existing_var_x_coords, self.x_coords)
                var_y_coords_match = np.allclose(existing_var_y_coords, self.y_coords)
                if var_x_coords_match and var_y_coords_match:
                    vars_match_dict[existing_var_name] = True
            except ValueError:
                vars_match_dict[existing_var_name] = False

        if all(list(vars_match_dict.values())) and existing_num_vars == new_num_vars:
            vars_match = True
        # Raise if not full match
        repo_match = crs_match and vars_match
        if not repo_match:
            raise ValueError(
                f"Provided data does not match existing icechunk repository {self.repo}: "
                f"Existing CRS: {existing_crs.to_epsg()}, "
                f"New CRS: {self.crs.to_epsg()}, "
                f"Matching variables coordinates: {vars_match_dict}. "
            )

    def write_arrays(
        self, array_dict: Dict[str, np.ndarray], sim_time: datetime | timedelta
    ) -> None:
        """Write results to an icechunk repository"""
        vars_to_write = array_dict.keys()
        if set(self.out_var_names) != set(vars_to_write):
            raise ValueError(
                "Variables names do not match: "
                f"Expected: {self.out_var_names} "
                f"Received: {vars_to_write}"
            )
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
        }

        coordinates = [("time", time_coordinate, time_attrs)] + self.spatial_coordinates
        for key, arr in array_dict.items():
            coords_shape = (len(self.y_coords), len(self.x_coords))
            if arr.shape != coords_shape:
                raise ValueError(
                    f"Array shape {arr.shape} incompatible with coordinates shape {coords_shape}"
                )

            var_attributes = {
                "units": self.cf_units[key],
                "standard_name": self.cf_names[key],
                "long_name": self.descriptions[key],
            }
            data_array = xr.DataArray(
                data=np.expand_dims(arr, axis=0),
                coords=coordinates,
                name=key,
                attrs=var_attributes,
            )
            assert data_array["time"].dtype == time_dtype

            data_array.encoding["time"] = time_encoding
            data_vars[key] = data_array
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
        """Write max values."""
        if self.out_map_names["water_depth"]:
            pass
        if self.out_map_names["v"]:
            pass
