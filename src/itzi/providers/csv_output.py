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
from datetime import datetime, timedelta
from typing import TypedDict, TYPE_CHECKING, Tuple, List
from io import StringIO
import csv
import dataclasses

import pandas as pd

from itzi.providers.base import VectorOutputProvider
from itzi.data_containers import DrainageLinkData, DrainageLinkAttributes
from itzi.data_containers import DrainageNodeData, DrainageNodeAttributes

if TYPE_CHECKING:
    from itzi.data_containers import DrainageNetworkData

try:
    import obstore
    import pyproj
except ImportError:
    raise ImportError(
        "To use the CSV backend, install itzi with: "
        "'uv tool install itzi[cloud]' "
        "or 'pip install itzi[cloud]'"
    )


class CSVVectorOutputConfig(TypedDict):
    crs: pyproj.CRS | None
    store: obstore.store.ObjectStore
    results_prefix: str
    drainage_results_name: str
    overwrite: bool


class CSVVectorOutputProvider(VectorOutputProvider):
    """Save drainage simulation outputs in CSV files hosted on a cloud object storage.
    Write two files:
        - one for nodes, with suffix *_nodes.csv
        - one for links, with suffix *_nodes.csv
    If a file already exists at the prefix and overwrite is False, the results are appended if possible."""

    def __init__(self, config: CSVVectorOutputConfig) -> None:
        """Initialize output provider with provider configuration."""
        try:
            self.srid = config["crs"].to_epsg()
        except AttributeError:
            self.srid = 0
        self.store = config["store"]
        results_prefix = config["results_prefix"]

        self.existing_ids = {"link": None, "node": None}  # Objects ids already in the file
        self.existing_max_time = {"link": None, "node": None}  # Max of sim_time in existing_file
        self.number_of_writes = {"link": 0, "node": 0}
        self.file_paths = {"link": None, "node": None}
        self.headers = {"link": None, "node": None}
        self.append_mode = {"link": True, "node": True}
        if config["overwrite"]:
            self.append_mode = {"link": False, "node": False}

        for geom_type, obj in [("node", DrainageNodeAttributes), ("link", DrainageLinkAttributes)]:
            base_headers = [field.name for field in dataclasses.fields(obj)]
            self.headers[geom_type] = ["sim_time"] + list(base_headers) + ["srid", "geometry"]

            results_name = f"{config['drainage_results_name']}_{geom_type}s.csv"
            self.file_paths[geom_type] = results_prefix + "/" + results_name
            # No need to check if we overwrite
            if not config["overwrite"]:
                self._check_existing_csv(geom_type)
            # create the CSV files
            if not self.append_mode[geom_type]:
                self._write_headers(geom_type)
        print(self.existing_ids)
        print(self.existing_max_time)

    def write_vector(
        self, drainage_data: DrainageNetworkData, sim_time: datetime | timedelta
    ) -> None:
        """Save simulation data for current time step."""
        # Validate time on first write
        self._validate_time_on_first_write(sim_time)
        # Convert sim_time to ISO8601 format
        if isinstance(sim_time, timedelta):
            # ISO8601 duration format: PT{seconds}S
            sim_time_str = f"PT{sim_time.total_seconds()}S"
        else:
            sim_time_str = sim_time.isoformat()
        # Nodes
        self._update_csv(sim_time_str, "node", drainage_data.nodes)
        self.number_of_writes["node"] += 1
        # Links
        self._update_csv(sim_time_str, "link", drainage_data.links)
        self.number_of_writes["link"] += 1

    def finalize(self, drainage_data: DrainageNetworkData) -> None:
        """Finalize outputs and cleanup."""
        pass

    def _check_existing_csv(self, geom_type: str):
        """In order to be compatible, an existing CSV should have:
            - Same headers
        Other compatibility issues, like:
            - Same sim_time type
            - new sim_time < existing
            - new object ID ≠ existing ones
        could not be checked without drainage network data
        """
        existing_csv = None
        try:
            existing_csv = StringIO(
                bytes(obstore.get(self.store, self.file_paths[geom_type]).bytes()).decode("utf-8")
            )
        except FileNotFoundError:
            self.append_mode[geom_type] = False
            return
        df_csv = pd.read_csv(existing_csv)
        existing_headers = list(df_csv.columns)
        expected_headers = self.headers[geom_type]
        if not existing_headers == expected_headers:
            raise ValueError(f"Headers mismatch in existing file {self.file_paths[geom_type]}.")
            self.append_mode[geom_type] = False
        id_col = f"{geom_type}_id"

        # Store values existing ids
        self.existing_ids[geom_type] = set(df_csv[id_col])
        # Store maximum sim_time values
        try:
            self.existing_max_time[geom_type] = pd.to_timedelta(
                max(df_csv["sim_time"])
            ).to_pytimedelta()
        except ValueError:
            try:
                self.existing_max_time[geom_type] = pd.to_datetime(
                    max(df_csv["sim_time"])
                ).to_pydatetime()
            except ValueError:
                raise ValueError(
                    f"Unknown sim_time column in existing file {self.file_paths[geom_type]}."
                )
        print(df_csv)

    def _write_headers(self, geom_type: str):
        """Create an in-memory CSV file with headers and save it in the store."""
        f_obj = StringIO()
        writer = csv.writer(f_obj)
        writer.writerow(self.headers[geom_type])
        csv_content = f_obj.getvalue()
        obstore.put(self.store, self.file_paths[geom_type], file=csv_content.encode("utf-8"))
        return self

    def _validate_time_on_first_write(self, sim_time: datetime | timedelta) -> None:
        """Validate sim_time type matches existing files on first write."""

        for geom_type in ["node", "link"]:
            # Only validate on first write
            if self.number_of_writes[geom_type] > 0 or self.existing_max_time[geom_type] is None:
                continue
            # Type must match
            if type(self.existing_max_time[geom_type]) is not type(sim_time):
                time_type_name = (
                    "relative (timedelta)"
                    if isinstance(sim_time, timedelta)
                    else "absolute (datetime)"
                )
                existing_type_name = (
                    "relative"
                    if isinstance(self.existing_max_time[geom_type], timedelta)
                    else "absolute"
                )
                raise ValueError(
                    f"Time type mismatch for {geom_type}: "
                    f"attempting to write {time_type_name} but existing file has {existing_type_name}"
                )
            # Time must increase
            if not sim_time > self.existing_max_time[geom_type]:
                raise ValueError(
                    f"Time not increasing for {geom_type}: attempting to write {sim_time} but "
                    f"existing file has a max sim_time value of {self.existing_max_time[geom_type]}"
                )

    def _update_csv(
        self,
        sim_time_str: str,
        geom_type: str,
        drainage_elements: Tuple[DrainageNodeData, ... | DrainageLinkData, ...],
    ):
        """Update adequate CSV in object store"""
        # Check compatibility on first write
        if 0 == self.number_of_writes[geom_type] and self.existing_ids[geom_type]:
            # IDs must match
            new_ids = set(
                [
                    dataclasses.asdict(drainage_elem.attributes)[f"{geom_type}_id"]
                    for drainage_elem in drainage_elements
                ]
            )
            if not new_ids == self.existing_ids[geom_type]:
                raise ValueError(
                    f"Object ids mismatch for {geom_type}: "
                    f"attempting to write {new_ids} but existing file has {self.existing_ids[geom_type]}"
                )
        f_obj = StringIO()
        csv_writer = csv.writer(f_obj)
        for drainage_elem in drainage_elements:
            data_line_list = [sim_time_str] + self._attrs_line(drainage_elem)
            csv_writer.writerow(data_line_list)
        new_rows = f_obj.getvalue()
        # Get the file from the store as bytes and decode it
        existing_csv = bytes(obstore.get(self.store, self.file_paths[geom_type]).bytes()).decode(
            "utf-8"
        )
        updated_csv = existing_csv + new_rows
        obstore.put(self.store, self.file_paths[geom_type], file=updated_csv.encode("utf-8"))

    def _attrs_line(self, drainage_element: DrainageNodeData | DrainageLinkData) -> List[str, ...]:
        """Return a list of attributes"""
        # Convert attributes to list
        attributes = [str(a) for a in dataclasses.asdict(drainage_element.attributes).values()]
        # Create geometry WKT
        if isinstance(drainage_element, DrainageNodeData):
            if drainage_element.coordinates is not None:
                x, y = drainage_element.coordinates
                geom_wkt = f"POINT({x} {y})"
            else:
                geom_wkt = ""
        elif isinstance(drainage_element, DrainageLinkData):
            if drainage_element.vertices is not None and len(drainage_element.vertices) > 0:
                # Filter out None vertices
                valid_vertices = [v for v in drainage_element.vertices if v is not None]
                if len(valid_vertices) >= 2:
                    coords_str = ", ".join([f"{x} {y}" for x, y in valid_vertices])
                    geom_wkt = f"LINESTRING({coords_str})"
                else:
                    # Not enough valid vertices to create a linestring
                    geom_wkt = ""
            else:
                geom_wkt = ""
        else:
            raise RuntimeError(f"Unknown drainage_element: {type(drainage_element)}")
        return attributes + [str(self.srid), geom_wkt]
