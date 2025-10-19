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


class CSVVectorOutputProvider(VectorOutputProvider):
    """Save drainage simulation outputs in CSV files hosted on a cloud object storage."""

    def __init__(self, config: CSVVectorOutputConfig) -> None:
        """Initialize output provider with provider configuration."""
        self.srid = config["crs"].to_epsg()
        self.store = config["store"]
        results_prefix = config["results_prefix"]
        nodes_results_name = f"{config['drainage_results_name']}_nodes.csv"
        links_results_name = f"{config['drainage_results_name']}_links.csv"
        self.nodes_file = results_prefix + "/" + nodes_results_name
        self.links_file = results_prefix + "/" + links_results_name
        # create the CSV files
        self._write_headers("node")
        self._write_headers("link")

    def write_vector(
        self, drainage_data: DrainageNetworkData, sim_time: datetime | timedelta
    ) -> None:
        """Save simulation data for current time step."""
        # Convert sim_time to ISO8601 format
        if isinstance(sim_time, timedelta):
            # ISO8601 duration format: PT{seconds}S
            sim_time_str = f"PT{sim_time.total_seconds()}S"
        else:
            sim_time_str = sim_time.isoformat()
        # Nodes
        self._update_csv(sim_time_str, drainage_data.nodes)
        # Links
        self._update_csv(sim_time_str, drainage_data.links)

    def finalize(self, drainage_data: DrainageNetworkData) -> None:
        """Finalize outputs and cleanup."""
        pass

    def _write_headers(self, geom_type: str):
        """Create an in-memory CSV file with headers and save it in the store."""
        if "node" == geom_type:
            file_path = self.nodes_file
            attrs_class = DrainageNodeAttributes
        elif "link" == geom_type:
            file_path = self.links_file
            attrs_class = DrainageLinkAttributes
        else:
            raise RuntimeError(f"Unknown geometry type: {geom_type}")

        headers = [field.name for field in dataclasses.fields(attrs_class)]
        headers = ["sim_time"] + list(headers) + ["srid", "geometry"]
        f_obj = StringIO()
        writer = csv.writer(f_obj)
        writer.writerow(headers)
        csv_content = f_obj.getvalue()
        obstore.put(self.store, file_path, file=csv_content.encode("utf-8"))
        return self

    def _update_csv(
        self, sim_time_str, drainage_elements: Tuple[DrainageNodeData, ... | DrainageLinkData, ...]
    ):
        """Update adequate CSV in object store"""
        if isinstance(drainage_elements[0], DrainageNodeData):
            file_path = self.nodes_file
        elif isinstance(drainage_elements[0], DrainageLinkData):
            file_path = self.links_file
        else:  # nothing to write
            return

        f_obj = StringIO()
        csv_writer = csv.writer(f_obj)
        for drainage_elem in drainage_elements:
            data_line_list = [sim_time_str] + self._attrs_line(drainage_elem)
            csv_writer.writerow(data_line_list)
        new_rows = f_obj.getvalue()
        # Get the file from the store as bytes and decode it
        existing_csv = bytes(obstore.get(self.store, file_path).bytes()).decode("utf-8")
        updated_csv = existing_csv + new_rows
        obstore.put(self.store, file_path, file=updated_csv.encode("utf-8"))

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
