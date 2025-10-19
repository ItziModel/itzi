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
from typing import TypedDict, TYPE_CHECKING
from io import StringIO
import csv
import dataclasses

from itzi.providers.base import VectorOutputProvider
from itzi.data_containers import DrainageLinkAttributes, DrainageNodeAttributes


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
        self.crs = config["crs"]
        self.store = config["store"]
        results_prefix = config["results_prefix"]
        nodes_results_name = f"{config['drainage_results_name']}_nodes.csv"
        links_results_name = f"{config['drainage_results_name']}_links.csv"
        self.nodes_file = results_prefix + "/" + nodes_results_name
        self.links_file = results_prefix + "/" + links_results_name

        # create the CSV files
        self._write_headers("node")
        self._write_headers("link")

    def _write_headers(self, geom_type: str):
        """Create an in-memory CSV file with headers and save it in the store."""
        if "node" == geom_type:
            file_path = self.nodes_file
            dclass = DrainageNodeAttributes
        elif "link" == geom_type:
            file_path = self.links_file
            dclass = DrainageLinkAttributes
        else:
            raise RuntimeError(f"Unknown geometry type: {geom_type}")

        f_obj = StringIO()
        writer = csv.writer(f_obj)
        headers = [field.name for field in dataclasses.fields(dclass)]
        headers = ["sim_time"] + list(headers) + ["srid", "geometry"]
        writer.writerow(headers)
        csv_content = f_obj.getvalue()
        obstore.put(self.store, file_path, file=csv_content.encode("utf-8"))
        return self

    def write_vector(
        self, drainage_data: DrainageNetworkData, sim_time: datetime | timedelta
    ) -> None:
        """Save simulation data for current time step."""
        # Nodes
        with obstore.open_reader(self.store, self.nodes_file):
            pass

    def finalize(self, drainage_data: DrainageNetworkData) -> None:
        """Finalize outputs and cleanup."""
        pass
