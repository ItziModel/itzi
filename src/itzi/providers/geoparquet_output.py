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

from datetime import datetime, timedelta
from typing import Dict, Self, Tuple, TYPE_CHECKING
from dataclasses import asdict
from pathlib import Path

import geopandas

from itzi.providers.base import VectorOutputProvider

if TYPE_CHECKING:
    from itzi.data_containers import DrainageNetworkData, DrainageNodeData, DrainageLinkData


class ParquetVectorOutputProvider(VectorOutputProvider):
    """Save drainage simulation outputs in memory."""

    def initialize(self, config: Dict | None = None) -> Self:
        """Initialize output provider."""
        self.crs = config["crs"]
        self.output_dir = Path(config["output_dir"])
        self.drainage_map_name = config["drainage_map_name"]
        return self

    def write_vector(
        self, drainage_data: "DrainageNetworkData", sim_time: datetime | timedelta
    ) -> None:
        """Save simulation data for current time step."""
        df_nodes = self.create_nodes(drainage_data.nodes)
        df_links = self.create_links(drainage_data.links)
        links_map_name = f"{self.drainage_map_name}_links_{sim_time}.parquet"
        nodes_map_name = f"{self.drainage_map_name}_nodes_{sim_time}.parquet"
        df_nodes.to_parquet(path=self.output_dir / Path(nodes_map_name), index=True)
        df_links.to_parquet(path=self.output_dir / Path(links_map_name), index=True)

    def create_nodes(self, nodes: Tuple["DrainageNodeData", ...]) -> geopandas.GeoDataFrame:
        """Create a geodataframe from multiple drainage nodes"""
        features = []
        for node in nodes:
            feature = {
                "type": "Point",
                "properties": asdict(node.attributes),
                "geometry": {"type": "Point", "coordinates": node.coordinates},
            }
            features.append(feature)
        return geopandas.GeoDataFrame.from_features(features, crs=self.crs)

    def create_links(self, links: Tuple["DrainageLinkData", ...]) -> geopandas.GeoDataFrame:
        """Create a geodataframe from multiple drainage links"""
        features = []
        for link in links:
            feature = {
                "type": "LineString",
                "properties": asdict(link.attributes),
                "geometry": {"type": "LineString", "coordinates": link.vertices},
            }
            features.append(feature)
        return geopandas.GeoDataFrame.from_features(features, crs=self.crs)

    def finalize(self, drainage_data: "DrainageNetworkData") -> None:
        """Function not needed for geoparquet."""
        pass
