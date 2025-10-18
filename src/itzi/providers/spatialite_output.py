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
from typing import TypedDict, TYPE_CHECKING
from dataclasses import asdict
from pathlib import Path
import sqlite3


from itzi.providers.base import VectorOutputProvider
from itzi.data_containers import DrainageNodeAttributes, DrainageLinkAttributes

if TYPE_CHECKING:
    from itzi.data_containers import DrainageNetworkData
    import pyproj


class SpatialiteVectorOutputConfig(TypedDict):
    crs: "pyproj.CRS"
    output_dir: Path
    drainage_map_name: str


class SpatialiteVectorOutputProvider(VectorOutputProvider):
    """Save drainage simulation outputs to SQLite/Spatialite database."""

    def __init__(self, config: SpatialiteVectorOutputConfig) -> None:
        """Initialize output provider and create SQLite database with tables."""
        try:
            self.srid = config["crs"].to_epsg()
        except Exception:  # Set to undefined
            self.srid = 0

        # Ensure output directory exists
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create database file path
        drainage_map_name = config["drainage_map_name"]
        db_name = f"{drainage_map_name}.db"
        self.db_path = output_dir / db_name

        # Connect to database and enable WAL mode
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA journal_mode=WAL")

        # Load spatialite extension
        self.conn.enable_load_extension(True)
        try:
            self.conn.load_extension("mod_spatialite")
        except sqlite3.OperationalError:
            # Try alternative spatialite extension name
            try:
                self.conn.load_extension("libspatialite")
            except sqlite3.OperationalError:
                raise RuntimeError("spatialite not found")

        # Initialize spatial metadata
        self.cursor.execute("SELECT InitSpatialMetadata(1)")

        # Create tables
        self._create_table("nodes")
        self._create_table("links")
        self.conn.commit()

    def _create_table(self, table_name: str) -> None:
        """Create a nodes or links table with spatialite geometry."""
        # Get column definitions
        if table_name == "nodes":
            columns = DrainageNodeAttributes.get_columns_definition(cat_primary_key=False)
        elif table_name == "links":
            columns = DrainageLinkAttributes.get_columns_definition(cat_primary_key=False)
        else:
            raise ValueError("Unknown table name")

        # Create table with auto-incrementing primary key
        cols_sql = (
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            + "sim_time TEXT, "
            + ", ".join([f"{name} {dtype}" for name, dtype in columns])
        )
        create_table = f"CREATE TABLE IF NOT EXISTS {table_name} ({cols_sql})"
        self.cursor.execute(create_table)

        # Add geometry column using spatialite
        if table_name == "nodes":
            self.cursor.execute(
                f"SELECT AddGeometryColumn('{table_name}', 'geometry', {self.srid}, 'POINT', 'XY')"
            )
        elif table_name == "links":
            self.cursor.execute(
                f"SELECT AddGeometryColumn('{table_name}', 'geometry', {self.srid}, 'LINESTRING', 'XY')"
            )
        else:
            raise ValueError("Unknown table name")

    def write_vector(
        self, drainage_data: "DrainageNetworkData", sim_time: datetime | timedelta
    ) -> None:
        """Save simulation data for current time step."""
        # Convert sim_time to ISO8601 format
        if isinstance(sim_time, timedelta):
            # ISO8601 duration format: PT{seconds}S
            sim_time_str = f"PT{sim_time.total_seconds()}S"
        else:
            sim_time_str = sim_time.isoformat()

        # Write nodes
        for node_data in drainage_data.nodes:
            # Convert attributes to dict
            attrs_dict = asdict(node_data.attributes)

            # Create geometry WKT
            if node_data.coordinates is not None:
                x, y = node_data.coordinates
                geom_wkt = f"POINT({x} {y})"
            else:
                geom_wkt = None

            # Prepare column names and values
            columns = ["sim_time"] + list(attrs_dict.keys())
            values = [sim_time_str] + list(attrs_dict.values())

            # Build SQL with geometry
            if geom_wkt is not None:
                columns.append("geometry")
                values.append(geom_wkt)
                placeholders = ["?"] * (len(columns) - 1) + [f"GeomFromText(?, {self.srid})"]
            else:
                placeholders = ["?"] * len(columns)

            # Insert row
            sql = f"INSERT INTO nodes ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
            self.cursor.execute(sql, values)

        # Write links
        for link_data in drainage_data.links:
            # Convert attributes to dict
            attrs_dict = asdict(link_data.attributes)

            # Create geometry WKT
            if link_data.vertices is not None and len(link_data.vertices) > 0:
                # Filter out None vertices
                valid_vertices = [v for v in link_data.vertices if v is not None]
                if len(valid_vertices) >= 2:
                    coords_str = ", ".join([f"{x} {y}" for x, y in valid_vertices])
                    geom_wkt = f"LINESTRING({coords_str})"
                else:
                    # Not enough valid vertices to create a linestring
                    geom_wkt = None
            else:
                geom_wkt = None

            # Prepare column names and values
            columns = ["sim_time"] + list(attrs_dict.keys())
            values = [sim_time_str] + list(attrs_dict.values())

            # Build SQL with geometry
            if geom_wkt is not None:
                columns.append("geometry")
                values.append(geom_wkt)
                placeholders = ["?"] * (len(columns) - 1) + [f"GeomFromText(?, {self.srid})"]
            else:
                placeholders = ["?"] * len(columns)

            # Insert row
            sql = f"INSERT INTO links ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
            self.cursor.execute(sql, values)

        # Commit transaction
        self.conn.commit()

    def finalize(self, drainage_data: "DrainageNetworkData") -> None:
        """Close database connection."""
        if hasattr(self, "conn"):
            self.conn.close()
