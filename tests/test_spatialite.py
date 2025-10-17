import tempfile
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import fields

import pytest
import pyproj

from itzi.data_containers import (
    DrainageNetworkData,
    DrainageNodeData,
    DrainageNodeAttributes,
    DrainageLinkData,
    DrainageLinkAttributes,
)
from itzi.drainage import CouplingTypes
from itzi.providers.spatialite_output import SpatialiteVectorOutputProvider


@pytest.fixture(scope="module")
def temp_dir():
    return tempfile.TemporaryDirectory()


def create_dummy_drainage_network():
    """Create a dummy DrainageNetwork object with 3 nodes and 2 links."""

    # Create 3 drainage nodes with correct types from drainage.py
    node1_attributes = DrainageNodeAttributes(
        node_id="N1",
        node_type="junction",  # Valid: junction, outfall, divider, storage
        coupling_type=CouplingTypes.ORIFICE,
        coupling_flow=0.5,
        inflow=1.0,
        outflow=0.8,
        lateral_inflow=0.2,
        losses=0.1,
        overflow=0.0,
        depth=1.5,
        head=101.5,
        crest_elevation=100.0,
        invert_elevation=98.0,
        initial_depth=0.5,
        full_depth=2.0,
        surcharge_depth=0.5,
        ponding_area=10.0,
        volume=15.0,
        full_volume=20.0,
    )

    node2_attributes = DrainageNodeAttributes(
        node_id="N2",
        node_type="junction",
        coupling_type=CouplingTypes.FREE_WEIR,
        coupling_flow=0.3,
        inflow=0.8,
        outflow=0.6,
        lateral_inflow=0.1,
        losses=0.05,
        overflow=0.0,
        depth=1.2,
        head=99.2,
        crest_elevation=98.0,
        invert_elevation=96.0,
        initial_depth=0.3,
        full_depth=1.8,
        surcharge_depth=0.4,
        ponding_area=8.0,
        volume=9.6,
        full_volume=14.4,
    )

    node3_attributes = DrainageNodeAttributes(
        node_id="N3",
        node_type="outfall",
        coupling_type=CouplingTypes.NOT_COUPLED,
        coupling_flow=0.0,
        inflow=0.6,
        outflow=0.6,
        lateral_inflow=0.0,
        losses=0.0,
        overflow=0.0,
        depth=0.8,
        head=96.8,
        crest_elevation=96.0,
        invert_elevation=94.0,
        initial_depth=0.2,
        full_depth=1.5,
        surcharge_depth=0.3,
        ponding_area=5.0,
        volume=4.0,
        full_volume=7.5,
    )

    # Create node data objects
    node1 = DrainageNodeData(coordinates=(100.0, 200.0), attributes=node1_attributes)

    node2 = DrainageNodeData(coordinates=(150.0, 180.0), attributes=node2_attributes)

    node3 = DrainageNodeData(coordinates=(200.0, 160.0), attributes=node3_attributes)

    # Create 2 drainage links with correct types from drainage.py
    link1_attributes = DrainageLinkAttributes(
        link_id="L1",
        link_type="conduit",  # Valid: conduit, pump, orifice, weir, outlet
        flow=0.8,
        depth=0.6,
        volume=12.0,
        inlet_offset=0.0,
        outlet_offset=0.0,
        froude=0.4,
    )

    link2_attributes = DrainageLinkAttributes(
        link_id="L2",
        link_type="conduit",
        flow=0.6,
        depth=0.5,
        volume=10.0,
        inlet_offset=0.0,
        outlet_offset=0.0,
        froude=0.3,
    )

    # Create link data objects with vertices connecting the nodes
    link1 = DrainageLinkData(
        vertices=((100.0, 200.0), (125.0, 190.0), (150.0, 180.0)),  # N1 to N2
        attributes=link1_attributes,
    )

    link2 = DrainageLinkData(
        vertices=((150.0, 180.0), (175.0, 170.0), (200.0, 160.0)),  # N2 to N3
        attributes=link2_attributes,
    )

    # Create the drainage network
    drainage_network = DrainageNetworkData(nodes=(node1, node2, node3), links=(link1, link2))

    return drainage_network


def create_dummy_drainage_network_no_geometry():
    """Create a dummy DrainageNetwork object with nodes and links without coordinates/vertices."""

    # Create 2 drainage nodes without coordinates
    node1_attributes = DrainageNodeAttributes(
        node_id="N1_no_coords",
        node_type="junction",
        coupling_type=CouplingTypes.ORIFICE,
        coupling_flow=0.5,
        inflow=1.0,
        outflow=0.8,
        lateral_inflow=0.2,
        losses=0.1,
        overflow=0.0,
        depth=1.5,
        head=101.5,
        crest_elevation=100.0,
        invert_elevation=98.0,
        initial_depth=0.5,
        full_depth=2.0,
        surcharge_depth=0.5,
        ponding_area=10.0,
        volume=15.0,
        full_volume=20.0,
    )

    node2_attributes = DrainageNodeAttributes(
        node_id="N2_no_coords",
        node_type="outfall",
        coupling_type=CouplingTypes.NOT_COUPLED,
        coupling_flow=0.0,
        inflow=0.6,
        outflow=0.6,
        lateral_inflow=0.0,
        losses=0.0,
        overflow=0.0,
        depth=0.8,
        head=96.8,
        crest_elevation=96.0,
        invert_elevation=94.0,
        initial_depth=0.2,
        full_depth=1.5,
        surcharge_depth=0.3,
        ponding_area=5.0,
        volume=4.0,
        full_volume=7.5,
    )

    # Create node data objects without coordinates
    node1 = DrainageNodeData(coordinates=None, attributes=node1_attributes)
    node2 = DrainageNodeData(coordinates=None, attributes=node2_attributes)

    # Create drainage link without vertices
    link1_attributes = DrainageLinkAttributes(
        link_id="L1_no_vertices",
        link_type="conduit",
        flow=0.8,
        depth=0.6,
        volume=12.0,
        inlet_offset=0.0,
        outlet_offset=0.0,
        froude=0.4,
    )

    # Create link data object without vertices
    link1 = DrainageLinkData(vertices=None, attributes=link1_attributes)

    # Create the drainage network
    drainage_network = DrainageNetworkData(nodes=(node1, node2), links=(link1,))

    return drainage_network


@pytest.fixture(scope="module")
def sim_time():
    return datetime(year=2020, month=3, day=23, hour=10)


@pytest.fixture(scope="module")
def write_sqlite(temp_dir, sim_time):
    """Write SQLite database using dummy network data."""
    drainage_network = create_dummy_drainage_network()

    provider_config = {
        "crs": pyproj.CRS.from_epsg(6372),
        "output_dir": Path(temp_dir.name),
        "drainage_map_name": "test_drainage",
    }
    sqlite_provider = SpatialiteVectorOutputProvider(provider_config)
    sqlite_provider.write_vector(drainage_network, sim_time)


@pytest.mark.usefixtures("write_sqlite")
def test_links(temp_dir, sim_time):
    """Verify SQLite database links table."""

    # Verify the database was created
    output_dir = Path(temp_dir.name)
    db_file = output_dir / "test_drainage.db"

    assert db_file.exists(), f"Database file not created: {db_file}"

    # Connect to database and query links
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Check that links table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='links'")
    assert cursor.fetchone() is not None, "Links table not found"

    # Query all links
    cursor.execute("SELECT * FROM links")
    rows = cursor.fetchall()

    assert len(rows) == 2, f"Expected 2 links, got {len(rows)}"

    # Get column names
    cursor.execute("PRAGMA table_info(links)")
    columns = [col[1] for col in cursor.fetchall()]

    # Check that sim_time column exists
    assert "sim_time" in columns, "sim_time column not found in links table"

    # Check that geometry column exists
    assert "geom" in columns or "geometry" in columns, "Geometry column not found in links table"

    # Check all link attribute fields are present
    expected_link_fields = [field.name for field in fields(DrainageLinkAttributes)]
    for field in expected_link_fields:
        assert field in columns, f"Missing link field: {field}"

    # Query links with specific data
    cursor.execute("SELECT link_id, link_type, flow, depth FROM links ORDER BY link_id")
    link_data = cursor.fetchall()

    # Verify link IDs
    link_ids = [row[0] for row in link_data]
    assert set(link_ids) == {"L1", "L2"}, f"Link IDs mismatch: {link_ids}"

    # Verify specific attribute values for L1
    l1_data = [row for row in link_data if row[0] == "L1"][0]
    assert l1_data[1] == "conduit", f"L1 link_type mismatch: {l1_data[1]}"
    assert l1_data[2] == 0.8, f"L1 flow mismatch: {l1_data[2]}"
    assert l1_data[3] == 0.6, f"L1 depth mismatch: {l1_data[3]}"

    conn.close()


@pytest.mark.usefixtures("write_sqlite")
def test_nodes(temp_dir, sim_time):
    """Verify SQLite database nodes table."""

    # Verify the database was created
    output_dir = Path(temp_dir.name)
    db_file = output_dir / "test_drainage.db"

    assert db_file.exists(), f"Database file not created: {db_file}"

    # Connect to database and query nodes
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Check that nodes table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='nodes'")
    assert cursor.fetchone() is not None, "Nodes table not found"

    # Query all nodes
    cursor.execute("SELECT * FROM nodes")
    rows = cursor.fetchall()

    assert len(rows) == 3, f"Expected 3 nodes, got {len(rows)}"

    # Get column names
    cursor.execute("PRAGMA table_info(nodes)")
    columns = [col[1] for col in cursor.fetchall()]

    # Check that sim_time column exists
    assert "sim_time" in columns, "sim_time column not found in nodes table"

    # Check that geometry column exists
    assert "geom" in columns or "geometry" in columns, "Geometry column not found in nodes table"

    # Check all node attribute fields are present
    expected_node_fields = [field.name for field in fields(DrainageNodeAttributes)]
    for field in expected_node_fields:
        assert field in columns, f"Missing node field: {field}"

    # Query nodes with specific data
    cursor.execute("SELECT node_id, node_type, depth, head FROM nodes ORDER BY node_id")
    node_data = cursor.fetchall()

    # Verify node IDs
    node_ids = [row[0] for row in node_data]
    assert set(node_ids) == {"N1", "N2", "N3"}, f"Node IDs mismatch: {node_ids}"

    # Verify specific attribute values for N1
    n1_data = [row for row in node_data if row[0] == "N1"][0]
    assert n1_data[1] == "junction", f"N1 node_type mismatch: {n1_data[1]}"
    assert n1_data[2] == 1.5, f"N1 depth mismatch: {n1_data[2]}"
    assert n1_data[3] == 101.5, f"N1 head mismatch: {n1_data[3]}"

    conn.close()


@pytest.mark.usefixtures("write_sqlite")
def test_geometry_points(temp_dir):
    """Verify node geometries are Points with correct coordinates."""

    output_dir = Path(temp_dir.name)
    db_file = output_dir / "test_drainage.db"

    conn = sqlite3.connect(db_file)
    # Enable spatialite extension
    conn.enable_load_extension(True)
    try:
        conn.load_extension("mod_spatialite")
    except Exception:
        # Try alternative loading method
        cursor = conn.cursor()
        cursor.execute("SELECT load_extension('mod_spatialite')")

    cursor = conn.cursor()

    # Get geometry column name
    cursor.execute("PRAGMA table_info(nodes)")
    columns = [col[1] for col in cursor.fetchall()]
    geom_col = "geom" if "geom" in columns else "geometry"

    # Query node geometries as WKT
    cursor.execute(f"SELECT node_id, AsText({geom_col}) FROM nodes ORDER BY node_id")
    geom_data = cursor.fetchall()

    # Expected coordinates
    expected_coords = {
        "N1": (100.0, 200.0),
        "N2": (150.0, 180.0),
        "N3": (200.0, 160.0),
    }

    for node_id, wkt in geom_data:
        assert wkt.startswith("POINT"), f"Node {node_id} geometry is not a POINT: {wkt}"
        # Extract coordinates from WKT
        coords_str = wkt.replace("POINT(", "").replace(")", "").strip()
        x, y = map(float, coords_str.split())
        expected_x, expected_y = expected_coords[node_id]
        assert abs(x - expected_x) < 0.001, (
            f"Node {node_id} X coordinate mismatch: {x} vs {expected_x}"
        )
        assert abs(y - expected_y) < 0.001, (
            f"Node {node_id} Y coordinate mismatch: {y} vs {expected_y}"
        )

    conn.close()


@pytest.mark.usefixtures("write_sqlite")
def test_geometry_linestrings(temp_dir):
    """Verify link geometries are LineStrings with correct vertices."""

    output_dir = Path(temp_dir.name)
    db_file = output_dir / "test_drainage.db"

    conn = sqlite3.connect(db_file)
    # Enable spatialite extension
    conn.enable_load_extension(True)
    try:
        conn.load_extension("mod_spatialite")
    except Exception:
        cursor = conn.cursor()
        cursor.execute("SELECT load_extension('mod_spatialite')")

    cursor = conn.cursor()

    # Get geometry column name
    cursor.execute("PRAGMA table_info(links)")
    columns = [col[1] for col in cursor.fetchall()]
    geom_col = "geom" if "geom" in columns else "geometry"

    # Query link geometries as WKT
    cursor.execute(f"SELECT link_id, AsText({geom_col}) FROM links ORDER BY link_id")
    geom_data = cursor.fetchall()

    # Expected vertices
    expected_vertices = {
        "L1": [(100.0, 200.0), (125.0, 190.0), (150.0, 180.0)],
        "L2": [(150.0, 180.0), (175.0, 170.0), (200.0, 160.0)],
    }

    for link_id, wkt in geom_data:
        assert wkt.startswith("LINESTRING"), f"Link {link_id} geometry is not a LINESTRING: {wkt}"
        # Extract coordinates from WKT
        coords_str = wkt.replace("LINESTRING(", "").replace(")", "").strip()
        coords_pairs = coords_str.split(",")
        actual_vertices = [tuple(map(float, pair.strip().split())) for pair in coords_pairs]
        expected = expected_vertices[link_id]

        assert len(actual_vertices) == len(expected), (
            f"Link {link_id} vertex count mismatch: {len(actual_vertices)} vs {len(expected)}"
        )

        for i, (actual, exp) in enumerate(zip(actual_vertices, expected)):
            assert abs(actual[0] - exp[0]) < 0.001, (
                f"Link {link_id} vertex {i} X mismatch: {actual[0]} vs {exp[0]}"
            )
            assert abs(actual[1] - exp[1]) < 0.001, (
                f"Link {link_id} vertex {i} Y mismatch: {actual[1]} vs {exp[1]}"
            )

    conn.close()


@pytest.fixture(scope="module")
def write_sqlite_no_geometry(temp_dir, sim_time):
    """Write SQLite database using dummy network data without geometry."""
    drainage_network = create_dummy_drainage_network_no_geometry()

    provider_config = {
        "crs": pyproj.CRS.from_epsg(6372),
        "output_dir": Path(temp_dir.name),
        "drainage_map_name": "test_drainage_no_geom",
    }
    sqlite_provider = SpatialiteVectorOutputProvider(provider_config)
    sqlite_provider.write_vector(drainage_network, sim_time)


@pytest.mark.usefixtures("write_sqlite_no_geometry")
def test_nodes_without_coordinates(temp_dir, sim_time):
    """Verify SQLite database nodes without coordinates."""

    output_dir = Path(temp_dir.name)
    db_file = output_dir / "test_drainage_no_geom.db"

    assert db_file.exists(), f"Database file not created: {db_file}"

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Query all nodes
    cursor.execute("SELECT * FROM nodes")
    rows = cursor.fetchall()

    assert len(rows) == 2, f"Expected 2 nodes, got {len(rows)}"

    # Get column names
    cursor.execute("PRAGMA table_info(nodes)")
    columns = [col[1] for col in cursor.fetchall()]

    # Check all node attribute fields are present
    expected_node_fields = [field.name for field in fields(DrainageNodeAttributes)]
    for field in expected_node_fields:
        assert field in columns, f"Missing node field: {field}"

    # Query nodes with specific data
    cursor.execute("SELECT node_id, node_type, coupling_flow, depth FROM nodes ORDER BY node_id")
    node_data = cursor.fetchall()

    # Verify node IDs
    node_ids = [row[0] for row in node_data]
    assert set(node_ids) == {"N1_no_coords", "N2_no_coords"}, f"Node IDs mismatch: {node_ids}"

    # Verify specific attribute values for N1_no_coords
    n1_data = [row for row in node_data if row[0] == "N1_no_coords"][0]
    assert n1_data[1] == "junction", f"N1 node_type mismatch: {n1_data[1]}"
    assert n1_data[2] == 0.5, f"N1 coupling_flow mismatch: {n1_data[2]}"
    assert n1_data[3] == 1.5, f"N1 depth mismatch: {n1_data[3]}"

    conn.close()


@pytest.mark.usefixtures("write_sqlite_no_geometry")
def test_links_without_vertices(temp_dir, sim_time):
    """Verify SQLite database links without vertices."""

    output_dir = Path(temp_dir.name)
    db_file = output_dir / "test_drainage_no_geom.db"

    assert db_file.exists(), f"Database file not created: {db_file}"

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Query all links
    cursor.execute("SELECT * FROM links")
    rows = cursor.fetchall()

    assert len(rows) == 1, f"Expected 1 link, got {len(rows)}"

    # Get column names
    cursor.execute("PRAGMA table_info(links)")
    columns = [col[1] for col in cursor.fetchall()]

    # Check all link attribute fields are present
    expected_link_fields = [field.name for field in fields(DrainageLinkAttributes)]
    for field in expected_link_fields:
        assert field in columns, f"Missing link field: {field}"

    # Query links with specific data
    cursor.execute("SELECT link_id, link_type, flow, depth FROM links")
    link_data = cursor.fetchall()

    # Verify link IDs
    link_ids = [row[0] for row in link_data]
    assert set(link_ids) == {"L1_no_vertices"}, f"Link IDs mismatch: {link_ids}"

    # Verify specific attribute values for L1_no_vertices
    l1_data = link_data[0]
    assert l1_data[1] == "conduit", f"L1 link_type mismatch: {l1_data[1]}"
    assert l1_data[2] == 0.8, f"L1 flow mismatch: {l1_data[2]}"
    assert l1_data[3] == 0.6, f"L1 depth mismatch: {l1_data[3]}"

    conn.close()


@pytest.fixture(scope="module")
def relative_time():
    """Fixture for relative time in seconds (ISO 8601 duration)."""
    return timedelta(seconds=3600)


@pytest.fixture(scope="module")
def write_sqlite_relative_time(temp_dir, relative_time):
    """Write SQLite database using dummy network data with relative time."""
    drainage_network = create_dummy_drainage_network()

    provider_config = {
        "crs": pyproj.CRS.from_epsg(6372),
        "output_dir": Path(temp_dir.name),
        "drainage_map_name": "test_drainage_relative",
    }
    sqlite_provider = SpatialiteVectorOutputProvider(provider_config)
    sqlite_provider.write_vector(drainage_network, relative_time)


@pytest.mark.usefixtures("write_sqlite_relative_time")
def test_relative_time_storage(temp_dir, relative_time):
    """Verify SQLite database stores relative time correctly as ISO duration in seconds."""

    output_dir = Path(temp_dir.name)
    db_file = output_dir / "test_drainage_relative.db"

    assert db_file.exists(), f"Database file not created: {db_file}"

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Check nodes table
    cursor.execute("PRAGMA table_info(nodes)")
    columns = [col[1] for col in cursor.fetchall()]
    assert "sim_time" in columns, "sim_time column not found in nodes table"

    # Query sim_time from nodes
    cursor.execute("SELECT DISTINCT sim_time FROM nodes")
    node_times = cursor.fetchall()
    assert len(node_times) == 1, f"Expected 1 unique sim_time, got {len(node_times)}"

    # Verify the relative time is stored as an integer (seconds)
    stored_time = node_times[0][0]
    assert isinstance(stored_time, str), f"sim_time should be string, got {type(stored_time)}"
    expected_iso_duration = f"PT{relative_time.total_seconds()}S"
    assert stored_time == expected_iso_duration, (
        f"sim_time mismatch: {stored_time} vs expected {expected_iso_duration}"
    )

    # Check links table
    cursor.execute("SELECT DISTINCT sim_time FROM links")
    link_times = cursor.fetchall()
    assert len(link_times) == 1, f"Expected 1 unique sim_time, got {len(link_times)}"

    stored_time_links = link_times[0][0]
    assert stored_time_links == expected_iso_duration, (
        f"Links sim_time mismatch: {stored_time_links} vs expected {expected_iso_duration}"
    )

    conn.close()
