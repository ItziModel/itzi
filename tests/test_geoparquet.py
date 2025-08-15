import tempfile
from datetime import datetime
from pathlib import Path
from dataclasses import fields

import pytest
import pyproj
import geopandas as gpd
from shapely.geometry import Point, LineString

from itzi.data_containers import (
    DrainageNetworkData,
    DrainageNodeData,
    DrainageNodeAttributes,
    DrainageLinkData,
    DrainageLinkAttributes,
)
from itzi.drainage import CouplingTypes
from itzi.providers.geoparquet_output import ParquetVectorOutputProvider


@pytest.fixture(scope="function")
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


def test_geoparquet(temp_dir):
    """Test creating a dummy drainage network and verifying parquet output."""
    drainage_network = create_dummy_drainage_network()

    parquet_provider = ParquetVectorOutputProvider()
    provider_config = {
        "crs": pyproj.CRS.from_epsg(6372),
        "output_dir": temp_dir.name,
        "drainage_map_name": "test_drainage",
    }
    sim_time = datetime(year=2020, month=3, day=23, hour=10)
    parquet_provider.initialize(provider_config)
    parquet_provider.write_vector(drainage_network, sim_time)

    # Verify the files were created
    output_dir = Path(temp_dir.name)
    nodes_file = output_dir / f"test_drainage_nodes_{sim_time}.parquet"
    links_file = output_dir / f"test_drainage_links_{sim_time}.parquet"

    assert nodes_file.exists(), f"Nodes file not created: {nodes_file}"
    assert links_file.exists(), f"Links file not created: {links_file}"

    # Read the parquet files with geopandas
    nodes_gdf = gpd.read_parquet(nodes_file)
    links_gdf = gpd.read_parquet(links_file)

    # Verify nodes data
    assert len(nodes_gdf) == 3, f"Expected 3 nodes, got {len(nodes_gdf)}"
    assert nodes_gdf.crs == pyproj.CRS.from_epsg(6372), f"CRS mismatch for nodes: {nodes_gdf.crs}"

    # Check node geometries are Points
    assert all(isinstance(geom, Point) for geom in nodes_gdf.geometry), (
        "Not all node geometries are Points"
    )

    # Check node coordinates match original data
    expected_node_coords = [(100.0, 200.0), (150.0, 180.0), (200.0, 160.0)]
    actual_node_coords = [(geom.x, geom.y) for geom in nodes_gdf.geometry]
    assert actual_node_coords == expected_node_coords, (
        f"Node coordinates mismatch: {actual_node_coords}"
    )

    # Check all node attribute fields are present
    expected_node_fields = [field.name for field in fields(DrainageNodeAttributes)]
    for field in expected_node_fields:
        assert field in nodes_gdf.columns, f"Missing node field: {field}"

    # Check specific node values
    node_ids = nodes_gdf["node_id"].tolist()
    assert set(node_ids) == {"N1", "N2", "N3"}, f"Node IDs mismatch: {node_ids}"

    # Verify links data
    assert len(links_gdf) == 2, f"Expected 2 links, got {len(links_gdf)}"
    assert links_gdf.crs == pyproj.CRS.from_epsg(6372), f"CRS mismatch for links: {links_gdf.crs}"

    # Check link geometries are LineStrings
    assert all(isinstance(geom, LineString) for geom in links_gdf.geometry), (
        "Not all link geometries are LineStrings"
    )

    # Check link coordinates match original data
    expected_link1_coords = [(100.0, 200.0), (125.0, 190.0), (150.0, 180.0)]
    expected_link2_coords = [(150.0, 180.0), (175.0, 170.0), (200.0, 160.0)]

    # Sort by link_id to ensure consistent ordering
    links_sorted = links_gdf.sort_values("link_id")
    link1_geom = links_sorted[links_sorted["link_id"] == "L1"].geometry.iloc[0]
    link2_geom = links_sorted[links_sorted["link_id"] == "L2"].geometry.iloc[0]

    assert list(link1_geom.coords) == expected_link1_coords, (
        f"Link1 coordinates mismatch: {list(link1_geom.coords)}"
    )
    assert list(link2_geom.coords) == expected_link2_coords, (
        f"Link2 coordinates mismatch: {list(link2_geom.coords)}"
    )

    # Check all link attribute fields are present
    expected_link_fields = [field.name for field in fields(DrainageLinkAttributes)]
    for field in expected_link_fields:
        assert field in links_gdf.columns, f"Missing link field: {field}"

    # Check specific link values
    link_ids = links_gdf["link_id"].tolist()
    assert set(link_ids) == {"L1", "L2"}, f"Link IDs mismatch: {link_ids}"

    # Verify specific attribute values for nodes
    n1_row = nodes_gdf[nodes_gdf["node_id"] == "N1"].iloc[0]
    assert n1_row["node_type"] == "junction", f"N1 node_type mismatch: {n1_row['node_type']}"
    assert n1_row["coupling_flow"] == 0.5, f"N1 coupling_flow mismatch: {n1_row['coupling_flow']}"
    assert n1_row["depth"] == 1.5, f"N1 depth mismatch: {n1_row['depth']}"

    # Verify specific attribute values for links
    l1_row = links_gdf[links_gdf["link_id"] == "L1"].iloc[0]
    assert l1_row["link_type"] == "conduit", f"L1 link_type mismatch: {l1_row['link_type']}"
    assert l1_row["flow"] == 0.8, f"L1 flow mismatch: {l1_row['flow']}"
    assert l1_row["depth"] == 0.6, f"L1 depth mismatch: {l1_row['depth']}"

    print("All geoparquet tests passed successfully!")
