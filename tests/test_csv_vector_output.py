import tempfile
from datetime import datetime
from pathlib import Path
import dataclasses

import pytest
import pyproj
import obstore
import pandas as pd

from itzi.data_containers import (
    DrainageNetworkData,
    DrainageNodeData,
    DrainageNodeAttributes,
    DrainageLinkData,
    DrainageLinkAttributes,
)
from itzi.drainage import CouplingTypes
from itzi.providers.csv_output import CSVVectorOutputProvider


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
def write_csv(temp_dir, sim_time):
    """Write cloud csv using dummy network data."""
    drainage_network = create_dummy_drainage_network()

    provider_config = {
        "crs": pyproj.CRS.from_epsg(6372),
        "store": obstore.store.LocalStore(),
        "results_prefix": temp_dir.name,
        "drainage_results_name": "test_drainage",
    }
    csv_provider = CSVVectorOutputProvider(provider_config)
    csv_provider.write_vector(drainage_network, sim_time)


@pytest.mark.usefixtures("write_csv")
def test_links(temp_dir, sim_time):
    """Verify CSV links file."""

    links_file = Path(temp_dir.name) / Path("test_drainage_links.csv")
    assert links_file.exists(), f"Database file not created: {links_file}"

    df_links = pd.read_csv(links_file)

    # Check that sim_time, srid, and geometry columns exist
    assert "sim_time" in df_links.columns, "sim_time column not found in links table"
    assert "srid" in df_links.columns, "srid column not found in links table"
    assert "geometry" in df_links.columns, "Geometry column not found in links table"

    assert len(df_links) == 2, f"Expected 2 links, got {len(df_links)}"

    # Check all link attribute fields are present
    expected_link_fields = [field.name for field in dataclasses.fields(DrainageLinkAttributes)]
    for field in expected_link_fields:
        assert field in df_links.columns, f"Missing link field: {field}"
    # Verify link IDs
    link_ids = df_links["link_id"]
    assert set(link_ids) == {"L1", "L2"}, f"Link IDs mismatch: {link_ids}"
    print(df_links)

    assert False


@pytest.mark.usefixtures("write_csv")
def test_nodes(temp_dir, sim_time):
    """Verify CSV nodes file."""
    nodes_file = Path(temp_dir.name) / Path("test_drainage_nodes.csv")
    assert nodes_file.exists(), f"Database file not created: {nodes_file}"
    df_nodes = pd.read_csv(nodes_file)
    assert len(df_nodes) == 3, f"Expected 3 nodes, got {len(df_nodes)}"

    print(df_nodes)

    assert False
