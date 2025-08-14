import tempfile
from datetime import datetime

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
    """Test creating a dummy drainage network."""
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
