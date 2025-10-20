from datetime import datetime, timedelta
from pathlib import Path
import dataclasses

import numpy as np
import pytest
import pyproj
import obstore
import pandas as pd

from itzi.data_containers import (
    DrainageNodeAttributes,
    DrainageLinkAttributes,
    DrainageNodeData,
    DrainageLinkData,
    DrainageNetworkData,
)
from itzi.providers.csv_output import CSVVectorOutputProvider
from itzi.drainage import CouplingTypes

from tests.fixtures_vector_output import create_dummy_drainage_network
from tests.fixtures_vector_output import expected_node_coords, expected_vertices

pytest_plugins = ["tests.fixtures_vector_output"]


@pytest.fixture(scope="module")
def write_csv(test_data_temp_path, sim_time):
    """Write cloud csv using dummy network data."""
    drainage_network = create_dummy_drainage_network()

    provider_config = {
        "crs": pyproj.CRS.from_epsg(6372),
        "store": obstore.store.LocalStore(),
        "results_prefix": test_data_temp_path,
        "drainage_results_name": "test_drainage",
        "overwrite": True,
    }
    csv_provider = CSVVectorOutputProvider(provider_config)
    csv_provider.write_vector(drainage_network, sim_time)
    csv_provider.write_vector(drainage_network, sim_time + timedelta(seconds=60))


@pytest.fixture(scope="module")
def write_csv_no_geom(test_data_temp_path, sim_time):
    """Write cloud csv using dummy network data without geometry."""
    drainage_network = create_dummy_drainage_network(with_coords=False)

    provider_config = {
        "crs": pyproj.CRS.from_epsg(6372),
        "store": obstore.store.LocalStore(),
        "results_prefix": test_data_temp_path,
        "drainage_results_name": "test_drainage_no_geom",
        "overwrite": True,
    }
    csv_provider = CSVVectorOutputProvider(provider_config)
    csv_provider.write_vector(drainage_network, sim_time)
    csv_provider.write_vector(drainage_network, sim_time + timedelta(seconds=60))


@pytest.mark.usefixtures("write_csv")
def test_csv(test_data_temp_path, sim_time):
    """Verify CSV vector files."""

    links_file = Path(test_data_temp_path) / Path("test_drainage_links.csv")
    nodes_file = Path(test_data_temp_path) / Path("test_drainage_nodes.csv")
    assert links_file.exists(), f"CSV file not created: {links_file}"
    assert nodes_file.exists(), f"CSV file not created: {nodes_file}"

    df_links = pd.read_csv(links_file)
    df_nodes = pd.read_csv(nodes_file)

    for df in [df_links, df_nodes]:
        _validate_common_fields(df, sim_time)

    # nodes
    _validate_nodes_attributes(df_nodes)
    _validate_nodes_geometries(df_nodes)
    # links
    _validate_links_attributes(df_links)
    _validate_links_geometries(df_links)


@pytest.mark.usefixtures("write_csv_no_geom")
def test_csv_no_geom(test_data_temp_path, sim_time):
    """Verify CSV vector files without geometries."""

    links_file = Path(test_data_temp_path) / Path("test_drainage_no_geom_links.csv")
    nodes_file = Path(test_data_temp_path) / Path("test_drainage_no_geom_nodes.csv")
    assert links_file.exists(), f"CSV file not created: {links_file}"
    assert nodes_file.exists(), f"CSV file not created: {nodes_file}"

    df_links = pd.read_csv(links_file)
    df_nodes = pd.read_csv(nodes_file)

    for df in [df_links, df_nodes]:
        _validate_common_fields(df, sim_time)

    # nodes
    _validate_nodes_attributes(df_nodes)
    # links
    _validate_links_attributes(df_links)


def _validate_common_fields(df, sim_time):
    # Check that sim_time, srid, and geometry columns exist
    assert "sim_time" in df.columns, "sim_time column not found in links table"
    assert "srid" in df.columns, "srid column not found in links table"
    assert "geometry" in df.columns, "Geometry column not found in links table"
    # check sim_time
    if isinstance(sim_time, timedelta):
        df["sim_time"] = pd.to_timedelta(df["sim_time"])
    else:
        df["sim_time"] = pd.to_datetime(df["sim_time"])
    assert sim_time == df["sim_time"][0], f"Expected {sim_time}, got {df['sim_time'][0]}"
    assert 6372 == df["srid"][0], f"Expected {6372}, got {df['srid'][0]}"


def _validate_nodes_attributes(df_nodes):
    # check number of records
    expected_times_num = 2
    expected_node_num = len(expected_node_coords)
    expected_total_records = expected_times_num * expected_node_num
    assert len(df_nodes) == expected_total_records, (
        f"Expected {expected_total_records} node records, got {len(df_nodes)}"
    )
    assert len(set(df_nodes["node_id"])) == expected_node_num, (
        f"Expected {expected_node_num} individual nodes, got {len(set(df_nodes['node_id']))}"
    )
    assert len(set(df_nodes["sim_time"])) == expected_times_num, (
        f"Expected {expected_times_num} individual times, got {len(set(df_nodes['sim_time']))}"
    )
    # Check all attribute fields are present
    expected_node_fields = [field.name for field in dataclasses.fields(DrainageNodeAttributes)]
    for field in expected_node_fields:
        assert field in df_nodes.columns, f"Missing node field: {field}"
    # Verify IDs
    node_ids = df_nodes["node_id"]
    assert set(node_ids) == set(expected_node_coords.keys()), f"Node IDs mismatch: {node_ids}"
    # Verify specific attribute values for N1
    n1_data = df_nodes[df_nodes["node_id"] == "N1"]
    print(n1_data)
    assert n1_data["node_type"][0] == "junction", f"N1 node_type mismatch: {n1_data[1]}"
    assert n1_data["depth"][0] == 1.5, f"N1 depth mismatch: {n1_data[2]}"
    assert n1_data["head"][0] == 101.5, f"N1 head mismatch: {n1_data[3]}"


def _validate_nodes_geometries(df_nodes):
    for node_row in df_nodes.loc[:, ["node_id", "geometry"]].iterrows():
        node_id = node_row[1][0]
        geom_wkt = node_row[1][1]
        assert geom_wkt.startswith("POINT"), f"Node {node_id} geometry is not a POINT: {geom_wkt}"
        coords_str = geom_wkt.replace("POINT(", "").replace(")", "").strip()
        x, y = map(float, coords_str.split())
        expected_x, expected_y = expected_node_coords[node_id]
        assert np.isclose(x, expected_x), (
            f"Node {node_id} X coordinate mismatch: {x} vs {expected_x}"
        )
        assert np.isclose(y, expected_y), (
            f"Node {node_id} Y coordinate mismatch: {y} vs {expected_y}"
        )


def _validate_links_attributes(df_links):
    expected_times_num = 2
    expected_link_num = len(expected_vertices)
    expected_total_records = expected_times_num * expected_link_num
    assert len(df_links) == expected_total_records, (
        f"Expected {expected_total_records} link records, got {len(df_links)}"
    )
    assert len(set(df_links["link_id"])) == expected_link_num, (
        f"Expected {expected_link_num} individual links, got {len(set(df_links['link_id']))}"
    )
    assert len(set(df_links["sim_time"])) == expected_times_num, (
        f"Expected {expected_times_num} individual times, got {len(set(df_links['sim_time']))}"
    )
    # Check all link attribute fields are present
    expected_link_fields = [field.name for field in dataclasses.fields(DrainageLinkAttributes)]
    for field in expected_link_fields:
        assert field in df_links.columns, f"Missing link field: {field}"
    # Verify link IDs
    link_ids = df_links["link_id"]
    assert set(link_ids) == {"L1", "L2"}, f"Link IDs mismatch: {link_ids}"
    # Verify specific attribute values for L1
    l1_data = df_links["L1" == df_links["link_id"]]
    assert l1_data["link_type"][0] == "conduit", f"L1  mismatch: {l1_data['link_type']}"
    assert l1_data["flow"][0] == 0.8, f"L1 flow mismatch: {l1_data['flow']}"
    assert l1_data["depth"][0] == 0.6, f"L1 depth mismatch: {l1_data['depth']}"


def _validate_links_geometries(df_links):
    for link_row in df_links.loc[:, ["link_id", "geometry"]].iterrows():
        link_id = link_row[1][0]
        wkt = link_row[1][1]
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
            assert np.isclose(actual[0], exp[0]), (
                f"Link {link_id} vertex {i} X mismatch: {actual[0]} vs {exp[0]}"
            )
            assert np.isclose(actual[1], exp[1]), (
                f"Link {link_id} vertex {i} Y mismatch: {actual[1]} vs {exp[1]}"
            )


def test_append_success(test_data_temp_path):
    """Verify successful append mode when conditions are met."""
    drainage_network = create_dummy_drainage_network()
    sim_time_1 = timedelta(seconds=0)
    sim_time_2 = timedelta(seconds=60)
    sim_time_3 = timedelta(seconds=120)

    provider_config = {
        "crs": pyproj.CRS.from_epsg(6372),
        "store": obstore.store.LocalStore(),
        "results_prefix": test_data_temp_path,
        "drainage_results_name": "test_append_success",
        "overwrite": True,
    }

    # Write initial data
    csv_provider = CSVVectorOutputProvider(provider_config)
    csv_provider.write_vector(drainage_network, sim_time_1)
    csv_provider.write_vector(drainage_network, sim_time_2)

    # Switch to append mode and write additional data
    provider_config["overwrite"] = False
    csv_provider_append = CSVVectorOutputProvider(provider_config)
    csv_provider_append.write_vector(drainage_network, sim_time_3)

    # Verify all three time steps are present
    nodes_file = Path(test_data_temp_path) / Path("test_append_success_nodes.csv")
    df_nodes = pd.read_csv(nodes_file)

    assert len(set(df_nodes["sim_time"])) == 3, (
        f"Expected 3 time steps after append, got {len(set(df_nodes['sim_time']))}"
    )
    expected_times_num = 3
    expected_node_num = len(expected_node_coords)
    expected_total_records = expected_times_num * expected_node_num
    assert len(df_nodes) == expected_total_records, (
        f"Expected {expected_total_records} records after append, got {len(df_nodes)}"
    )


def test_append_column_mismatch_nodes_to_links(test_data_temp_path):
    """Verify ValueError when trying to append nodes data to links file."""
    drainage_network = create_dummy_drainage_network()
    sim_time = timedelta(seconds=0)

    # First, write links data
    provider_config = {
        "crs": pyproj.CRS.from_epsg(6372),
        "store": obstore.store.LocalStore(),
        "results_prefix": test_data_temp_path,
        "drainage_results_name": "test_column_mismatch",
        "overwrite": True,
    }
    csv_provider = CSVVectorOutputProvider(provider_config)
    csv_provider.write_vector(drainage_network, sim_time)

    # Manually swap the files to simulate column mismatch
    nodes_file = Path(test_data_temp_path) / Path("test_column_mismatch_nodes.csv")
    links_file = Path(test_data_temp_path) / Path("test_column_mismatch_links.csv")
    # Overwrite nodes file with links content
    nodes_file.write_text(links_file.read_text())

    # Try to append - should fail because nodes file now has links columns
    provider_config["overwrite"] = False
    with pytest.raises(ValueError, match="column.*mismatch|header.*mismatch|columns.*match"):
        csv_provider_append = CSVVectorOutputProvider(provider_config)
        csv_provider_append.write_vector(drainage_network, timedelta(seconds=60))


def test_append_time_type_mismatch_timedelta_to_datetime(test_data_temp_path):
    """Verify ValueError when appending datetime to file with timedelta."""
    drainage_network = create_dummy_drainage_network()

    # Write with timedelta
    provider_config = {
        "crs": pyproj.CRS.from_epsg(6372),
        "store": obstore.store.LocalStore(),
        "results_prefix": test_data_temp_path,
        "drainage_results_name": "test_time_type_mismatch_td",
        "overwrite": True,
    }
    csv_provider = CSVVectorOutputProvider(provider_config)
    csv_provider.write_vector(drainage_network, timedelta(seconds=0))

    # Try to append with datetime
    provider_config["overwrite"] = False
    csv_provider_append = CSVVectorOutputProvider(provider_config)

    with pytest.raises(ValueError, match="time.*type|type.*mismatch"):
        csv_provider_append.write_vector(
            drainage_network, datetime(year=2020, month=3, day=23, hour=10)
        )


def test_append_time_type_mismatch_datetime_to_timedelta(test_data_temp_path):
    """Verify ValueError when appending timedelta to file with datetime."""
    drainage_network = create_dummy_drainage_network()

    # Write with datetime
    provider_config = {
        "crs": pyproj.CRS.from_epsg(6372),
        "store": obstore.store.LocalStore(),
        "results_prefix": test_data_temp_path,
        "drainage_results_name": "test_time_type_mismatch_dt",
        "overwrite": True,
    }
    csv_provider = CSVVectorOutputProvider(provider_config)
    csv_provider.write_vector(drainage_network, datetime(year=2020, month=3, day=23, hour=10))

    # Try to append with timedelta
    provider_config["overwrite"] = False
    csv_provider_append = CSVVectorOutputProvider(provider_config)

    with pytest.raises(ValueError, match="time.*type|type.*mismatch"):
        csv_provider_append.write_vector(drainage_network, timedelta(seconds=60))


def test_append_node_ids_mismatch(test_data_temp_path):
    """Verify ValueError when appending with different node IDs."""
    drainage_network = create_dummy_drainage_network()
    sim_time = timedelta(seconds=0)

    # Write initial data
    provider_config = {
        "crs": pyproj.CRS.from_epsg(6372),
        "store": obstore.store.LocalStore(),
        "results_prefix": test_data_temp_path,
        "drainage_results_name": "test_node_ids_mismatch",
        "overwrite": True,
    }
    csv_provider = CSVVectorOutputProvider(provider_config)
    csv_provider.write_vector(drainage_network, sim_time)

    # Create network with different node IDs
    node_new_attributes = DrainageNodeAttributes(
        node_id="N_DIFFERENT",
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
    node_new = DrainageNodeData(coordinates=(100.0, 200.0), attributes=node_new_attributes)

    # Get existing nodes but replace first one
    modified_network = DrainageNetworkData(
        nodes=(node_new, drainage_network.nodes[1], drainage_network.nodes[2]),
        links=drainage_network.links,
    )

    # Try to append with different node IDs
    provider_config["overwrite"] = False
    csv_provider_append = CSVVectorOutputProvider(provider_config)

    with pytest.raises(ValueError, match="[Oo]bject.*ids.*mismatch"):
        csv_provider_append.write_vector(modified_network, timedelta(seconds=60))


def test_append_link_ids_mismatch(test_data_temp_path):
    """Verify ValueError when appending with different link IDs."""
    drainage_network = create_dummy_drainage_network()
    sim_time = timedelta(seconds=0)

    # Write initial data
    provider_config = {
        "crs": pyproj.CRS.from_epsg(6372),
        "store": obstore.store.LocalStore(),
        "results_prefix": test_data_temp_path,
        "drainage_results_name": "test_link_ids_mismatch",
        "overwrite": True,
    }
    csv_provider = CSVVectorOutputProvider(provider_config)
    csv_provider.write_vector(drainage_network, sim_time)

    # Create network with different link IDs
    link_new_attributes = DrainageLinkAttributes(
        link_id="L_DIFFERENT",
        link_type="conduit",
        flow=0.8,
        depth=0.6,
        volume=12.0,
        inlet_offset=0.0,
        outlet_offset=0.0,
        froude=0.4,
    )
    link_new = DrainageLinkData(
        vertices=[(100.0, 200.0), (125.0, 190.0), (150.0, 180.0)],
        attributes=link_new_attributes,
    )

    # Get existing network but replace first link
    existing_network = create_dummy_drainage_network()
    modified_network = DrainageNetworkData(
        nodes=existing_network.nodes, links=(link_new, existing_network.links[1])
    )

    # Try to append with different link IDs
    provider_config["overwrite"] = False
    csv_provider_append = CSVVectorOutputProvider(provider_config)

    with pytest.raises(ValueError, match="[Oo]bject.*ids.*mismatch"):
        csv_provider_append.write_vector(modified_network, timedelta(seconds=60))


def test_append_time_not_increasing(test_data_temp_path):
    """Verify ValueError when appending with time < maximum existing time."""
    drainage_network = create_dummy_drainage_network()

    # Write initial data with two time steps
    provider_config = {
        "crs": pyproj.CRS.from_epsg(6372),
        "store": obstore.store.LocalStore(),
        "results_prefix": test_data_temp_path,
        "drainage_results_name": "test_time_not_increasing",
        "overwrite": True,
    }
    csv_provider = CSVVectorOutputProvider(provider_config)
    csv_provider.write_vector(drainage_network, timedelta(seconds=0))
    csv_provider.write_vector(drainage_network, timedelta(seconds=120))

    # Try to append with time before the maximum
    provider_config["overwrite"] = False
    csv_provider_append = CSVVectorOutputProvider(provider_config)

    with pytest.raises(ValueError, match="Time not increasing"):
        csv_provider_append.write_vector(drainage_network, timedelta(seconds=60))


def test_append_time_equal_to_maximum(test_data_temp_path):
    """Verify ValueError when appending with time equal to maximum existing time."""
    drainage_network = create_dummy_drainage_network()

    # Write initial data
    provider_config = {
        "crs": pyproj.CRS.from_epsg(6372),
        "store": obstore.store.LocalStore(),
        "results_prefix": test_data_temp_path,
        "drainage_results_name": "test_time_equal",
        "overwrite": True,
    }
    csv_provider = CSVVectorOutputProvider(provider_config)
    csv_provider.write_vector(drainage_network, timedelta(seconds=60))

    # Try to append with same time
    provider_config["overwrite"] = False
    csv_provider_append = CSVVectorOutputProvider(provider_config)

    with pytest.raises(ValueError, match="Time not increasing"):
        csv_provider_append.write_vector(drainage_network, timedelta(seconds=60))
