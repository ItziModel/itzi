"""Test the drainage component."""

import os
from datetime import timedelta

import pandas as pd
import pytest
import pyswmm

from itzi import SwmmInputParser
from itzi.drainage import DrainageNode, CouplingTypes, DrainageSimulation
from itzi.simulation_factories import get_links_list


@pytest.fixture(scope="class")
def drainage_sim_results(test_data_path):
    # SWMM config file based on EA test 8b
    inp_file = os.path.join(test_data_path, "test_drainage.inp")
    # from input file
    coupling_node_id = "J1"
    # Dummy surface state
    surface_states = {}
    surface_states[coupling_node_id] = {"z": 100, "h": 0.0}
    cell_surf = 25.0
    # Create simulation
    swmm_sim = pyswmm.Simulation(inp_file)
    swmm_inp = SwmmInputParser(inp_file)
    sim_start_time = swmm_sim.start_time
    sim_end_time = swmm_sim.end_time
    # Create Node objects
    pyswmm_nodes = pyswmm.Nodes(swmm_sim)
    nodes_coors_dict = swmm_inp.get_nodes_id_as_dict()
    pyswmm_node = pyswmm_nodes[coupling_node_id]
    coupling_node = DrainageNode(
        pyswmm_node, nodes_coors_dict[coupling_node_id], CouplingTypes.COUPLED_NO_FLOW
    )
    nodes_list = [coupling_node]
    # Create Link objects
    links_vertices_dict = swmm_inp.get_links_id_as_dict()
    links_list = get_links_list(pyswmm.Links(swmm_sim), links_vertices_dict, nodes_coors_dict)
    # Create simulation object
    drainage = DrainageSimulation(swmm_sim, nodes_list, links_list)

    # Run simulation
    coupling_flows = {}
    current_time = sim_start_time
    # Add fudge
    while current_time < sim_end_time - timedelta(milliseconds=500):
        # update simulation
        drainage.step()
        # coupling
        calculated_flows = drainage.apply_coupling_to_nodes(surface_states, cell_surf)
        node_flow = calculated_flows[coupling_node_id]
        coupling_flows[current_time - sim_start_time] = node_flow
        # update time
        current_time += drainage.dt

    ds_results = pd.Series(coupling_flows)
    # things start happening after 3000 seconds
    ds_results = ds_results[ds_results.index >= timedelta(seconds=3000)]
    return ds_results


def test_drainage_coupling_stability(drainage_sim_results, helpers):
    """Test the stability of the drainage coupling."""
    # roughness = helpers.roughness(drainage_sim_results)
    # assert roughness < 5
    autocorrelation = drainage_sim_results.autocorr(lag=1)
    assert autocorrelation > 0.9
