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

from typing import Dict

import numpy as np
import pyswmm

from itzi.surfaceflow import SurfaceFlowSimulation
import itzi.rasterdomain as rasterdomain
from itzi.massbalance import MassBalanceLogger
from itzi.report import Report
from itzi.drainage import DrainageSimulation, DrainageNode, DrainageLink, CouplingTypes
from itzi.swmm_input_parser import SwmmInputParser
import itzi.messenger as msgr
import itzi.infiltration as infiltration
from itzi.hydrology import Hydrology
from itzi.simulation import Simulation
from itzi.data_containers import DrainageNodeCouplingData


def get_nodes_list(
    pswmm_nodes, nodes_coor_dict, drainage_params, g_interface, g
) -> list[DrainageNodeCouplingData]:
    """Check if the drainage nodes are inside the region and can be coupled.
    Return a list of DrainageNodeCouplingData
    """
    nodes_list = []
    for pyswmm_node in pswmm_nodes:
        coors = nodes_coor_dict[pyswmm_node.nodeid]
        node = DrainageNode(
            node_object=pyswmm_node,
            coordinates=coors,
            coupling_type=CouplingTypes.NOT_COUPLED,
            orifice_coeff=drainage_params["orifice_coeff"],
            free_weir_coeff=drainage_params["free_weir_coeff"],
            submerged_weir_coeff=drainage_params["submerged_weir_coeff"],
            g=g,
        )
        # a node without coordinates cannot be coupled
        if coors is None or not g_interface.is_in_region(coors.x, coors.y):
            x_coor = None
            y_coor = None
            row = None
            col = None
        else:
            # Set node as coupled with no flow
            node.coupling_type = CouplingTypes.COUPLED_NO_FLOW
            x_coor = coors.x
            y_coor = coors.y
            row, col = g_interface.coor2pixel(coors)
        # populate list
        drainage_node_data = DrainageNodeCouplingData(
            node_id=pyswmm_node.nodeid, node_object=node, x=x_coor, y=y_coor, row=row, col=col
        )
        nodes_list.append(drainage_node_data)
    return nodes_list


def get_links_list(pyswmm_links, links_vertices_dict, nodes_coor_dict):
    """ """
    links_list = []
    for pyswmm_link in pyswmm_links:
        # Add nodes coordinates to the vertices list
        in_node_coor = nodes_coor_dict[pyswmm_link.inlet_node]
        out_node_coor = nodes_coor_dict[pyswmm_link.outlet_node]
        vertices = [in_node_coor]
        vertices.extend(links_vertices_dict[pyswmm_link.linkid].vertices)
        vertices.append(out_node_coor)
        link = DrainageLink(link_object=pyswmm_link, vertices=vertices)
        # add link to the list
        links_list.append(link)
    return links_list


# correspondance between internal numpy arrays and map names
in_k_corresp = {
    "dem": "dem",
    "friction": "friction",
    "h": "start_h",
    "y": "start_y",
    "effective_porosity": "effective_porosity",
    "capillary_pressure": "capillary_pressure",
    "hydraulic_conductivity": "hydraulic_conductivity",
    "in_inf": "infiltration",
    "losses": "losses",
    "rain": "rain",
    "inflow": "inflow",
    "bcval": "bcval",
    "bctype": "bctype",
}


def create_grass_simulation(
    sim_times,
    input_maps,
    output_maps,
    sim_param,
    drainage_params,
    grass_interface,
    dtype=np.float32,
    stats_file=None,
) -> tuple[Simulation, Dict[str, rasterdomain.TimedArray]]:
    """A factory function that returns a Simulation object."""
    msgr.verbose("Setting up models...")
    from itzi.providers import GrassRasterOutputProvider, GrassVectorOutputProvider

    arr_mask = grass_interface.get_npmask()
    msgr.verbose("Reading maps information from GIS...")
    grass_interface.read(input_maps)
    # Timed arrays
    tarr = {}
    # TimedArray expects a function as an init parameter
    zeros_array = lambda: np.zeros(shape=raster_shape, dtype=dtype)  # noqa: E731
    for k in in_k_corresp.keys():
        tarr[k] = rasterdomain.TimedArray(in_k_corresp[k], grass_interface, zeros_array)
    msgr.debug("Setting up raster domain...")
    # RasterDomain
    raster_shape = (grass_interface.yr, grass_interface.xr)
    try:
        raster_domain = rasterdomain.RasterDomain(
            dtype=dtype,
            arr_mask=arr_mask,
            cell_shape=(grass_interface.dx, grass_interface.dy),
        )
    except MemoryError:
        msgr.fatal("Out of memory.")
    # Infiltration
    inf_model = sim_param["inf_model"]
    dtinf = sim_param["dtinf"]
    msgr.debug("Setting up raster infiltration...")
    inf_class = {
        "constant": infiltration.InfConstantRate,
        "green-ampt": infiltration.InfGreenAmpt,
        "null": infiltration.InfNull,
    }
    try:
        infiltration_model = inf_class[inf_model](raster_domain, dtinf)
    except KeyError:
        assert False, f"Unknow infiltration model: {inf_model}"
    # Hydrology
    msgr.debug("Setting up hydrologic model...")
    hydrology_model = Hydrology(raster_domain, dtinf, infiltration_model)
    # Surface flows simulation
    msgr.debug("Setting up surface model...")
    surface_flow = SurfaceFlowSimulation(raster_domain, sim_param)
    # Instantiate Massbal object
    if stats_file:
        msgr.debug("Setting up mass balance object...")
        massbal = MassBalanceLogger(
            file_name=stats_file,
            start_time=sim_times.start,
            temporal_type=sim_times.temporal_type,
        )
    else:
        massbal = None
    # Drainage
    if drainage_params["swmm_inp"]:
        msgr.debug("Setting up drainage model...")
        swmm_sim = pyswmm.Simulation(drainage_params["swmm_inp"])
        swmm_inp = SwmmInputParser(drainage_params["swmm_inp"])
        # Create Node objects
        all_nodes = pyswmm.Nodes(swmm_sim)
        nodes_coors_dict = swmm_inp.get_nodes_id_as_dict()
        nodes_list = get_nodes_list(
            all_nodes, nodes_coors_dict, drainage_params, grass_interface, sim_param["g"]
        )
        # Create Link objects
        links_vertices_dict = swmm_inp.get_links_id_as_dict()
        links_list = get_links_list(pyswmm.Links(swmm_sim), links_vertices_dict, nodes_coors_dict)
        node_objects_only = [i.node_object for i in nodes_list]
        drainage_sim = DrainageSimulation(swmm_sim, node_objects_only, links_list)
    else:
        nodes_list = None
        drainage_sim = None
    # reporting object
    msgr.debug("Setting up reporting object...")
    raster_output_provider = GrassRasterOutputProvider()
    raster_output_provider.initialize(
        {
            "grass_interface": grass_interface,
            "out_map_names": output_maps,
            "hmin": sim_param["hmin"],
            "temporal_type": sim_times.temporal_type,
        }
    )
    vector_output_provider = GrassVectorOutputProvider()
    vector_output_provider.initialize(
        {
            "grass_interface": grass_interface,
            "temporal_type": sim_times.temporal_type,
            "drainage_map_name": drainage_params["output"],
        }
    )
    report = Report(
        start_time=sim_times.start,
        raster_output_provider=raster_output_provider,
        vector_output_provider=vector_output_provider,
        mass_balance_logger=massbal,
        out_map_names=output_maps,
        dt=sim_times.record_step,
    )
    msgr.verbose("Models set up")
    simulation = Simulation(
        sim_times.start,
        sim_times.end,
        raster_domain,
        hydrology_model,
        surface_flow,
        drainage_sim,
        nodes_list,
        report,
        mass_balance_error_threshold=sim_param["max_error"],
    )
    return (simulation, tarr)


def create_memory_simulation(
    sim_times,
    input_maps,
    output_maps,
    sim_param,
    drainage_params,
    grass_interface,
    dtype=np.float32,
    stats_file=None,
) -> Simulation:
    pass
