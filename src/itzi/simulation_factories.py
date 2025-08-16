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

from typing import Dict, TYPE_CHECKING, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
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
from itzi.array_definitions import ARRAY_DEFINITIONS, ArrayCategory
from itzi.const import InfiltrationModelType

if TYPE_CHECKING:
    from itzi.providers.grass_interface import GrassInterface
    from itzi.providers.grass_input import GrassRasterInputConfig
    from itzi.providers.domain_data import DomainData
    from itzi.data_containers import SimulationConfig
    import icechunk


def get_nodes_list(
    pswmm_nodes: list,
    nodes_coor_dict: Dict,
    orifice_coeff: float,
    free_weir_coeff: float,
    submerged_weir_coeff: float,
    domain_data: "DomainData",
    g: float,
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
            orifice_coeff=orifice_coeff,
            free_weir_coeff=free_weir_coeff,
            submerged_weir_coeff=submerged_weir_coeff,
            g=g,
        )
        # a node without coordinates cannot be coupled
        if coors is None or not domain_data.is_in_domain(x=coors.x, y=coors.y):
            x_coor = None
            y_coor = None
            row = None
            col = None
        else:
            # Set node as coupled with no flow
            node.coupling_type = CouplingTypes.COUPLED_NO_FLOW
            x_coor = coors.x
            y_coor = coors.y
            row, col = domain_data.coordinates_to_pixel(x=x_coor, y=y_coor)
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


def _create_raster_domain(
    dtype: DTypeLike,
    arr_mask: ArrayLike,
    cell_shape: tuple,
) -> rasterdomain.RasterDomain:
    """Create a raster domain with error handling."""
    msgr.debug("Setting up raster domain...")
    try:
        raster_domain = rasterdomain.RasterDomain(
            dtype=dtype,
            arr_mask=arr_mask,
            cell_shape=cell_shape,
        )
    except MemoryError:
        msgr.fatal("Out of memory.")
    return raster_domain


def _create_infiltration_model(
    sim_config: "SimulationConfig",
    raster_domain: rasterdomain.RasterDomain,
) -> infiltration.InfiltrationModel:
    """Create an infiltration model based on configuration."""
    inf_model = sim_config.infiltration_model
    dtinf = sim_config.dtinf
    msgr.debug("Setting up raster infiltration...")

    inf_class = {
        InfiltrationModelType.CONSTANT: infiltration.InfConstantRate,
        InfiltrationModelType.GREEN_AMPT: infiltration.InfGreenAmpt,
        InfiltrationModelType.NULL: infiltration.InfNull,
    }
    try:
        infiltration_model = inf_class[inf_model](raster_domain, dtinf)
    except KeyError:
        assert False, f"Unknow infiltration model: {inf_model}"
    return infiltration_model


def _create_drainage_simulation(
    sim_config: "SimulationConfig",
    domain_data: "DomainData",
) -> Tuple[Optional[list], Optional[DrainageSimulation]]:
    """Create drainage simulation components if SWMM input is provided."""
    if not sim_config.swmm_inp:
        return None, None

    msgr.debug("Setting up drainage model...")
    swmm_sim = pyswmm.Simulation(sim_config.swmm_inp)
    swmm_inp = SwmmInputParser(sim_config.swmm_inp)

    # Create Node objects
    all_nodes = pyswmm.Nodes(swmm_sim)
    nodes_coors_dict = swmm_inp.get_nodes_id_as_dict()
    nodes_list = get_nodes_list(
        all_nodes,
        nodes_coors_dict,
        orifice_coeff=sim_config.orifice_coeff,
        free_weir_coeff=sim_config.free_weir_coeff,
        submerged_weir_coeff=sim_config.submerged_weir_coeff,
        domain_data=domain_data,
        g=sim_config.surface_flow_parameters.g,
    )

    # Create Link objects
    links_vertices_dict = swmm_inp.get_links_id_as_dict()
    links_list = get_links_list(pyswmm.Links(swmm_sim), links_vertices_dict, nodes_coors_dict)
    node_objects_only = [i.node_object for i in nodes_list]
    drainage_sim = DrainageSimulation(swmm_sim, node_objects_only, links_list)

    return nodes_list, drainage_sim


def create_grass_simulation(
    sim_config: "SimulationConfig",
    grass_interface: "GrassInterface",
    dtype=np.float32,
) -> tuple[Simulation, Dict[str, rasterdomain.TimedArray]]:
    """A factory function that returns a Simulation object."""
    msgr.verbose("Setting up models...")
    from itzi.providers.grass_output import GrassRasterOutputProvider, GrassVectorOutputProvider
    from itzi.providers.grass_input import GrassRasterInputProvider

    arr_mask = grass_interface.get_npmask()
    msgr.verbose("Reading maps information from GIS...")
    # Timed arrays
    timed_arrays = {}
    # TimedArray expects a function as an init parameter
    zeros_array = lambda: np.zeros(shape=raster_shape, dtype=dtype)  # noqa: E731
    input_keys = [
        arr_def.key for arr_def in ARRAY_DEFINITIONS if ArrayCategory.INPUT in arr_def.category
    ]
    raster_input_provider_config: "GrassRasterInputConfig" = {
        "grass_interface": grass_interface,
        "input_map_names": sim_config.input_map_names,
        "default_start_time": sim_config.start_time,
        "default_end_time": sim_config.end_time,
    }
    raster_input_provider = GrassRasterInputProvider(config=raster_input_provider_config)
    for arr_key in input_keys:
        timed_arrays[arr_key] = rasterdomain.TimedArray(
            arr_key, raster_input_provider, zeros_array
        )
    # RasterDomain
    raster_shape = (grass_interface.yr, grass_interface.xr)
    raster_domain = _create_raster_domain(
        dtype=dtype,
        arr_mask=arr_mask,
        cell_shape=(grass_interface.dx, grass_interface.dy),
    )

    # Infiltration
    infiltration_model = _create_infiltration_model(sim_config, raster_domain)

    # Hydrology
    msgr.debug("Setting up hydrologic model...")
    hydrology_model = Hydrology(raster_domain, sim_config.dtinf, infiltration_model)

    # Surface flows simulation
    msgr.debug("Setting up surface model...")
    surface_flow = SurfaceFlowSimulation(raster_domain, sim_config.surface_flow_parameters)

    # Instantiate Massbal object
    if sim_config.stats_file:
        msgr.debug("Setting up mass balance object...")
        massbal = MassBalanceLogger(
            file_name=sim_config.stats_file,
        )
    else:
        massbal = None

    # Drainage
    domain_data = raster_input_provider.get_domain_data()
    nodes_list, drainage_sim = _create_drainage_simulation(sim_config, domain_data)
    # reporting object
    msgr.debug("Setting up reporting object...")
    raster_output_provider = GrassRasterOutputProvider(
        {
            "grass_interface": grass_interface,
            "out_map_names": sim_config.output_map_names,
            "hmin": sim_config.surface_flow_parameters.hmin,
            "temporal_type": sim_config.temporal_type,
        }
    )
    vector_output_provider = GrassVectorOutputProvider(
        {
            "grass_interface": grass_interface,
            "temporal_type": sim_config.temporal_type,
            "drainage_map_name": sim_config.drainage_output,
        }
    )

    report = Report(
        start_time=sim_config.start_time,
        temporal_type=sim_config.temporal_type,
        raster_output_provider=raster_output_provider,
        vector_output_provider=vector_output_provider,
        mass_balance_logger=massbal,
        out_map_names=sim_config.output_map_names,
        dt=sim_config.record_step,
    )
    msgr.verbose("Models set up")
    simulation = Simulation(
        sim_config.start_time,
        sim_config.end_time,
        raster_domain,
        hydrology_model,
        surface_flow,
        drainage_sim,
        nodes_list,
        report,
        mass_balance_error_threshold=sim_config.surface_flow_parameters.max_error,
    )
    return (simulation, timed_arrays)


def create_memory_simulation(
    sim_config: "SimulationConfig",
    domain_data: "DomainData",
    arr_mask: ArrayLike,
    dtype: DTypeLike = np.float32,
) -> Simulation:
    from itzi.providers.memory_output import MemoryRasterOutputProvider, MemoryVectorOutputProvider

    # raster domain
    raster_domain = _create_raster_domain(
        dtype=dtype,
        arr_mask=arr_mask,
        cell_shape=domain_data.cell_shape,
    )

    # Infiltration
    infiltration_model = _create_infiltration_model(sim_config, raster_domain)

    # Hydrology
    msgr.debug("Setting up hydrologic model...")
    hydrology_model = Hydrology(raster_domain, sim_config.dtinf, infiltration_model)

    # Surface flows simulation
    msgr.debug("Setting up surface model...")
    surface_flow = SurfaceFlowSimulation(raster_domain, sim_config.surface_flow_parameters)

    # Instantiate Massbal object
    if sim_config.stats_file:
        msgr.debug("Setting up mass balance object...")
        massbal = MassBalanceLogger(
            file_name=sim_config.stats_file,
        )
    else:
        massbal = None

    # Drainage
    nodes_list, drainage_sim = _create_drainage_simulation(sim_config, domain_data)
    # reporting object
    msgr.debug("Setting up reporting object...")
    raster_output_provider = MemoryRasterOutputProvider(
        {
            "out_map_names": sim_config.output_map_names,
        }
    )

    vector_output_provider = MemoryVectorOutputProvider({})
    report = Report(
        start_time=sim_config.start_time,
        temporal_type=sim_config.temporal_type,
        raster_output_provider=raster_output_provider,
        vector_output_provider=vector_output_provider,
        mass_balance_logger=massbal,
        out_map_names=sim_config.output_map_names,
        dt=sim_config.record_step,
    )
    msgr.verbose("Models set up")
    simulation = Simulation(
        sim_config.start_time,
        sim_config.end_time,
        raster_domain,
        hydrology_model,
        surface_flow,
        drainage_sim,
        nodes_list,
        report,
        mass_balance_error_threshold=sim_config.surface_flow_parameters.max_error,
    )
    return simulation


def create_icechunk_simulation(
    sim_config: "SimulationConfig",
    arr_mask: ArrayLike,
    icechunk_storage: "icechunk.Storage",
    icechunk_group: str = "main",
    output_dir: str = ".",
    dtype: DTypeLike = np.float32,
) -> tuple[Simulation, Dict[str, rasterdomain.TimedArray]]:
    """A factory function that returns a Simulation object with Icechunk raster backend and Parquet vector backend."""
    msgr.verbose("Setting up models...")
    try:
        import pyproj
        from itzi.providers.icechunk_input import (
            IcechunkRasterInputProvider,
            IcechunkRasterInputConfig,
        )
        from itzi.providers.icechunk_output import IcechunkRasterOutputProvider
        from itzi.providers.geoparquet_output import ParquetVectorOutputProvider
    except ImportError:
        raise ImportError(
            "To use the Icechunk backend, install itzi with: "
            "'uv tool install itzi[cloud]' "
            "or 'pip install itzi[cloud]'"
        )

    # Set up raster input provider
    raster_input_provider_config: IcechunkRasterInputConfig = {
        "icechunk_storage": icechunk_storage,
        "icechunk_group": icechunk_group,
        "input_map_names": sim_config.input_map_names,
        "simulation_start_time": sim_config.start_time,
        "simulation_end_time": sim_config.end_time,
    }
    raster_input_provider = IcechunkRasterInputProvider(config=raster_input_provider_config)
    domain_data = raster_input_provider.get_domain_data()
    # Timed arrays
    timed_arrays = {}
    # TimedArray expects a function as an init parameter
    raster_shape = (domain_data.rows, domain_data.cols)
    zeros_array = lambda: np.zeros(shape=raster_shape, dtype=dtype)  # noqa: E731
    input_keys = [
        arr_def.key for arr_def in ARRAY_DEFINITIONS if ArrayCategory.INPUT in arr_def.category
    ]

    for arr_key in input_keys:
        timed_arrays[arr_key] = rasterdomain.TimedArray(
            arr_key, raster_input_provider, zeros_array
        )

    # RasterDomain
    raster_domain = _create_raster_domain(
        dtype=dtype,
        arr_mask=arr_mask,
        cell_shape=domain_data.cell_shape,
    )

    # Infiltration
    infiltration_model = _create_infiltration_model(sim_config, raster_domain)

    # Hydrology
    msgr.debug("Setting up hydrologic model...")
    hydrology_model = Hydrology(raster_domain, sim_config.dtinf, infiltration_model)

    # Surface flows simulation
    msgr.debug("Setting up surface model...")
    surface_flow = SurfaceFlowSimulation(raster_domain, sim_config.surface_flow_parameters)

    # Instantiate Massbal object
    if sim_config.stats_file:
        msgr.debug("Setting up mass balance object...")
        massbal = MassBalanceLogger(
            file_name=sim_config.stats_file,
        )
    else:
        massbal = None

    # Drainage
    nodes_list, drainage_sim = _create_drainage_simulation(sim_config, domain_data)

    # reporting object
    msgr.debug("Setting up reporting object...")

    # Set up raster output provider
    # Generate coordinate arrays from domain_data
    x_coords, y_coords = domain_data.get_coordinates()

    crs = pyproj.CRS.from_wkt(domain_data.crs_wkt)

    raster_output_provider = IcechunkRasterOutputProvider(
        {
            "out_var_names": sim_config.output_map_names,
            "crs": crs,
            "x_coords": x_coords,
            "y_coords": y_coords,
            "icechunk_storage": icechunk_storage,
        }
    )

    # Set up vector output provider
    vector_output_provider = ParquetVectorOutputProvider(
        {
            "crs": crs,
            "output_dir": output_dir,
            "drainage_map_name": sim_config.drainage_output,
        }
    )

    report = Report(
        start_time=sim_config.start_time,
        temporal_type=sim_config.temporal_type,
        raster_output_provider=raster_output_provider,
        vector_output_provider=vector_output_provider,
        mass_balance_logger=massbal,
        out_map_names=sim_config.output_map_names,
        dt=sim_config.record_step,
    )

    msgr.verbose("Models set up")
    simulation = Simulation(
        sim_config.start_time,
        sim_config.end_time,
        raster_domain,
        hydrology_model,
        surface_flow,
        drainage_sim,
        nodes_list,
        report,
        mass_balance_error_threshold=sim_config.surface_flow_parameters.max_error,
    )
    return (simulation, timed_arrays)
