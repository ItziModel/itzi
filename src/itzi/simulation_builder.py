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
    from itzi.providers.domain_data import DomainData
    from itzi.providers.base import RasterInputProvider, RasterOutputProvider, VectorOutputProvider
    from itzi.data_containers import SimulationConfig


class SimulationBuilder:
    """Builder for creating Simulation objects with different provider configurations."""

    def __init__(
        self,
        sim_config: "SimulationConfig",
        arr_mask: ArrayLike,
        dtype: DTypeLike = np.float32,
    ):
        self.sim_config = sim_config
        self.arr_mask = arr_mask
        self.dtype = dtype

        # Optional components (set via builder methods)
        self.raster_input_provider: Optional["RasterInputProvider"] = None
        self.domain_data: Optional["DomainData"] = None
        self.raster_output_provider: Optional["RasterOutputProvider"] = None
        self.vector_output_provider: Optional["VectorOutputProvider"] = None

    def with_input_provider(self, provider: "RasterInputProvider") -> "SimulationBuilder":
        """Set the raster input provider."""
        self.raster_input_provider = provider
        self.domain_data = provider.get_domain_data()
        return self

    def with_domain_data(self, domain_data: "DomainData") -> "SimulationBuilder":
        """Set domain data directly (for memory simulations without input provider)."""
        self.domain_data = domain_data
        return self

    def with_raster_output_provider(self, provider: "RasterOutputProvider") -> "SimulationBuilder":
        """Set the raster output provider."""
        self.raster_output_provider = provider
        return self

    def with_vector_output_provider(self, provider: "VectorOutputProvider") -> "SimulationBuilder":
        """Set the vector output provider."""
        self.vector_output_provider = provider
        return self

    def build(self) -> tuple[Simulation, Optional[Dict[str, rasterdomain.TimedArray]]]:
        """Build and return the simulation with optional timed arrays."""
        # Validate required components
        if self.domain_data is None:
            raise ValueError("Domain data must be set via input provider or directly")
        if self.raster_output_provider is None or self.vector_output_provider is None:
            raise ValueError("Output providers are mandatory")
        # Create timed arrays if input provider exists
        timed_arrays = None
        if self.raster_input_provider:
            timed_arrays = self._create_timed_arrays()

        # Create raster domain
        raster_domain = self._create_raster_domain(self.domain_data.cell_shape)

        # Create models
        infiltration_model = self._create_infiltration_model(raster_domain)
        hydrology_model = Hydrology(raster_domain, self.sim_config.dtinf, infiltration_model)
        surface_flow = SurfaceFlowSimulation(
            raster_domain, self.sim_config.surface_flow_parameters
        )

        # Create drainage
        nodes_list, drainage_sim = self._create_drainage_simulation()

        # Create report
        self.mass_balance = None
        if self.sim_config.stats_file:
            self.mass_balance = MassBalanceLogger(
                file_name=self.sim_config.stats_file,
            )
        report = Report(
            start_time=self.sim_config.start_time,
            temporal_type=self.sim_config.temporal_type,
            raster_output_provider=self.raster_output_provider,
            vector_output_provider=self.vector_output_provider,
            mass_balance_logger=self.mass_balance,
            out_map_names=self.sim_config.output_map_names,
            dt=self.sim_config.record_step,
        )

        # Create simulation
        simulation = Simulation(
            self.sim_config.start_time,
            self.sim_config.end_time,
            raster_domain,
            hydrology_model,
            surface_flow,
            drainage_sim,
            nodes_list,
            report=report,
            mass_balance_error_threshold=self.sim_config.surface_flow_parameters.max_error,
        )

        return simulation, timed_arrays

    def _create_timed_arrays(self) -> Dict[str, rasterdomain.TimedArray]:
        """ """
        timed_arrays = {}
        input_keys = [
            arr_def.key for arr_def in ARRAY_DEFINITIONS if ArrayCategory.INPUT in arr_def.category
        ]
        raster_shape = (self.domain_data.rows, self.domain_data.cols)
        # TimedArray expects a function as an init parameter
        zeros_array_func = lambda: np.zeros(shape=raster_shape, dtype=self.dtype)  # noqa: E731
        for arr_key in input_keys:
            timed_arrays[arr_key] = rasterdomain.TimedArray(
                arr_key, self.raster_input_provider, zeros_array_func
            )
        return timed_arrays

    def _create_raster_domain(self, cell_shape) -> rasterdomain.RasterDomain:
        """Create a raster domain."""
        msgr.debug("Setting up raster domain...")
        try:
            raster_domain = rasterdomain.RasterDomain(
                dtype=self.dtype,
                arr_mask=self.arr_mask,
                cell_shape=cell_shape,
            )
        except MemoryError:
            msgr.fatal("Out of memory.")
        return raster_domain

    def _create_infiltration_model(
        self,
        raster_domain: rasterdomain.RasterDomain,
    ) -> infiltration.InfiltrationModel:
        """Create an infiltration model based on configuration."""
        inf_model = self.sim_config.infiltration_model
        dtinf = self.sim_config.dtinf
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

    def _create_drainage_simulation(self) -> Tuple[Optional[list], Optional[DrainageSimulation]]:
        """Create drainage simulation components if SWMM input is provided."""
        if not self.sim_config.swmm_inp:
            return None, None

        msgr.debug("Setting up drainage model...")
        swmm_sim = pyswmm.Simulation(self.sim_config.swmm_inp)
        swmm_inp = SwmmInputParser(self.sim_config.swmm_inp)

        # Create Node objects
        all_nodes = pyswmm.Nodes(swmm_sim)
        nodes_coors_dict = swmm_inp.get_nodes_id_as_dict()
        nodes_list = self._get_nodes_list(
            all_nodes,
            nodes_coors_dict,
            orifice_coeff=self.sim_config.orifice_coeff,
            free_weir_coeff=self.sim_config.free_weir_coeff,
            submerged_weir_coeff=self.sim_config.submerged_weir_coeff,
            g=self.sim_config.surface_flow_parameters.g,
        )

        # Create Link objects
        links_vertices_dict = swmm_inp.get_links_id_as_dict()
        links_list = get_links_list(pyswmm.Links(swmm_sim), links_vertices_dict, nodes_coors_dict)
        node_objects_only = [i.node_object for i in nodes_list]
        drainage_sim = DrainageSimulation(swmm_sim, node_objects_only, links_list)

        return nodes_list, drainage_sim

    def _get_nodes_list(
        self,
        pswmm_nodes: list,
        nodes_coor_dict: Dict,
        orifice_coeff: float,
        free_weir_coeff: float,
        submerged_weir_coeff: float,
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
            if coors is None or not self.domain_data.is_in_domain(x=coors.x, y=coors.y):
                x_coor = None
                y_coor = None
                row = None
                col = None
            else:
                # Set node as coupled with no flow
                node.coupling_type = CouplingTypes.COUPLED_NO_FLOW
                x_coor = coors.x
                y_coor = coors.y
                row, col = self.domain_data.coordinates_to_pixel(x=x_coor, y=y_coor)
            # populate list
            drainage_node_data = DrainageNodeCouplingData(
                node_id=pyswmm_node.nodeid, node_object=node, x=x_coor, y=y_coor, row=row, col=col
            )
            nodes_list.append(drainage_node_data)
        return nodes_list


# Not in the main class to allow manual creation of a DrainageModel object for testing
def get_links_list(pyswmm_links, links_vertices_dict, nodes_coor_dict) -> list[DrainageLink]:
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
