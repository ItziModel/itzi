# coding=utf8
"""
Copyright (C) 2016-2025 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

import math
from datetime import timedelta
from enum import StrEnum

import pyswmm
import numpy as np

from itzi import DefaultValues
from itzi import messenger as msgr
from itzi.data_containers import (
    DrainageNodeData,
    DrainageLinkData,
    DrainageNetworkData,
    DrainageLinkAttributes,
    DrainageNodeAttributes,
)


class CouplingTypes(StrEnum):
    NOT_COUPLED = "not coupled"
    COUPLED_NO_FLOW = "coupled, no flow"
    FREE_WEIR = "coupled, free weir"
    SUBMERGED_WEIR = "coupled, submerged weir"
    ORIFICE = "coupled, orifice"


class DrainageSimulation:
    """manage simulation of the pipe network"""

    def __init__(self, pyswmm_sim, nodes_list, links_list):
        # A list of DrainageNode object
        self.nodes = nodes_list
        # A list of DrainageLink objects
        self.links = links_list
        # create swmm object, open files and start simulation
        self.swmm_sim = pyswmm_sim
        self.swmm_model = self.swmm_sim._model
        # Check if the unit is m3/s
        if self.swmm_sim.flow_units != "CMS":
            msgr.fatal("SWMM simulation unit must be CMS")
        self.swmm_model.swmm_start()
        # allow ponding
        # TODO: check if allowing ponding is necessary
        self._dt = 0.0
        self.elapsed_time = 0.0

    def __del__(self):
        """Make sure the swmm simulation is ended and closed properly."""
        self.swmm_model.swmm_report()
        self.swmm_model.swmm_close()

    @property
    def dt(self):
        return timedelta(seconds=self._dt)

    @dt.setter
    def dt(self, value):
        raise NotImplementedError("Can't set time-step of a SWMM simulation")

    def step(self):
        """Run a swmm time-step
        calculate the exchanges with raster domain
        """
        elapsed_days = self.swmm_model.swmm_step()
        elapsed_seconds = elapsed_days * 24 * 3600
        # calculate the time step
        self._dt = elapsed_seconds - self.elapsed_time
        assert self._dt > 0.0
        self.elapsed_time = elapsed_seconds
        return self

    def apply_coupling_to_nodes(self, surface_states, cell_surf):
        """For each coupled node,
        calculate the flow entering or leaving the drainage network
        surface_states {node_id: {'z': float, 'h': float}}
        """
        calculated_flows = {}
        for node in self.nodes:
            node_id = node.node_id
            if node.is_coupled() and (node_id in surface_states):
                state = surface_states[node_id]
                node.apply_coupling(state["z"], state["h"], self._dt, cell_surf)
                calculated_flows[node_id] = node.coupling_flow
        return calculated_flows

    def get_drainage_network_data(self) -> DrainageNetworkData:
        nodes_data = []
        links_data = []
        for node in self.nodes:
            nodes_data.append(node.get_data())
        for link in self.links:
            links_data.append(link.get_data())
        return DrainageNetworkData(nodes=tuple(nodes_data), links=tuple(links_data))


class DrainageNode(object):
    """A wrapper around the pyswmm node object.
    Includes the flow coupling logic
    """

    def __init__(
        self,
        node_object,
        coordinates=None,
        coupling_type=CouplingTypes.NOT_COUPLED,
        orifice_coeff=DefaultValues.ORIFICE_COEFF,
        free_weir_coeff=DefaultValues.FREE_WEIR_COEFF,
        submerged_weir_coeff=DefaultValues.SUBMERGED_WEIR_COEFF,
        g=DefaultValues.G,
        relaxation_factor=DefaultValues.RELAXATION_FACTOR,
        damping_factor=DefaultValues.DAMPING_FACTOR,
    ):
        self.g = g
        self._model = node_object._model
        self.pyswmm_node = node_object
        # need to add a node validity check
        self.node_id = node_object.nodeid
        self.coordinates = coordinates
        self.orifice_coeff = orifice_coeff
        self.free_weir_coeff = free_weir_coeff
        self.submerged_weir_coeff = submerged_weir_coeff
        self.node_type = self.get_node_type()
        self.surface_area = self._model.getSimAnalysisSetting(
            pyswmm.toolkitapi.SimulationParameters.MinSurfArea
        )
        # weir width is the circumference (node considered circular)
        self.weir_width = 2 * math.sqrt(self.surface_area * math.pi)
        # Set default values
        self.coupling_type = coupling_type
        self.coupling_flow = 0.0
        # TODO: set surcharge depth
        self.relaxation_factor = relaxation_factor
        self.damping_factor = damping_factor

    def get_node_type(self):
        """ """
        if self.pyswmm_node.is_junction():
            return "junction"
        elif self.pyswmm_node.is_outfall():
            return "outfall"
        elif self.pyswmm_node.is_divider():
            return "divider"
        elif self.pyswmm_node.is_storage():
            return "storage"
        else:
            raise ValueError(f"Unknown node type for node {self.node_id}")

    def get_full_volume(self):
        return self.surface_area * self.pyswmm_node.full_depth

    def get_overflow(self):
        return self._model.getNodeResult(self.node_id, pyswmm.toolkitapi.NodeResults.overflow)

    def get_crest_elev(self):
        """Return the crest elevation of the node."""
        return self.pyswmm_node.invert_elevation + self.pyswmm_node.full_depth

    def is_coupled(self):
        """return True if the node is coupled to the 2D domain"""
        return self.coupling_type != CouplingTypes.NOT_COUPLED

    def get_attrs(self) -> DrainageNodeAttributes:
        """ """
        return DrainageNodeAttributes(
            node_id=self.node_id,
            node_type=self.node_type,
            coupling_type=self.coupling_type.value,
            coupling_flow=self.coupling_flow,
            inflow=self.pyswmm_node.total_inflow,
            outflow=self.pyswmm_node.total_outflow,
            lateral_inflow=self.pyswmm_node.lateral_inflow,
            losses=self.pyswmm_node.losses,
            overflow=self.get_overflow(),
            depth=self.pyswmm_node.depth,
            head=self.pyswmm_node.head,
            # crownElev=values['crown_elev'],
            crest_elevation=self.get_crest_elev(),
            invert_elevation=self.pyswmm_node.invert_elevation,
            initial_depth=self.pyswmm_node.initial_depth,
            full_depth=self.pyswmm_node.full_depth,
            surcharge_depth=self.pyswmm_node.surcharge_depth,
            ponding_area=self.pyswmm_node.ponding_area,
            # degree=values['degree'],
            volume=self.pyswmm_node.volume,
            full_volume=self.get_full_volume(),
        )

    def get_data(self) -> DrainageNodeData:
        return DrainageNodeData(coordinates=self.coordinates, attributes=self.get_attrs())

    def apply_coupling(self, z, h, dt_drainage, cell_surf):
        """Apply the coupling to the node"""
        wse = z + h
        # Default crest elevation to the DEM elevation
        crest_elev = z

        # Calculate the coupling type and flow
        self.coupling_type = self._get_coupling_type(wse, crest_elev)
        new_coupling_flow = self._get_coupling_flow(wse, crest_elev)

        ## flow relaxation ##
        # Apply a relaxation factor (blend new flow with previous flow)
        new_coupling_flow = (
            self.relaxation_factor * new_coupling_flow
            + (1 - self.relaxation_factor) * self.coupling_flow
        )

        ## flow limiter ##
        # flow leaving the 2D domain can't drain the corresponding cell
        if new_coupling_flow < 0.0:
            maxflow = (h * cell_surf) / dt_drainage
            new_coupling_flow = max(new_coupling_flow, -maxflow)

        ## Reduce flow by damping factor in case of inversion ##
        old_flow = self.coupling_flow
        overflow_to_drainage = old_flow > 0 and new_coupling_flow < 0
        drainage_to_overflow = old_flow < 0 and new_coupling_flow > 0
        if overflow_to_drainage or drainage_to_overflow:
            new_coupling_flow = new_coupling_flow * self.damping_factor

        ## Apply flow to internal values and pyswmm node ##
        # pyswmm fails if type not forced to double
        # itzi considers the flow as positive when leaving the drainage, pyswmm is the opposite
        self.pyswmm_node.generated_inflow(np.float64(-new_coupling_flow))
        # update internal values
        self.coupling_flow = new_coupling_flow
        return self

    def _get_coupling_type(self, wse, crest_elev):
        """orifice, free- and submerged-weir
        M. Rubinato et al. (2017)
        “Experimental Calibration and Validation of Sewer/surface Flow Exchange Equations
        in Steady and Unsteady Flow Conditions.”
        https://doi.org/10.1016/j.jhydrol.2017.06.024.
        """
        depth_2d = wse - crest_elev
        weir_ratio = self.surface_area / self.weir_width
        node_head = self.pyswmm_node.head
        overflow = node_head > wse
        drainage = node_head < wse
        free_weir = drainage and (node_head < crest_elev)
        submerged_weir = drainage and (node_head > crest_elev) and (depth_2d < weir_ratio)
        drainage_orifice = drainage and (node_head > crest_elev) and (depth_2d > weir_ratio)
        # orifice
        if overflow or drainage_orifice:
            new_coupling_type = CouplingTypes.ORIFICE
        # drainage free weir
        elif free_weir:
            new_coupling_type = CouplingTypes.FREE_WEIR
        # drainage submerged weir
        elif submerged_weir:
            new_coupling_type = CouplingTypes.SUBMERGED_WEIR
        else:
            new_coupling_type = CouplingTypes.COUPLED_NO_FLOW
        return new_coupling_type

    def _get_coupling_flow(self, wse, crest_elev):
        """flow sign is :
        - negative when entering the drainage (leaving the 2D model)
        - positive when leaving the drainage (entering the 2D model)
        """
        node_head = self.pyswmm_node.head
        head_up = max(wse, node_head)
        head_down = min(wse, node_head)
        head_diff = head_up - head_down
        upstream_depth = head_up - crest_elev

        # calculate the flow
        if self.coupling_type in [
            CouplingTypes.COUPLED_NO_FLOW,
            CouplingTypes.NOT_COUPLED,
        ]:
            unsigned_q = 0.0
        elif self.coupling_type == CouplingTypes.ORIFICE:
            unsigned_q = (
                self.orifice_coeff * self.surface_area * math.sqrt(2.0 * self.g * head_diff)
            )
        elif self.coupling_type == CouplingTypes.FREE_WEIR:
            unsigned_q = (
                (2.0 / 3.0)
                * self.free_weir_coeff
                * self.weir_width
                * math.pow(upstream_depth, 3 / 2.0)
                * math.sqrt(2.0 * self.g)
            )
        elif self.coupling_type == CouplingTypes.SUBMERGED_WEIR:
            unsigned_q = (
                self.submerged_weir_coeff
                * self.weir_width
                * upstream_depth
                * math.sqrt(2.0 * self.g * upstream_depth)
            )
        return math.copysign(unsigned_q, node_head - wse)


class DrainageLink(object):
    """A wrapper around the pyswmm link object"""

    def __init__(self, link_object, vertices=[]):
        self.pyswmm_link = link_object
        self.link_id = self.pyswmm_link.linkid
        self.link_type = self._get_link_type()
        self.start_node_id = self.pyswmm_link.inlet_node
        self.end_node_id = self.pyswmm_link.outlet_node
        # vertices include the coordinates of the start and end nodes
        self.vertices = vertices

    def _get_link_type(self):
        """return the type of the link"""
        if self.pyswmm_link.is_conduit():
            link_type = "conduit"
        elif self.pyswmm_link.is_pump():
            link_type = "pump"
        elif self.pyswmm_link.is_orifice():
            link_type = "orifice"
        elif self.pyswmm_link.is_weir():
            link_type = "weir"
        elif self.pyswmm_link.is_outlet():
            link_type = "outlet"
        else:
            raise ValueError(f"Unknown link type for link {self.link_id}")
        return link_type

    def get_attrs(self) -> DrainageLinkAttributes:
        """ """
        return DrainageLinkAttributes(
            link_id=self.link_id,
            link_type=self.link_type,
            flow=self.pyswmm_link.flow,
            depth=self.pyswmm_link.depth,
            # velocity=values['velocity'],
            volume=self.pyswmm_link.volume,
            inlet_offset=self.pyswmm_link.inlet_offset,
            outlet_offset=self.pyswmm_link.outlet_offset,
            # full_depth=values['full_depth'],
            froude=self.pyswmm_link.froude,
        )

    def get_data(self) -> DrainageLinkData:
        return DrainageLinkData(vertices=self.vertices, attributes=self.get_attrs())
