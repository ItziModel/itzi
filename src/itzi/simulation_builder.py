"""
Copyright (C) 2025-2026 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING
import io
from pathlib import Path

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
from itzi.hotstart import HotstartLoader
from itzi.itzi_error import HotstartError

if TYPE_CHECKING:
    from itzi.providers.domain_data import DomainData
    from itzi.providers.base import RasterInputProvider, RasterOutputProvider, VectorOutputProvider
    from itzi.data_containers import SimulationConfig


class SimulationBuilder:
    """Builder for creating Simulation objects with different provider configurations."""

    def __init__(
        self,
        sim_config: SimulationConfig,
        arr_mask: ArrayLike,
        dtype: DTypeLike = np.float32,
    ):
        self.sim_config = sim_config
        self.arr_mask = arr_mask
        self.dtype = dtype

        # Optional components (set via builder methods)
        self.raster_input_provider: RasterInputProvider | None = None
        self.domain_data: DomainData | None = None
        self.raster_output_provider: RasterOutputProvider | None = None
        self.vector_output_provider: VectorOutputProvider | None = None

        # Hotstart data (set via with_hotstart)
        self.hotstart_loader: HotstartLoader | None = None

    def with_hotstart(
        self, hotstart_path_or_bytes: Path | str | io.BytesIO | bytes
    ) -> "SimulationBuilder":
        """Load and store validated hotstart data for state restoration during build.

        This method loads and validates the hotstart archive but does not perform
        congruence checks against providers. Congruence validation happens during
        build() when all providers are available.

        Args:
            hotstart_path_or_bytes: Path to hotstart file, or hotstart data as
                BytesIO/bytes.

        Returns:
            self for method chaining.

        Raises:
            HotstartError: If the hotstart archive is invalid or corrupted.
        """
        if isinstance(hotstart_path_or_bytes, (Path, str)):
            self.hotstart_loader = HotstartLoader.from_file(hotstart_path_or_bytes)
        else:
            self.hotstart_loader = HotstartLoader.from_bytes(hotstart_path_or_bytes)
        return self

    def with_input_provider(self, provider: RasterInputProvider) -> SimulationBuilder:
        """Set the raster input provider."""
        self.raster_input_provider = provider
        self.domain_data = provider.get_domain_data()
        return self

    def with_domain_data(self, domain_data: DomainData) -> SimulationBuilder:
        """Set domain data directly (for memory simulations without input provider)."""
        self.domain_data = domain_data
        return self

    def with_raster_output_provider(self, provider: RasterOutputProvider) -> SimulationBuilder:
        """Set the raster output provider."""
        self.raster_output_provider = provider
        return self

    def with_vector_output_provider(self, provider: VectorOutputProvider) -> SimulationBuilder:
        """Set the vector output provider."""
        self.vector_output_provider = provider
        return self

    def _validate_hotstart_congruence(self) -> None:
        """Validate hotstart data against builder configuration.

        This method performs congruence checks between the hotstart metadata
        and the current builder configuration. It must be called after all
        providers are attached but before any state mutation.

        Raises:
            HotstartError: If any congruence check fails.
        """

        hotstart_domain = self.hotstart_loader.get_domain_data()
        hotstart_config = self.hotstart_loader.get_simulation_config()

        # Validate domain metadata
        self._validate_domain_congruence(hotstart_domain)

        # Validate mask compatibility
        self._validate_mask_congruence(hotstart_domain)

        # Validate drainage expectations
        self._validate_drainage_congruence(hotstart_config)

    def _validate_domain_congruence(self, hotstart_domain: DomainData) -> None:
        """Validate that domain metadata matches between hotstart and builder."""
        assert self.domain_data is not None  # Already validated in build()

        # Check spatial bounds
        if not np.isclose(self.domain_data.north, hotstart_domain.north):
            raise HotstartError(
                f"Domain north mismatch: builder={self.domain_data.north}, "
                f"hotstart={hotstart_domain.north}"
            )
        if not np.isclose(self.domain_data.south, hotstart_domain.south):
            raise HotstartError(
                f"Domain south mismatch: builder={self.domain_data.south}, "
                f"hotstart={hotstart_domain.south}"
            )
        if not np.isclose(self.domain_data.east, hotstart_domain.east):
            raise HotstartError(
                f"Domain east mismatch: builder={self.domain_data.east}, "
                f"hotstart={hotstart_domain.east}"
            )
        if not np.isclose(self.domain_data.west, hotstart_domain.west):
            raise HotstartError(
                f"Domain west mismatch: builder={self.domain_data.west}, "
                f"hotstart={hotstart_domain.west}"
            )

        # Check dimensions
        if self.domain_data.rows != hotstart_domain.rows:
            raise HotstartError(
                f"Domain rows mismatch: builder={self.domain_data.rows}, "
                f"hotstart={hotstart_domain.rows}"
            )
        if self.domain_data.cols != hotstart_domain.cols:
            raise HotstartError(
                f"Domain cols mismatch: builder={self.domain_data.cols}, "
                f"hotstart={hotstart_domain.cols}"
            )

        # Check CRS
        if self.domain_data.crs_wkt != hotstart_domain.crs_wkt:
            raise HotstartError(
                "Domain CRS mismatch: builder and hotstart have different coordinate reference systems."
            )

    def _validate_mask_congruence(self, hotstart_domain: DomainData) -> None:
        """Validate that the builder mask is compatible with hotstart mask.

        The mask shape must match. The actual mask values will be validated
        during raster state restoration in RasterDomain.load_state().
        """
        expected_shape = hotstart_domain.shape
        builder_shape = self.arr_mask.shape

        if builder_shape != expected_shape:
            raise HotstartError(
                f"Mask shape mismatch: builder mask has shape {builder_shape}, "
                f"hotstart expects {expected_shape}"
            )

    def _validate_drainage_congruence(self, hotstart_config: SimulationConfig) -> None:
        """Validate drainage expectations match between hotstart and current config."""
        hotstart_has_drainage = hotstart_config.swmm_inp is not None
        builder_has_drainage = self.sim_config.swmm_inp is not None

        if hotstart_has_drainage and not builder_has_drainage:
            raise HotstartError(
                "Hotstart contains drainage state but current configuration has no drainage model"
            )

        if not hotstart_has_drainage and builder_has_drainage:
            raise HotstartError(
                "Hotstart has no drainage state but current configuration includes a drainage model"
            )

        # If both have drainage, check that SWMM hotstart bytes are present
        if hotstart_has_drainage and not self.hotstart_loader.has_swmm_hotstart():
            raise HotstartError(
                "Hotstart metadata indicates drainage but SWMM hotstart file is missing from archive"
            )

    def build(self) -> Simulation:
        """Build and return the simulation with optional timed arrays.

        Hotstart Restore Order (when hotstart is provided):
        ============================================================
        The restore sequence is carefully ordered to work with the existing
        constructor side effects:

        1. Validate hotstart congruence (before any object creation)
        2. Build normal objects from providers/config:
           - timed arrays (if input provider exists)
           - raster domain
           - infiltration model
           - hydrology model
           - surface flow model
           - drainage model (with SWMM hotstart injection if applicable)
           - report
        3. Create Simulation object (constructor calls update_input_arrays())
        4. Restore hotstart raster state into the domain
        5. Restore hotstart simulation runtime/scheduler state
        6. No explicit post-restore adjustments needed - the restore methods
           maintain the required invariants.

        This order ensures:
        - Provider-driven construction remains intact
        - SWMM hotstart is injected during drainage construction (required by pyswmm)
        - Raster state is restored after Simulation.__init__ to avoid being
          clobbered by update_input_arrays()
        - Scheduler invariants are maintained via restore_state()
        """
        # Validate required components
        if self.domain_data is None:
            raise ValueError("Domain data must be set via input provider or directly")
        if self.raster_output_provider is None or self.vector_output_provider is None:
            raise ValueError("Output providers are mandatory")

        # Validate hotstart congruence before building
        if self.hotstart_loader:
            self._validate_hotstart_congruence()

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

        # Create drainage with optional SWMM hotstart injection
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
            self.sim_config,
            self.domain_data,
            raster_domain,
            timed_arrays,
            hydrology_model,
            surface_flow,
            drainage_sim,
            nodes_list,
            report=report,
        )

        # Apply hotstart restore if hotstart data is present
        if self.hotstart_loader:
            # Restore raster state after simulation object exists
            # This must happen after Simulation.__init__ because the constructor
            # calls update_input_arrays() which could modify domain arrays
            raster_state_buffer = self.hotstart_loader.get_raster_state_buffer()
            raster_domain.load_state(raster_state_buffer)

            # Restore simulation runtime/scheduler state
            simulation_state = self.hotstart_loader.get_simulation_state()
            simulation.restore_state(simulation_state)

        return simulation

    def _create_timed_arrays(self) -> dict[str, rasterdomain.TimedArray]:
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

    def _create_drainage_simulation(self) -> tuple[list | None, DrainageSimulation | None]:
        """Create drainage simulation components if SWMM input is provided.

        If hotstart data includes SWMM state, writes the SWMM hotstart bytes
        to a temporary file and passes it to DrainageSimulation for restoration.
        The temporary file is cleaned up after DrainageSimulation reads it.
        """
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

        # Handle SWMM hotstart injection if present
        if self.hotstart_loader and self.hotstart_loader.has_swmm_hotstart():
            swmm_bytes = self.hotstart_loader.get_swmm_hotstart_bytes()
            # Create a temporary file for SWMM to read.
            # delete_on_close=False keeps the file after closing so SWMM can open it.
            with tempfile.NamedTemporaryFile(suffix=".hsf", delete_on_close=False) as tmp:
                tmp.write(swmm_bytes)
                hotstart_filename = tmp.name
                tmp.close()  # Allows SWMM to exclusively open the file
                drainage_sim = DrainageSimulation(
                    swmm_sim, node_objects_only, links_list, hotstart_filename=hotstart_filename
                )
        else:
            drainage_sim = DrainageSimulation(swmm_sim, node_objects_only, links_list)

        return nodes_list, drainage_sim

    def _get_nodes_list(
        self,
        pswmm_nodes: list,
        nodes_coor_dict: dict,
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
