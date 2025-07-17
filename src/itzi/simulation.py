# coding=utf8
"""
Copyright (C) 2015-2025 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

from datetime import datetime, timedelta
from collections import namedtuple
from typing import Self
import copy

import numpy as np
import pyswmm

from itzi.surfaceflow import SurfaceFlowSimulation
import itzi.rasterdomain as rasterdomain
from itzi.massbalance import MassBalanceLogger
from itzi.report import Report
from itzi.data_containers import ContinuityData, SimulationData
from itzi.drainage import DrainageSimulation, DrainageNode, DrainageLink, CouplingTypes
from itzi import SwmmInputParser
import itzi.messenger as msgr
import itzi.infiltration as infiltration
from itzi.hydrology import Hydrology
from itzi.itzi_error import NullError, MassBalanceError
from itzi import rastermetrics


DrainageNodeCouplingData = namedtuple(
    "DrainageNodeCouplingData", ["id", "object", "x", "y", "row", "col"]
)


def get_nodes_list(pswmm_nodes, nodes_coor_dict, drainage_params, igis, g):
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
        if coors is None or not igis.is_in_region(coors.x, coors.y):
            x_coor = None
            y_coor = None
            row = None
            col = None
        else:
            # Set node as coupled with no flow
            node.coupling_type = CouplingTypes.COUPLED_NO_FLOW
            x_coor = coors.x
            y_coor = coors.y
            row, col = igis.coor2pixel(coors)
        # populate list
        drainage_node_data = DrainageNodeCouplingData(
            id=pyswmm_node.nodeid, object=node, x=x_coor, y=y_coor, row=row, col=col
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


def create_simulation(
    sim_times,
    input_maps,
    output_maps,
    sim_param,
    drainage_params,
    grass_interface,
    dtype=np.float32,
    stats_file=None,
):
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
            fields=[
                "simulation_time",
                "average_timestep",
                "#timesteps",
                "boundary_volume",
                "rainfall_volume",
                "infiltration_volume",
                "inflow_volume",
                "losses_volume",
                "drainage_network_volume",
                "domain_volume",
                "volume_change",
                "volume_error",
                "percent_error",
            ],
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
        node_objects_only = [i.object for i in nodes_list]
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


class Simulation:
    """ """

    _array_keys = [
        "elevation",
        "manning_n",
        "depth",
        "effective_porosity",
        "capillary_pressure",
        "hydraulic_conductivity",
        "infiltration",
        "losses",
        "rain",
        "etp",
        "effective_precipitation",
        "inflow",
        "bcval",
        "bctype",
        "hmax",
        "ext",
        "hfe",
        "hfs",
        "qe",
        "qs",
        "qe_new",
        "qs_new",
        "ue",
        "us",
        "v",
        "vdir",
        "vmax",
        "froude",
        "n_drain",
        "capped_losses",
        "dire",
        "dirs",
    ]

    def __init__(
        self,
        start_time,
        end_time,
        raster_domain,
        hydrology_model,
        surface_flow,
        drainage_model,
        nodes_list,
        report,
        mass_balance_error_threshold: float,
    ):
        self.raster_domain = raster_domain
        self.start_time = start_time
        self.end_time = end_time
        # set simulation time to start_time
        self.sim_time = self.start_time
        self.raster_domain = raster_domain
        self.hydrology_model = hydrology_model
        self.drainage_model = drainage_model
        self.nodes_list = nodes_list
        self.surface_flow = surface_flow
        self.report = report
        # Mass balance error checking
        self.continuity_data: ContinuityData = None
        self.mass_balance_error_threshold = mass_balance_error_threshold
        # Accumulation array management
        self.accum_update_time: dict[str, datetime] = {
            "infiltration_accum": None,
            "rainfall_accum": None,
            "inflow_accum": None,
            "losses_accum": None,
            "drainage_network_accum": None,
        }
        self.accum_corresp: dict[str, str] = {
            "inf": "infiltration_accum",
            "rain": "rainfall_accum",
            "inflow": "inflow_accum",
            "capped_losses": "losses_accum",
            "n_drain": "drainage_network_accum",
        }
        # First time-step is forced
        self.dt = timedelta(seconds=0.001)
        self.nextstep = self.sim_time + self.dt
        # dict of next time-step (datetime object)
        self.next_ts = {"end": self.end_time}
        for k in ["hydrology", "surface_flow", "drainage"]:
            self.next_ts[k] = self.start_time
        # Schedule the first record step to avoid duplication with initialize()
        self.next_ts["record"] = self.start_time + self.report.dt
        # case if no drainage simulation
        if not self.drainage_model:
            self.next_ts["drainage"] = self.end_time
        else:
            self.node_id_to_loc = {
                n.id: (n.row, n.col) for n in self.nodes_list if n.object.is_coupled()
            }
        # Grid spacing (for BMI)
        self.spacing = (self.raster_domain.dy, self.raster_domain.dx)
        # time step counter
        self.time_steps_counter: int = 0

    def initialize(self) -> Self:
        """Record the initial stage of the simulation, before time-stepping."""
        self.old_domain_volume = rastermetrics.calculate_total_volume(
            depth_array=self.raster_domain.get_array("h"),
            cell_surface_area=self.raster_domain.cell_area,
        )
        for arr_key in self.accum_corresp.keys():
            self._update_accum_array(arr_key, self.sim_time)
        # Package data into SimulationData object
        raw_arrays = {
            k: self.raster_domain.get_unmasked(k)
            for k in self.raster_domain.k_all
            if k not in self.raster_domain.k_accum
        }
        accumulation_arrays = {
            k: self.raster_domain.get_unmasked(k) for k in self.raster_domain.k_accum
        }
        if self.drainage_model:
            drainage_network_data = self.drainage_model.get_drainage_network_data()
        else:
            drainage_network_data = None
        simulation_data = SimulationData(
            sim_time=self.sim_time,
            time_step=self.dt.total_seconds(),
            time_steps_counter=0,
            continuity_data=self.get_continuity_data(),
            raw_arrays=raw_arrays,
            accumulation_arrays=accumulation_arrays,
            cell_dx=self.raster_domain.dx,
            cell_dy=self.raster_domain.dy,
            drainage_network_data=drainage_network_data,
        )
        # Pass data to the reporting module
        self.report.step(simulation_data)

        # d. Reset accumulators
        self.raster_domain.reset_accumulations()
        for key in self.accum_update_time:
            self.accum_update_time[key] = self.sim_time
        return self

    def update(self) -> Self:
        # hydrology #
        if self.sim_time == self.next_ts["hydrology"]:
            self.hydrology_model.solve_dt()
            # calculate when will happen the next time-step
            self.next_ts["hydrology"] += self.hydrology_model.dt
            self.hydrology_model.step()

        # drainage #
        if self.sim_time == self.next_ts["drainage"] and self.drainage_model:
            self.drainage_model.step()
            # Update drainage nodes
            surface_states = {}
            cell_area = self.raster_domain.cell_area
            arr_z = self.raster_domain.get_array("dem")
            arr_h = self.raster_domain.get_array("h")
            for node_id, (row, col) in self.node_id_to_loc.items():
                surface_states[node_id] = {"z": arr_z[row, col], "h": arr_h[row, col]}
            coupling_flows = self.drainage_model.apply_coupling_to_nodes(surface_states, cell_area)
            # update drainage array with flux in m/s
            arr_qd = self.raster_domain.get_array("n_drain")
            for node_id, coupling_flow in coupling_flows.items():
                row, col = self.node_id_to_loc[node_id]
                arr_qd[row, col] = coupling_flow / cell_area
            # calculate when will happen the next time-step
            self.next_ts["drainage"] += self.drainage_model.dt

        # surface flow #
        # update arrays of infiltration, rainfall etc.
        self.raster_domain.update_ext_array()
        # force time-step to be the general time-step
        self.surface_flow.dt = self.dt
        # surface_flow.step() raise NullError in case of NaN/NULL cell
        # if this happen, stop simulation and
        # output a map showing the errors
        try:
            self.surface_flow.step()
        except NullError:
            msgr.fatal("{}: Null value detected in simulation, terminating".format(self.sim_time))
        # calculate when should happen the next surface time-step
        self.surface_flow.solve_dt()
        self.next_ts["surface_flow"] += self.surface_flow.dt

        # Update accumulation arrays
        for arr_key in self.accum_corresp.keys():
            self._update_accum_array(arr_key, self.sim_time)

        # Compute continuity error every x time steps
        is_first_ts = self.sim_time == self.start_time
        is_ts_over_threshold = self.time_steps_counter >= 500
        is_record_due = self.sim_time == self.next_ts["record"]
        is_error_comp_due = is_first_ts or is_ts_over_threshold or is_record_due
        if is_error_comp_due:
            self.continuity_data = self.get_continuity_data()

        # Reporting last to get simulated values #
        if is_record_due:
            msgr.verbose(f"{self.sim_time}: Writing output maps...")

            # Package data into SimulationData object
            raw_arrays = {
                k: self.raster_domain.get_unmasked(k)
                for k in self.raster_domain.k_all
                if k not in self.raster_domain.k_accum
            }
            accumulation_arrays = {
                k: self.raster_domain.get_unmasked(k) for k in self.raster_domain.k_accum
            }
            if self.drainage_model:
                drainage_network_data = self.drainage_model.get_drainage_network_data()
            else:
                drainage_network_data = None
            simulation_data = SimulationData(
                sim_time=self.sim_time,
                time_step=self.dt.total_seconds(),
                time_steps_counter=self.time_steps_counter,
                continuity_data=self.continuity_data,
                raw_arrays=raw_arrays,
                accumulation_arrays=accumulation_arrays,
                cell_dx=self.raster_domain.dx,
                cell_dy=self.raster_domain.dy,
                drainage_network_data=drainage_network_data,
            )

            # Pass data to the reporting module
            self.report.step(simulation_data)

            # Reset accumulation arrays
            self.old_domain_volume = copy.deepcopy(self.continuity_data.new_domain_vol)
            self.time_steps_counter = 0
            self.raster_domain.reset_accumulations()
            for key in self.accum_update_time:
                self.accum_update_time[key] = self.sim_time
            # Update next time step
            self.next_ts["record"] += self.report.dt

        # Perform a mass balance continuity check.
        if is_error_comp_due:
            if self.continuity_data.continuity_error >= self.mass_balance_error_threshold:
                raise (
                    MassBalanceError(
                        error_percentage=self.continuity_data.continuity_error,
                        threshold=self.mass_balance_error_threshold,
                    )
                )

        # Find next time step
        self.find_dt()
        # update simulation time
        self.sim_time += self.dt
        self.time_steps_counter += 1
        return self

    def update_until(self, then):
        """Run the simulation until a time in seconds after start_time"""
        assert isinstance(then, timedelta)
        end_time = self.start_time + then
        if end_time <= self.sim_time:
            raise ValueError("End time must be superior to current time")
        # Set temp end time (shorten last time step if necessary)
        self.next_ts["temp_end"] = end_time
        while self.sim_time < end_time:
            self.update()
        del self.next_ts["temp_end"]
        # Make sure everything went well
        assert self.sim_time == end_time
        return self

    def finalize(self):
        """ """
        # run surface flow simulation to get correct final volume
        self.raster_domain.update_ext_array()
        self.surface_flow.dt = self.dt
        self.surface_flow.step()
        # Update accumulation arrays
        for arr_key in self.accum_corresp.keys():
            self._update_accum_array(arr_key, self.sim_time)
        # Prepare SimulationData
        raw_arrays = {
            k: self.raster_domain.get_unmasked(k)
            for k in self.raster_domain.k_all
            if k not in self.raster_domain.k_accum
        }
        accumulation_arrays = {
            k: self.raster_domain.get_unmasked(k) for k in self.raster_domain.k_accum
        }
        if self.drainage_model:
            drainage_network_data = self.drainage_model.get_drainage_network_data()
        else:
            drainage_network_data = None
        simulation_data = SimulationData(
            sim_time=self.sim_time,
            time_step=self.dt.total_seconds(),
            time_steps_counter=self.time_steps_counter,
            continuity_data=self.get_continuity_data(),
            raw_arrays=raw_arrays,
            accumulation_arrays=accumulation_arrays,
            cell_dx=self.raster_domain.dx,
            cell_dy=self.raster_domain.dy,
            drainage_network_data=drainage_network_data,
        )
        # write final report.
        self.report.end(simulation_data)

    def set_array(self, arr_id, arr):
        """Set an array of the simulation domain"""
        assert isinstance(arr_id, str)
        assert isinstance(arr, np.ndarray)
        if arr_id in ["inflow", "rain"]:
            self._update_accum_array(arr_id, self.sim_time)
        self.raster_domain.update_array(arr_id, arr)
        if arr_id == "dem":
            self.surface_flow.update_flow_dir()
        return self

    def get_array(self, arr_id):
        """ """
        assert isinstance(arr_id, str)
        return self.raster_domain.get_array(arr_id)

    def find_dt(self):
        """find next time step"""
        self.nextstep = min(self.next_ts.values())
        # Surface flow model should always run
        self.next_ts["surface_flow"] = self.nextstep
        # Force a record step at the end of the simulation
        # The final step is taken in finalize() because the loop stops at the penultimate step
        self.next_ts["record"] = min(self.next_ts["end"], self.next_ts["record"])
        # If a Record is due, force hydrology
        self.next_ts["hydrology"] = min(self.next_ts["hydrology"], self.next_ts["record"])
        self.dt = self.nextstep - self.sim_time
        return self

    def get_continuity_data(self) -> ContinuityData:
        """ """
        cell_area = self.raster_domain.cell_area
        new_domain_vol = rastermetrics.calculate_total_volume(
            depth_array=self.raster_domain.get_array("h"), cell_surface_area=cell_area
        )
        volume_change = new_domain_vol - self.old_domain_volume
        volume_error = rastermetrics.calculate_total_volume(
            self.raster_domain.get_array("error_depth_accum"), cell_area
        )
        continuity_error = rastermetrics.calculate_continuity_error(
            volume_error=volume_error, volume_change=volume_change
        )
        return ContinuityData(new_domain_vol, volume_change, volume_error, continuity_error)

    def _update_accum_array(self, k: str, sim_time: datetime) -> None:
        """Update the accumulation arrays."""
        ak = self.accum_corresp[k]
        if self.accum_update_time[ak] is None:
            self.accum_update_time[ak] = sim_time
        time_diff = (sim_time - self.accum_update_time[ak]).total_seconds()
        if time_diff > 0:
            rate_array = self.raster_domain.get_array(k)
            accum_array = self.raster_domain.get_array(ak)
            rastermetrics.accumulate_rate_to_total(accum_array, rate_array, time_diff)
            self.accum_update_time[ak] = sim_time
