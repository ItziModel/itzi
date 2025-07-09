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

import warnings
from datetime import datetime, timedelta
import copy
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Self
import numpy as np
import pyswmm

from itzi.surfaceflow import SurfaceFlowSimulation
import itzi.rasterdomain as rasterdomain
from itzi.massbalance import MassBalanceLogger
from itzi.drainage import DrainageSimulation, DrainageNode, DrainageLink, CouplingTypes
from itzi import SwmmInputParser
import itzi.messenger as msgr
import itzi.infiltration as infiltration
from itzi.hydrology import Hydrology
from itzi.itzi_error import NullError
from itzi import rastermetrics
import itzi.flow as flow


@dataclass
class SimulationData:
    """Immutable data container for passing raw simulation state to Report.

    This is a pure data structure containing only the "raw ingredients"
    needed for a report. All report-specific calculations (e.g., WSE,
    average rates) are performed by the Report class itself.
    """

    sim_time: datetime
    time_step: float
    raw_arrays: Dict[str, np.ndarray]
    statistical_arrays: Dict[str, np.ndarray]
    cell_dx: float  # cell size in east-west direction
    cell_dy: float  # cell size in north-south direction


DrainageNodeData = namedtuple("DrainageNodeData", ["id", "object", "x", "y", "row", "col"])


def get_nodes_list(pswmm_nodes, nodes_coor_dict, drainage_params, igis, g):
    """Check if the drainage nodes are inside the region and can be coupled.
    Return a list of DrainageNodeData
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
        drainage_node_data = DrainageNodeData(
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
    grass_params,
    dtype=np.float32,
    stats_file=None,
):
    """A factory function that returns a Simulation object."""
    import itzi.gis as gis

    msgr.verbose("Setting up models...")
    # return error if output files exist
    gis.check_output_files(output_maps.values())
    msgr.debug("Output files OK")
    # GIS interface
    igis = gis.Igis(
        start_time=sim_times.start,
        end_time=sim_times.end,
        dtype=dtype,
        mkeys=input_maps.keys(),
        region_id=grass_params["region"],
        raster_mask_id=grass_params["mask"],
    )
    arr_mask = igis.get_npmask()
    msgr.verbose("Reading maps information from GIS...")
    igis.read(input_maps)
    # Timed arrays
    tarr = {}
    # TimedArray expects a function as an init parameter
    zeros_array = lambda: np.zeros(shape=raster_shape, dtype=dtype)  # noqa: E731
    for k in in_k_corresp.keys():
        tarr[k] = rasterdomain.TimedArray(in_k_corresp[k], igis, zeros_array)
    msgr.debug("Setting up raster domain...")
    # RasterDomain
    raster_shape = (igis.yr, igis.xr)
    try:
        raster_domain = rasterdomain.RasterDomain(
            dtype=dtype,
            arr_mask=arr_mask,
            cell_shape=(igis.dx, igis.dy),
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
                "sim_time",
                "avg_timestep",
                "#timesteps",
                "boundary_vol",
                "rain_vol",
                "inf_vol",
                "inflow_vol",
                "losses_vol",
                "drain_net_vol",
                "domain_vol",
                "created_vol",
                "%error",
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
            all_nodes, nodes_coors_dict, drainage_params, igis, sim_param["g"]
        )
        # Create Link objects
        links_vertices_dict = swmm_inp.get_links_id_as_dict()
        links_list = get_links_list(pyswmm.Links(swmm_sim), links_vertices_dict, nodes_coors_dict)
        node_objects_only = [i.object for i in nodes_list]
        drainage = DrainageSimulation(swmm_sim, node_objects_only, links_list)
    else:
        nodes_list = None
        drainage = None
    # reporting object
    msgr.debug("Setting up reporting object...")
    report = Report(
        igis,
        sim_times.temporal_type,
        sim_param["hmin"],
        massbal,
        output_maps,
        drainage,
        drainage_params["output"],
        sim_times.record_step,
    )
    msgr.verbose("Models set up")
    simulation = Simulation(
        sim_times.start,
        sim_times.end,
        raster_domain,
        hydrology_model,
        surface_flow,
        drainage,
        nodes_list,
        report,
        mass_balance_error_threshold=0.05,
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
        self.old_domain_volume: float = 0.0
        self.mass_balance_error_threshold = mass_balance_error_threshold
        # Moved from RasterDomain: statistical array management
        self.stats_update_time: dict[str, datetime] = {
            "st_inf": None,
            "st_rain": None,
            "st_inflow": None,
            "st_losses": None,
            "st_ndrain": None,
        }
        self.stats_corresp: dict[str, str] = {
            "inf": "st_inf",
            "rain": "st_rain",
            "inflow": "st_inflow",
            "capped_losses": "st_losses",
            "n_drain": "st_ndrain",
        }
        # First time-step is forced
        self.dt = timedelta(seconds=0.001)
        self.nextstep = self.sim_time + self.dt
        # dict of next time-step (datetime object)
        self.next_ts = {"end": self.end_time}
        for k in ["hydrology", "surface_flow", "drainage", "record"]:
            self.next_ts[k] = self.start_time
        # case if no drainage simulation
        if not self.drainage_model:
            self.next_ts["drainage"] = self.end_time
        else:
            self.node_id_to_loc = {
                n.id: (n.row, n.col) for n in self.nodes_list if n.object.is_coupled()
            }
        # Grid spacing (for BMI)
        self.spacing = (self.raster_domain.dy, self.raster_domain.dx)

    def initialize(self) -> Self:
        """Record the initial stage of the simulation, before time-stepping.
        """
        self._populate_stat_array("rain", self.sim_time)
        self._populate_stat_array("inflow", self.sim_time)
        self._populate_stat_array("inf", self.sim_time)
        self._populate_stat_array("capped_losses", self.sim_time)
        # Package data into SimulationData object
        raw_arrays = {
            k: self.raster_domain.get_unmasked(k)
            for k in self.raster_domain.k_all
            if k not in self.raster_domain.k_stats
        }
        statistical_arrays = {
            k: self.raster_domain.get_unmasked(k) for k in self.raster_domain.k_stats
        }
        simulation_data = SimulationData(
            sim_time=self.sim_time,
            time_step=self.dt.total_seconds(),
            raw_arrays=raw_arrays,
            statistical_arrays=statistical_arrays,
            cell_dx=self.raster_domain.dx,
            cell_dy=self.raster_domain.dy,
        )
        # Pass data to the reporting module
        self.report.step(simulation_data)

        # d. Reset statistical accumulators
        self.raster_domain.reset_stats()
        for key in self.stats_update_time:
            self.stats_update_time[key] = self.sim_time
        return self

    def update(self) -> Self:
        # hydrology #
        if self.sim_time == self.next_ts["hydrology"]:
            self.hydrology_model.solve_dt()
            # calculate when will happen the next time-step
            self.next_ts["hydrology"] += self.hydrology_model.dt
            self.hydrology_model.step()
            # update stat array
            self._populate_stat_array("inf", self.sim_time)
            self._populate_stat_array("capped_losses", self.sim_time)

        # drainage #
        if self.sim_time == self.next_ts["drainage"] and self.drainage_model:
            self.drainage_model.step()
            # Update drainage nodes
            surface_states = {}
            cell_surf = self.raster_domain.cell_surf
            arr_z = self.raster_domain.get_array("dem")
            arr_h = self.raster_domain.get_array("h")
            for node_id, (row, col) in self.node_id_to_loc.items():
                surface_states[node_id] = {"z": arr_z[row, col], "h": arr_h[row, col]}
            coupling_flows = self.drainage_model.apply_coupling_to_nodes(surface_states, cell_surf)
            # update drainage array
            arr_qd = self.raster_domain.get_array("n_drain")
            for node_id, coupling_flow in coupling_flows.items():
                row, col = self.node_id_to_loc[node_id]
                arr_qd[row, col] = coupling_flow / cell_surf
            # update stat array
            self._populate_stat_array("n_drain", self.sim_time)
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
            self.report.write_error_to_gis(self.surface_flow.arr_err)
            msgr.fatal("{}: Null value detected in simulation, terminating".format(self.sim_time))
        # calculate when should happen the next surface time-step
        self.surface_flow.solve_dt()
        self.next_ts["surface_flow"] += self.surface_flow.dt

        # send current time-step duration to mass balance object
        # if self.report.massbal:
        #     self.report.massbal.add_value("tstep", self.dt.total_seconds())

        # 2. Perform a mass balance continuity check.
        self._check_mass_balance_error()

        # Reporting last to get simulated values #
        if self.sim_time == self.next_ts["record"]:
            msgr.verbose(f"{self.sim_time}: Writing output maps...")

            # Populate statistical arrays before packaging data
            self._populate_stat_array("rain", self.sim_time)
            self._populate_stat_array("inflow", self.sim_time)
            self._populate_stat_array("inf", self.sim_time)
            self._populate_stat_array("capped_losses", self.sim_time)

            # Package data into SimulationData object
            raw_arrays = {
                k: self.raster_domain.get_unmasked(k)
                for k in self.raster_domain.k_all
                if k not in self.raster_domain.k_stats
            }
            statistical_arrays = {
                k: self.raster_domain.get_unmasked(k) for k in self.raster_domain.k_stats
            }

            simulation_data = SimulationData(
                sim_time=self.sim_time,
                time_step=self.dt.total_seconds(),
                raw_arrays=raw_arrays,
                statistical_arrays=statistical_arrays,
                cell_dx=self.raster_domain.dx,
                cell_dy=self.raster_domain.dy,
            )

            # Pass data to the reporting module
            self.report.step(simulation_data)

            # Reset statistical accumulators
            self.raster_domain.reset_stats()
            for key in self.stats_update_time:
                self.stats_update_time[key] = self.sim_time
            # Update next time step
            self.next_ts["record"] += self.report.dt
        # Find next time step
        self.find_dt()
        # update simulation time
        self.sim_time += self.dt
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
        self._populate_stat_array("rain", self.sim_time)
        self._populate_stat_array("inflow", self.sim_time)
        self._populate_stat_array("inf", self.sim_time)
        self._populate_stat_array("capped_losses", self.sim_time)

        raw_arrays = {
            k: self.raster_domain.get_unmasked(k)
            for k in self.raster_domain.k_all
            if k not in self.raster_domain.k_stats
        }
        statistical_arrays = {
            k: self.raster_domain.get_unmasked(k) for k in self.raster_domain.k_stats
        }

        simulation_data = SimulationData(
            sim_time=self.sim_time,
            time_step=self.dt.total_seconds(),
            raw_arrays=raw_arrays,
            statistical_arrays=statistical_arrays,
            cell_dx=self.raster_domain.dx,
            cell_dy=self.raster_domain.dy,
        )
        # write final report
        self.report.end(simulation_data)

    def set_array(self, arr_id, arr):
        """Set an array of the simulation domain"""
        assert isinstance(arr_id, str)
        assert isinstance(arr, np.ndarray)
        if arr_id in ["inflow", "rain"]:
            self._populate_stat_array(arr_id, self.sim_time)
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
        self.next_ts["record"] = min(self.next_ts["end"], self.next_ts["record"])
        # If a Record is due, force hydrology
        self.next_ts["hydrology"] = min(self.next_ts["hydrology"], self.next_ts["record"])
        self.dt = self.nextstep - self.sim_time
        return self

    def _check_mass_balance_error(self) -> None:
        # Private method to perform the continuity error check.
        # It will calculate all volumes using `rastermetrics` and raise
        # MassBalanceError if the threshold is exceeded.
        pass

    def _populate_stat_array(self, k: str, sim_time: datetime) -> None:
        """Manage statistical array updates.
        """
        sk = self.stats_corresp[k]
        if self.stats_update_time[sk] is None:
            self.stats_update_time[sk] = sim_time
        time_diff = (sim_time - self.stats_update_time[sk]).total_seconds()
        if time_diff >= 0:
            # Use rastermetrics.accumulate_rate_to_total instead of flow.populate_stat_array
            from itzi.rastermetrics import accumulate_rate_to_total

            rate_array = self.raster_domain.get_array(k)
            stat_array = self.raster_domain.get_array(sk)
            accumulate_rate_to_total(stat_array, rate_array, time_diff)
            self.stats_update_time[sk] = sim_time


class Report:
    """In charge of results reporting and writing"""

    def __init__(
        self,
        igis,
        temporal_type,
        hmin,
        mass_balance_logger,
        out_map_names,
        drainage_sim,
        drainage_out,
        dt,
    ):
        self.record_counter = 0
        self.gis = igis
        self.temporal_type = temporal_type
        self.out_map_names = out_map_names
        self.hmin = hmin
        self.mass_balance_logger = mass_balance_logger
        self.drainage_sim = drainage_sim
        self.drainage_out = drainage_out
        self.drainage_values = {"records": []}
        # a dict containing lists of maps written to gis to be registered
        self.output_maplist = {k: [] for k in self.out_map_names.keys()}
        self.vector_drainage_maplist = []
        self.output_arrays = {}
        self.dt = dt
        self.last_step = copy.copy(self.gis.start_time)

        # For mass balance calculations
        self.old_domain_volume = None

    def step(self, simulation_data: SimulationData):
        """write results at given time-step"""
        sim_time = simulation_data.sim_time
        self.get_output_arrays(simulation_data)
        self.write_results_to_gis(sim_time)
        if self.mass_balance_logger:
            self.write_mass_balance(simulation_data)
        if self.drainage_sim and self.drainage_out:
            self.save_drainage_values(sim_time)
        self.record_counter += 1
        self.last_step = copy.copy(sim_time)
        return self

    def end(self, final_data: SimulationData):
        """
        register maps in gis
        write max level maps
        """
        # do the last step
        # self.step(final_data)
        # Make sure all maps are written in the background process
        self.gis.finalize()
        # register maps and write max maps
        self.register_results_in_gis()
        if self.out_map_names["h"]:
            self.write_hmax_to_gis(final_data.raw_arrays["hmax"])
        if self.out_map_names["v"]:
            self.write_vmax_to_gis(final_data.raw_arrays["vmax"])
        # Cleanup the GIS state
        self.gis.cleanup()
        return self

    def get_output_arrays(self, data: SimulationData):
        """Returns a dict of arrays to be written to the disk"""
        raw = data.raw_arrays
        stats = data.statistical_arrays
        interval_s = (data.sim_time - self.last_step).total_seconds()
        cell_dx = data.cell_dx
        cell_dy = data.cell_dy
        cell_area = cell_dx * cell_dy

        for k in self.out_map_names:
            if self.out_map_names[k] is None:
                continue

            # --- Direct raw arrays ---
            if k in ["h", "v", "vdir", "froude", "hmax", "vmax"]:
                if k in raw:
                    self.output_arrays[k] = raw[k]
                continue  # go to next key

            # --- Calculated arrays ---
            if k == "wse":
                self.output_arrays["wse"] = rastermetrics.calculate_wse(raw["h"], raw["dem"])
            elif k == "qx":
                self.output_arrays["qx"] = rastermetrics.calculate_flux(raw["qe_new"], cell_dy)
            elif k == "qy":
                self.output_arrays["qy"] = rastermetrics.calculate_flux(raw["qs_new"], cell_dx)
            elif k == "verror":  # Volume error
                self.output_arrays["verror"] = stats["st_herr"] * cell_area

        # --- Averaged statistical arrays ---
        if interval_s <= 0:
            interval_s = data.time_step

        stat_map = {
            "boundaries": "st_bound",
            "inflow": "st_inflow",
            "losses": "st_losses",
            "drainage_stats": "st_ndrain",
        }
        for name, key in stat_map.items():
            if self.out_map_names.get(name) and key in stats:
                map_mean = np.mean(stats[key])
                self.output_arrays[name] = rastermetrics.calculate_average_rate_from_total(
                    stats[key], interval_s, 1.0
                )

        rain_inf_map = {
            "rainfall": "st_rain",
            "infiltration": "st_inf",
        }
        ms_to_mmh = 1000 * 3600  # m/s to mm/h
        for name, key in rain_inf_map.items():
            if self.out_map_names.get(name) and key in stats:
                self.output_arrays[name] = rastermetrics.calculate_average_rate_from_total(
                    stats[key], interval_s, ms_to_mmh
                )
        return self

    def write_mass_balance(self, data: SimulationData):
        """Calculate mass balance and log it."""
        # This logic is migrated from the old MassBal class

        # 1. Calculate all volumes using rastermetrics
        cell_area = data.cell_dy * data.cell_dy
        new_domain_vol = rastermetrics.calculate_total_volume(data.raw_arrays["h"], cell_area)
        if self.old_domain_volume is None:
            # On first time step, old_domain_volume is equal to new_domain_volume
            self.old_domain_volume = new_domain_vol.copy()

        boundary_vol = flow.arr_sum(data.statistical_arrays["st_bound"])

        rain_vol = flow.arr_sum(data.statistical_arrays["st_rain"])
        inf_vol = -flow.arr_sum(data.statistical_arrays["st_inf"])
        inflow_vol = flow.arr_sum(data.statistical_arrays["st_inflow"])
        losses_vol = -flow.arr_sum(data.statistical_arrays["st_losses"])
        drain_net_vol = flow.arr_sum(data.statistical_arrays["st_ndrain"])
        vol_error = rastermetrics.calculate_total_volume(
            data.statistical_arrays["st_herr"], cell_area
        )

        # 2. Calculate continuity error
        vol_change = new_domain_vol - self.old_domain_volume
        continuity_error = rastermetrics.calculate_continuity_error(vol_error, vol_change)

        # 3. Assemble data and log
        report_data = {
            "sim_time": data.sim_time,
            "avg_timestep": data.raw_arrays.get(
                "avg_timestep", "-"
            ),  # Should be passed in simData
            "#timesteps": data.raw_arrays.get("#timesteps", 0),  # Should be passed in simData
            "boundary_vol": boundary_vol,
            "rain_vol": rain_vol,
            "inf_vol": inf_vol,
            "inflow_vol": inflow_vol,
            "losses_vol": losses_vol,
            "drain_net_vol": drain_net_vol,
            "domain_vol": new_domain_vol,
            "created_vol": vol_error,
            "%error": continuity_error,
        }
        self.mass_balance_logger.log(report_data)

        # 4. Update state for next step
        self.old_domain_volume = new_domain_vol
        return self

    def write_results_to_gis(self, sim_time):
        """Format the name of each maps using the record number as suffix
        Send a tuple (array, name, key) to the GIS writing function.
        """
        for k, arr in self.output_arrays.items():
            if isinstance(arr, np.ndarray):
                suffix = str(self.record_counter).zfill(4)
                map_name = "{}_{}".format(self.out_map_names[k], suffix)
                # Export depth if above hmin. If not, export NaN
                if k == "h":
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        arr[arr <= self.hmin] = np.nan
                # write the raster
                self.gis.write_raster_map(arr, map_name, k)
                # add map name and time to the corresponding list
                self.output_maplist[k].append((map_name, sim_time))
        return self

    def write_error_to_gis(self, arr_error):
        """Write a given boolean array to the GIS"""
        map_h_name = "{}_error".format(self.out_map_names["h"])
        self.gis.write_raster_map(arr_error, map_h_name, "h")
        # add map name to the revelant list
        self.output_maplist["h"].append(map_h_name)
        return self

    def write_hmax_to_gis(self, arr_hmax):
        """Write a max depth array to the GIS"""
        map_hmax_name = "{}_max".format(self.out_map_names["h"])
        self.gis.write_raster_map(arr_hmax, map_hmax_name, "h")
        return self

    def write_vmax_to_gis(self, arr_vmax):
        """Write a max flow speed array to the GIS"""
        map_vmax_name = "{}_max".format(self.out_map_names["v"])
        self.gis.write_raster_map(arr_vmax, map_vmax_name, "v")
        return self

    def register_results_in_gis(self):
        """Register the generated maps in the temporal database
        Loop through output names
        if no output name is provided, don't do anything
        if name is populated, create a strds of the right temporal type
        and register the corresponding listed maps
        """
        # rasters
        for mkey, lst in self.output_maplist.items():
            strds_name = self.out_map_names[mkey]
            if strds_name is None:
                continue
            self.gis.register_maps_in_stds(mkey, strds_name, lst, "strds", self.temporal_type)
        # vector
        if self.drainage_sim and self.drainage_out:
            self.gis.register_maps_in_stds(
                stds_title="Itzï drainage results",
                stds_name=self.drainage_out,
                map_list=self.vector_drainage_maplist,
                stds_type="stvds",
                t_type=self.temporal_type,
            )
        return self

    def save_drainage_values(self, sim_time):
        """Write vector map of drainage network"""
        # format map name
        suffix = str(self.record_counter).zfill(4)
        map_name = f"{self.drainage_out}_{suffix}"
        # write the map
        self.gis.write_vector_map(self.drainage_sim, map_name)
        # add map name and time to the list
        self.vector_drainage_maplist.append((map_name, sim_time))
        return self
