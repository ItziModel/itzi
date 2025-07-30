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
from typing import Self, Union, TYPE_CHECKING
import copy

import numpy as np

from itzi.data_containers import ContinuityData, SimulationData
import itzi.messenger as msgr
from itzi.itzi_error import NullError, MassBalanceError
from itzi import rastermetrics
from itzi.array_definitions import ARRAY_DEFINITIONS, ArrayCategory

if TYPE_CHECKING:
    from itzi.drainage import DrainageSimulation
    from itzi.hydrology import Hydrology
    from itzi.surfaceflow import SurfaceFlowSimulation
    from itzi.rasterdomain import RasterDomain
    from itzi.report import Report


class Simulation:
    """ """

    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        raster_domain: "RasterDomain",
        hydrology_model: "Hydrology",
        surface_flow: "SurfaceFlowSimulation",
        drainage_model: Union["DrainageSimulation", None],
        nodes_list: list | None,
        report: "Report",
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
        # A mapping between source array and the corresponding accumulation array
        self.accum_mapping: dict[str, str] = {
            arr_def.computes_from: arr_def.key
            for arr_def in ARRAY_DEFINITIONS
            if arr_def.computes_from is not None and ArrayCategory.ACCUMULATION in arr_def.category
        }
        self.accum_update_time: dict[str, datetime] = {
            accum: None for source, accum in self.accum_mapping.items()
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
                n.node_id: (n.row, n.col) for n in self.nodes_list if n.node_object.is_coupled()
            }
        # Grid spacing (for BMI)
        self.spacing = (self.raster_domain.dy, self.raster_domain.dx)
        # time step counter
        self.time_steps_counter: int = 0

    def initialize(self) -> Self:
        """Record the initial stage of the simulation, before time-stepping."""
        self.old_domain_volume = rastermetrics.calculate_total_volume(
            depth_array=self.raster_domain.get_padded("water_depth"),
            cell_surface_area=self.raster_domain.cell_area,
            padded=True,
        )
        for arr_key in self.accum_mapping.keys():
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
            arr_h = self.raster_domain.get_array("water_depth")
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
        for arr_key in self.accum_mapping.keys():
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
        for arr_key in self.accum_mapping.keys():
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
        relative_volume_threshold = 1e-5
        cell_area = self.raster_domain.cell_area
        new_domain_vol = rastermetrics.calculate_total_volume(
            depth_array=self.raster_domain.get_padded("water_depth"),
            cell_surface_area=cell_area,
            padded=True,
        )
        volume_change = new_domain_vol - self.old_domain_volume
        volume_error = rastermetrics.calculate_total_volume(
            depth_array=self.raster_domain.get_padded("error_depth_accum"),
            cell_surface_area=cell_area,
            padded=True,
        )

        if new_domain_vol > 0:
            relative_volume_change = volume_change / new_domain_vol
        else:
            relative_volume_change = 0

        if volume_error == 0:
            continuity_error = 0.0
        # Prevent returning artificially high error close to steady state
        elif abs(relative_volume_change) < relative_volume_threshold or volume_change == 0:
            continuity_error = float("nan")
        else:
            continuity_error = volume_error / volume_change

        return ContinuityData(new_domain_vol, volume_change, volume_error, continuity_error)

    def _update_accum_array(self, k: str, sim_time: datetime) -> None:
        """Update the accumulation arrays."""
        ak = self.accum_mapping[k]
        if self.accum_update_time[ak] is None:
            self.accum_update_time[ak] = sim_time
        time_diff = (sim_time - self.accum_update_time[ak]).total_seconds()
        if time_diff > 0:
            rate_array = self.raster_domain.get_padded(k)
            accum_array = self.raster_domain.get_padded(ak)
            rastermetrics.accumulate_rate_to_total(accum_array, rate_array, time_diff, padded=True)
            self.accum_update_time[ak] = sim_time
