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

from datetime import datetime
import copy
import warnings
from typing import Dict
from dataclasses import dataclass

import numpy as np

from itzi import rastermetrics
from itzi import flow


@dataclass
class SimulationData:
    """Immutable data container for passing raw simulation state to Report.

    This is a pure data structure containing only the "raw ingredients"
    needed for a report. All report-specific calculations (e.g., WSE,
    average rates) are performed by the Report class itself.
    """

    sim_time: datetime
    time_step: float  # time step duration
    time_steps_counter: int  # number of time steps since last update
    raw_arrays: Dict[str, np.ndarray]
    accumulation_arrays: Dict[str, np.ndarray]
    cell_dx: float  # cell size in east-west direction
    cell_dy: float  # cell size in north-south direction


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
        self.step(final_data)
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
        accum_arrays = data.accumulation_arrays
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
                self.output_arrays["verror"] = accum_arrays["error_depth_accum"] * cell_area

        # --- Averaged accumulation arrays ---
        if interval_s <= 0:
            interval_s = data.time_step

        accum_maps = {
            "boundaries": "boundaries_accum",
            "inflow": "inflow_accum",
            "losses": "losses_accum",
            "drainage_stats": "drainage_network_accum",
        }
        for name, key in accum_maps.items():
            if self.out_map_names.get(name) and key in accum_arrays:
                map_mean = np.mean(accum_arrays[key])
                self.output_arrays[name] = rastermetrics.calculate_average_rate_from_total(
                    accum_arrays[key], interval_s, 1.0
                )

        rain_inf_map = {
            "rainfall": "rainfall_accum",
            "infiltration": "infiltration_accum",
        }
        ms_to_mmh = 1000 * 3600  # m/s to mm/h
        for name, key in rain_inf_map.items():
            if self.out_map_names.get(name) and key in accum_arrays:
                self.output_arrays[name] = rastermetrics.calculate_average_rate_from_total(
                    accum_arrays[key], interval_s, ms_to_mmh
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

        boundary_vol = flow.arr_sum(data.accumulation_arrays["boundaries_accum"])
        rain_vol = flow.arr_sum(data.accumulation_arrays["rainfall_accum"])
        inf_vol = -flow.arr_sum(data.accumulation_arrays["infiltration_accum"])
        inflow_vol = flow.arr_sum(data.accumulation_arrays["inflow_accum"])
        losses_vol = -flow.arr_sum(data.accumulation_arrays["losses_accum"])
        drain_net_vol = flow.arr_sum(data.accumulation_arrays["drainage_network_accum"])
        vol_error = rastermetrics.calculate_total_volume(
            data.accumulation_arrays["error_depth_accum"], cell_area
        )

        # 2. Calculate continuity error
        vol_change = new_domain_vol - self.old_domain_volume
        continuity_error = rastermetrics.calculate_continuity_error(vol_error, vol_change)

        # 3. Assemble data and log
        interval_s = (data.sim_time - self.last_step).total_seconds()
        if interval_s > 0:
            average_timestep = str(data.time_steps_counter / interval_s)
        else:
            average_timestep = "-"
        report_data = {
            "simulation_time": data.sim_time,
            "average_timestep": average_timestep,
            "#timesteps": data.time_steps_counter,
            "boundary_volume": boundary_vol,
            "rain_volume": rain_vol,
            "infiltration_volume": inf_vol,
            "inflow_volume": inflow_vol,
            "losses_volume": losses_vol,
            "drainage_network_volume": drain_net_vol,
            "domain_volume": new_domain_vol,
            "created_volume": vol_error,
            "percent_error": continuity_error,
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
