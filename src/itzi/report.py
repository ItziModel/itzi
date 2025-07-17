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

import copy
import warnings

import numpy as np

from itzi import rastermetrics
from itzi.data_containers import SimulationData


class Report:
    """In charge of results reporting and writing"""

    def __init__(
        self,
        igis,
        temporal_type,
        hmin,
        mass_balance_logger,
        out_map_names,
        drainage_out,
        dt,
    ):
        self.record_counter = 0
        self.gis = igis
        self.temporal_type = temporal_type
        self.out_map_names = out_map_names
        self.hmin = hmin
        self.mass_balance_logger = mass_balance_logger
        self.drainage_out = drainage_out  # name of output map for the drainage data
        self.drainage_values = {"records": []}
        # a dict containing lists of maps written to gis to be registered
        self.output_maplist = {k: [] for k in self.out_map_names.keys()}
        self.vector_drainage_maplist = []
        self.output_arrays = {}
        self.dt = dt
        self.last_step = copy.copy(self.gis.start_time)

    def step(self, simulation_data: SimulationData):
        """write results at given time-step"""
        sim_time = simulation_data.sim_time
        self.get_output_arrays(simulation_data)
        self.write_results_to_gis(sim_time)
        if self.mass_balance_logger:
            self.write_mass_balance(simulation_data)
        drainage_data = simulation_data.drainage_network_data
        if drainage_data and self.drainage_out:
            self.save_drainage_values(sim_time, drainage_data)
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
        has_drainage_sim = bool(final_data.drainage_network_data)
        self.register_results_in_gis(has_drainage_sim)
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
        continuity_data = data.continuity_data
        # 1. Calculate all volumes using rastermetrics
        cell_area = data.cell_dx * data.cell_dy

        boundary_vol = rastermetrics.calculate_total_volume(
            data.accumulation_arrays["boundaries_accum"], cell_area
        )
        rain_vol = rastermetrics.calculate_total_volume(
            data.accumulation_arrays["rainfall_accum"], cell_area
        )
        infiltration_vol = rastermetrics.calculate_total_volume(
            data.accumulation_arrays["infiltration_accum"], cell_area
        )
        inflow_vol = rastermetrics.calculate_total_volume(
            data.accumulation_arrays["inflow_accum"], cell_area
        )
        losses_vol = rastermetrics.calculate_total_volume(
            data.accumulation_arrays["losses_accum"], cell_area
        )
        drain_net_vol = rastermetrics.calculate_total_volume(
            data.accumulation_arrays["drainage_network_accum"], cell_area
        )

        # 3. Assemble data and log
        interval_s = (data.sim_time - self.last_step).total_seconds()
        if data.time_steps_counter > 0:
            average_timestep = interval_s / data.time_steps_counter
        else:
            average_timestep = float("nan")
        report_data = {
            "simulation_time": data.sim_time,
            "average_timestep": average_timestep,
            "#timesteps": data.time_steps_counter,
            "boundary_volume": boundary_vol,
            "rainfall_volume": rain_vol,
            "infiltration_volume": -infiltration_vol,  # negative because it leaves the domain
            "inflow_volume": inflow_vol,
            "losses_volume": -losses_vol,  # negative because it leaves the domain
            "drainage_network_volume": drain_net_vol,
            "domain_volume": continuity_data.new_domain_vol,
            "volume_change": continuity_data.volume_change,
            "volume_error": continuity_data.volume_error,
            "percent_error": continuity_data.continuity_error,
        }
        self.mass_balance_logger.log(report_data)

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
                    # Do not apply nan to the internal array
                    arr = arr.copy()
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

    def register_results_in_gis(self, has_drainage_sim: bool):
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
        if has_drainage_sim and self.drainage_out:
            self.gis.register_maps_in_stds(
                stds_title="Itzï drainage results",
                stds_name=self.drainage_out,
                map_list=self.vector_drainage_maplist,
                stds_type="stvds",
                t_type=self.temporal_type,
            )
        return self

    def save_drainage_values(self, sim_time, drainage_data):
        """Write vector map of drainage network"""
        # format map name
        suffix = str(self.record_counter).zfill(4)
        map_name = f"{self.drainage_out}_{suffix}"
        # write the map
        self.gis.write_vector_map(drainage_data, map_name)
        # add map name and time to the list
        self.vector_drainage_maplist.append((map_name, sim_time))
        return self
