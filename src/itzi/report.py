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

import numpy as np

from itzi import rastermetrics
from itzi.data_containers import SimulationData


class Report:
    """In charge of results reporting and writing"""

    def __init__(
        self,
        start_time,
        raster_output_provider,
        vector_output_provider,
        mass_balance_logger,
        out_map_names,
        dt,
    ):
        self.record_counter = 0
        self.raster_provider = raster_output_provider
        self.vector_provider = vector_output_provider
        self.out_map_names = out_map_names
        self.mass_balance_logger = mass_balance_logger
        # a dict containing lists of maps written to gis to be registered
        self.output_maplist = {k: [] for k in self.out_map_names.keys()}
        self.output_arrays = {}
        self.dt = dt
        self.last_step = copy.copy(start_time)

    def step(self, simulation_data: SimulationData):
        """write results at given time-step"""
        sim_time = simulation_data.sim_time
        self.get_output_arrays(simulation_data)
        self.save_array(sim_time)
        if self.mass_balance_logger:
            self.write_mass_balance(simulation_data)
        drainage_data = simulation_data.drainage_network_data
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
        # Write last maps
        self.raster_provider.finalize(final_data)
        self.vector_provider.finalize(final_data.drainage_network_data)
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

    def save_array(self, sim_time):
        """ """
        for k, arr in self.output_arrays.items():
            if isinstance(arr, np.ndarray):
                self.raster_provider.write_array(array=arr, map_key=k, sim_time=sim_time)
        return self

    def save_drainage_values(self, sim_time, drainage_data):
        """Write vector map of drainage network"""
        self.vector_provider.write_vector(drainage_data, sim_time)
        return self
