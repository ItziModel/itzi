#!/usr/bin/env python3
"""
NAME:      Itzï

AUTHOR(S): Laurent Courty

PURPOSE:   A distributed GIS computer model for flood simulation. See:
           Courty, L. G., Pedrozo-Acuña, A., & Bates, P. D. (2017).
           Itzï (version 17.1): an open-source,
           distributed GIS model for dynamic flood simulation.
           Geoscientific Model Development, 10(4), 1835–1847.
           https://doi.org/10.5194/gmd-10-1835-2017

COPYRIGHT: (C) 2015-2025 by Laurent Courty

            This program is free software; you can redistribute it and/or
            modify it under the terms of the GNU General Public License
            as published by the Free Software Foundation; either version 2
            of the License, or (at your option) any later version.

            This program is distributed in the hope that it will be useful,
            but WITHOUT ANY WARRANTY; without even the implied warranty of
            MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
            GNU General Public License for more details.
"""

import sys
import os
import time
import traceback
from importlib.metadata import version
from multiprocessing import Process
from datetime import datetime, timedelta

import numpy as np

from itzi.configreader import ConfigReader
import itzi.itzi_error as itzi_error
import itzi.messenger as msgr
from itzi.const import VerbosityLevel
from itzi import parser
from itzi.profiler import profile_context
from itzi.simulation_builder import SimulationBuilder
from itzi.grass_session import GrassSessionManager


def main():
    # default functions for subparsers
    parser.run_parser.set_defaults(func=itzi_run)
    parser.version_parser.set_defaults(func=itzi_version)
    # get parsed arguments
    args = parser.arg_parser.parse_args()
    try:
        args.func(args)
    except AttributeError:
        parser.arg_parser.print_usage()


class SimulationRunner:
    """Provide the necessary tools to run one simulation from a config file."""

    def __init__(self):
        self.conf = None
        self.sim = None
        self.grass_required_version = "8.4.0"
        self.input_wse = False

    def initialize(self, conf_data: ConfigReader):
        """Parse the configuration file, set GRASS,
        and initialize the simulation.
        """
        self.conf = conf_data
        sim_config = self.conf.get_sim_params()

        # display parameters (if verbose)
        self.conf.display_sim_param()

        if self.conf.input_map_names["water_surface_elevation"]:
            self.input_wse = True

        # Check GRASS version
        import grass.script as gscript

        grass_version = gscript.parse_command("g.version", flags="g")["version"]
        if grass_version < self.grass_required_version:
            msgr.fatal(
                (
                    f"Itzi requires at least GRASS {self.grass_required_version}, "
                    f"version {grass_version} detected."
                )
            )
        msgr.debug("GRASS session set")

        # return error if output files exist
        from itzi.providers import grass_interface

        grass_interface.check_output_files(self.conf.output_map_names.values())
        msgr.debug("Output files OK")

        data_type = np.float32
        # Create the grass_interface object
        self.g_interface = grass_interface.GrassInterface(
            start_time=self.conf.sim_times.start,
            end_time=self.conf.sim_times.end,
            dtype=data_type,
            region_id=self.conf.grass_params["region"],
            raster_mask_id=self.conf.grass_params["mask"],
            non_blocking_write=False,
        )
        # Create Simulation with GRASS backend
        msgr.verbose("Setting up GRASS simulation...")
        from itzi.providers.grass_input import GrassRasterInputProvider
        from itzi.providers.grass_output import GrassRasterOutputProvider
        from itzi.providers.grass_output import GrassVectorOutputProvider

        raster_input_provider = GrassRasterInputProvider(
            {
                "grass_interface": self.g_interface,
                "input_map_names": sim_config.input_map_names,
                "default_start_time": sim_config.start_time,
                "default_end_time": sim_config.end_time,
            }
        )

        raster_output_provider = GrassRasterOutputProvider(
            {
                "grass_interface": self.g_interface,
                "out_map_names": sim_config.output_map_names,
                "hmin": sim_config.surface_flow_parameters.hmin,
                "temporal_type": sim_config.temporal_type,
            }
        )
        vector_output_provider = GrassVectorOutputProvider(
            {
                "grass_interface": self.g_interface,
                "temporal_type": sim_config.temporal_type,
                "drainage_map_name": sim_config.drainage_output,
            }
        )

        self.sim, self.tarr = (
            SimulationBuilder(sim_config, self.g_interface.get_npmask(), data_type)
            .with_input_provider(raster_input_provider)
            .with_raster_output_provider(raster_output_provider)
            .with_vector_output_provider(vector_output_provider)
            .build()
        )
        # Initialize the simulation
        self.update_input_arrays()
        self.sim.initialize()
        return self

    def run(self):
        """Run a full simulation"""
        sim_start_time = datetime.now()
        msgr.verbose("Starting time-stepping...")
        while self.sim.sim_time < self.sim.end_time:
            # display advance of simulation
            msgr.percent(
                self.sim.start_time,
                self.sim.end_time,
                self.sim.sim_time,
                sim_start_time,
            )
            # step models
            self.step()
        return self

    def update_until(self, then):
        """Run the simulation until a time after start_time"""
        self.sim.update_until(then)
        return self

    def finalize(self):
        """Tear down the simulation and return to previous state."""
        self.sim.finalize()
        # Cleanup the grass interface object
        self.g_interface.finalize()
        self.g_interface.cleanup()
        return self

    def step(self):
        """Do one simulation step."""
        self.sim.update()
        self.update_input_arrays()
        return self

    @property
    def origin(self):
        # Get origin from a TimedArray object
        tarr = next(iter(self.tarr.values()))
        return tarr.origin

    def update_input_arrays(self):
        """Get new array using TimedArray
        And update
        """
        # DEM is needed for WSE and rain routing direction
        if not self.tarr["dem"].is_valid(self.sim.sim_time):
            self.sim.set_array("dem", self.tarr["dem"].get(self.sim.sim_time))
        # loop through the arrays
        for arr_key, ta in self.tarr.items():
            # DEM done before
            if arr_key == "dem":
                continue
            # WSE is updating water depth, either one of the other should update
            if (arr_key == "water_depth" and self.input_wse) or (
                arr_key == "water_surface_elevation" and not self.input_wse
            ):
                continue
            if not ta.is_valid(self.sim.sim_time):
                # Convert mm/h to m/s
                if arr_key in [
                    "rain",
                    "hydraulic_conductivity",
                    "infiltration",
                    "losses",
                ]:
                    new_arr = ta.get(self.sim.sim_time) / (1000 * 3600)
                # Convert mm to m
                elif arr_key in [
                    "capillary_pressure",
                ]:
                    new_arr = ta.get(self.sim.sim_time) / 1000
                else:
                    new_arr = ta.get(self.sim.sim_time)
                # update array
                msgr.debug("{}: update input array <{}>".format(self.sim.sim_time, arr_key))
                self.sim.set_array(arr_key, new_arr)
        return self


def sim_runner_worker(conf_file):
    """Run one simulation"""
    msgr.raise_on_error = True
    try:
        # Run the simulation
        msgr.message(f"Starting simulation of {os.path.basename(conf_file)}...")
        conf_data = ConfigReader(conf_file)
        with GrassSessionManager(conf_data.grass_params):
            with profile_context():
                sim_runner = SimulationRunner()
                sim_runner.initialize(conf_data).run().finalize()
    except itzi_error.ItziError:
        # if an Itzï error, only print the last line of the traceback
        traceback_lines = traceback.format_exc().splitlines()
        msgr.warning("Error during execution: {}".format(traceback_lines[-1]))
    except Exception:
        msgr.warning("Error during execution: {}".format(traceback.format_exc()))


def itzi_run_one(conf_file):
    """Run a simulation in a subprocess"""
    worker_args = (conf_file,)
    p = Process(target=sim_runner_worker, args=worker_args)
    p.start()
    p.join()
    if p.exitcode != 0:
        msgr.warning(("Execution of {} ended with an error").format(conf_file))
    p.close()


def itzi_run(cli_args):
    """Run one or multiple simulations from the command line."""
    # Do not raise on error when run from CLI
    msgr.raise_on_error = False
    # set environment variables
    if cli_args.o:
        os.environ["GRASS_OVERWRITE"] = "1"
    else:
        os.environ["GRASS_OVERWRITE"] = "0"
    # verbosity
    if cli_args.q and cli_args.q == 2:
        os.environ["ITZI_VERBOSE"] = str(VerbosityLevel.SUPER_QUIET)
    elif cli_args.q == 1:
        os.environ["ITZI_VERBOSE"] = str(VerbosityLevel.QUIET)
    elif cli_args.v == 1:
        os.environ["ITZI_VERBOSE"] = str(VerbosityLevel.VERBOSE)
    elif cli_args.v and cli_args.v >= 2:
        os.environ["ITZI_VERBOSE"] = str(VerbosityLevel.DEBUG)
    else:
        os.environ["ITZI_VERBOSE"] = str(VerbosityLevel.MESSAGE)

    # setting GRASS verbosity (especially for maps registration)
    if cli_args.q and cli_args.q >= 1:
        # no warnings
        os.environ["GRASS_VERBOSE"] = "-1"
    elif cli_args.v and cli_args.v >= 1:
        # normal
        os.environ["GRASS_VERBOSE"] = "2"
    else:
        # only warnings
        os.environ["GRASS_VERBOSE"] = "0"

    # start total time counter
    total_sim_start = time.time()
    # dictionary to store computation times
    times_list = []
    for conf_file in cli_args.config_file:
        sim_start = time.time()
        # Run the simulation
        itzi_run_one(conf_file)
        # store computational time
        comp_time = timedelta(seconds=int(time.time() - sim_start))
        list_elem = (os.path.basename(conf_file), comp_time)
        times_list.append(list_elem)

    # stop total time counter
    total_elapsed_time = timedelta(seconds=int(time.time() - total_sim_start))
    # display total computation duration
    msgr.message("Simulation(s) complete. Elapsed times:")
    for f, t in times_list:
        msgr.message("{}: {}".format(f, t))
    msgr.message("Total: {}".format(total_elapsed_time))
    avg_time_s = int(total_elapsed_time.total_seconds() / len(times_list))
    msgr.message("Average: {}".format(timedelta(seconds=avg_time_s)))


def itzi_version(cli_args):
    """Display the software version number from the installed version"""
    print(version("itzi"))


if __name__ == "__main__":
    sys.exit(main())
