#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import subprocess
from importlib.metadata import version
from multiprocessing import Process
from datetime import datetime, timedelta

from pyinstrument import Profiler
import numpy as np

from itzi.configreader import ConfigReader
import itzi.itzi_error as itzi_error
import itzi.messenger as msgr
from itzi.const import VerbosityLevel
from itzi import parser


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
    """Provide the necessary tools to run one simulation,
    including setting-up and tearing down GRASS session.
    """

    def __init__(self):
        self.conf = None
        self.sim = None
        self.grass_session = None
        self.grass_required_version = "8.4.0"

    def initialize(self, conf_file):
        """Parse the configuration file, set GRASS,
        and initialize the simulation.
        """
        # parsing configuration file
        self.conf = ConfigReader(conf_file)

        # display parameters (if verbose)
        msgr.message(f"Starting simulation of {os.path.basename(conf_file)}...")
        self.conf.display_sim_param()

        # If run outside of grass, set session
        try:
            import grass.script as gscript
            from itzi.simulation import create_simulation
        except ImportError:
            self.set_grass_session()
            from itzi.simulation import create_simulation
        # Check GRASS version
        grass_version = gscript.parse_command("g.version", flags="g")["version"]
        if grass_version < self.grass_required_version:
            msgr.fatal(
                (
                    f"Itzi requires at least GRASS {self.grass_required_version}, "
                    "version {grass_version} detected."
                )
            )
        msgr.debug("GRASS session set")

        # Instantiate Simulation object and initialize it
        self.sim, self.tarr = create_simulation(
            sim_times=self.conf.sim_times,
            stats_file=self.conf.stats_file,
            dtype=np.float32,
            input_maps=self.conf.input_map_names,
            output_maps=self.conf.output_map_names,
            sim_param=self.conf.sim_param,
            drainage_params=self.conf.drainage_params,
            grass_params=self.conf.grass_params,
        )
        self.update_input_arrays()
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
        # Close GRASS session
        if self.grass_session is not None:
            self.grass_session.finish()
        return self

    def step(self):
        self.sim.update()
        self.update_input_arrays()
        return self

    @property
    def origin(self):
        # Get a TimedArray object to reach the GIS interface
        tarr = next(iter(self.tarr.values()))
        return tarr.igis.origin

    def set_grass_session(self):
        """Set the GRASS session."""
        gisdb = self.conf.grass_params["grassdata"]
        location = self.conf.grass_params["location"]
        mapset = self.conf.grass_params["mapset"]
        if location is None:
            msgr.fatal(("[grass] section is missing."))

        # Check if the given parameters exist and can be accessed
        error_msg = "'{}' does not exist or does not have adequate permissions"
        if not os.access(gisdb, os.R_OK):
            msgr.fatal(error_msg.format(gisdb))
        elif not os.access(os.path.join(gisdb, location), os.R_OK):
            msgr.fatal(error_msg.format(location))
        elif not os.access(os.path.join(gisdb, location, mapset), os.W_OK):
            msgr.fatal(error_msg.format(mapset))

        # Set GRASS python path
        if self.conf.grass_params["grass_bin"]:
            grassbin = self.conf.grass_params["grass_bin"]
        else:
            grassbin = "grass"
        grass_cmd = [grassbin, "--config", "python_path"]
        grass_python_path = subprocess.check_output(grass_cmd, text=True).strip()
        sys.path.append(grass_python_path)
        # Now we can import grass modules
        import grass.script as gscript

        # set up session
        self.grass_session = gscript.setup.init(
            path=gisdb, location=location, mapset=mapset, grass_path=grassbin
        )

    def update_input_arrays(self):
        """Get new array using TimedArray
        And update
        """
        # make sure DEM is treated first
        if not self.tarr["dem"].is_valid(self.sim.sim_time):
            self.sim.set_array("dem", self.tarr["dem"].get(self.sim.sim_time))

        # loop through the arrays
        for k, ta in self.tarr.items():
            if not ta.is_valid(self.sim.sim_time):
                # z is done before
                if k == "dem":
                    continue
                # Convert mm/h to m/s
                if k in [
                    "rain",
                    "capillary_pressure",
                    "hydraulic_conductivity",
                    "in_inf",
                ]:
                    new_arr = ta.get(self.sim.sim_time) / (1000 * 3600)
                else:
                    new_arr = ta.get(self.sim.sim_time)
                # update array
                msgr.debug("{}: update input array <{}>".format(self.sim.sim_time, k))
                self.sim.set_array(k, new_arr)
        return self


def sim_runner_worker(conf_file, profile):
    """Run one simulation"""
    msgr.raise_on_error = True
    try:
        # Start profiler if requested
        if profile:
            prof = Profiler()
            prof.start()
        # Run the simulation
        sim_runner = SimulationRunner()
        sim_runner.initialize(conf_file).run().finalize()
        # end profiling and print results
        if profile:
            prof.stop()
            print(prof.output_text(unicode=True, color=True))
    except itzi_error.ItziError:
        # if an Itzï error, only print the last line of the traceback
        traceback_lines = traceback.format_exc().splitlines()
        msgr.warning("Error during execution: {}".format(traceback_lines[-1]))
    except Exception:
        msgr.warning("Error during execution: {}".format(traceback.format_exc()))


def itzi_run_one(conf_file, profile):
    """Run a simulation in a subprocess"""
    worker_args = (conf_file, profile)
    p = Process(target=sim_runner_worker, args=worker_args)
    p.start()
    p.join()
    if p.exitcode != 0:
        msgr.warning(("Execution of {} ended with an error").format(conf_file))
    p.close()


def itzi_run(cli_args):
    """Run one or multiple simulations from the command line."""
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

    profile = False
    if cli_args.p:
        profile = True

    # start total time counter
    total_sim_start = time.time()
    # dictionary to store computation times
    times_list = []
    for conf_file in cli_args.config_file:
        sim_start = time.time()
        # Run the simulation
        itzi_run_one(conf_file, profile)
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
