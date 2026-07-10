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

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from importlib.metadata import version
from multiprocessing import Process
from typing import TYPE_CHECKING, Callable

import numpy as np

from itzi.configreader import ConfigReader
import itzi.itzi_error as itzi_error
import itzi.messenger as msgr
from itzi.const import VerbosityLevel
from itzi.cli_parser import build_parser
from itzi.profiler import profile_context
from itzi.simulation_builder import SimulationBuilder
from itzi.grass_session import GrassSessionManager

if TYPE_CHECKING:
    from itzi.data_containers import SimulationConfig, GrassParams
    from itzi.providers.grass_interface import GrassInterface
    from itzi.simulation import Simulation


def main(argv=None):
    """argv: alternative CLI arguments, used for testing (default to sys.argv)"""
    args = build_parser().parse_args(argv)

    command_mapper: dict[str, Callable] = {
        "run": itzi_run,
        "version": itzi_version,
    }

    # args.command is the name of the subcommand
    command_mapper[args.command](args)


class SimulationRunner:
    """Provide the necessary tools to run one simulation."""

    def __init__(
        self,
        sim_config: SimulationConfig,
        grass_params: GrassParams,
        hotstart_path: str | None = None,
    ):
        self.grass_required_version = "8.4.0"
        self.g_interface: GrassInterface
        self.sim: Simulation

        # display parameters (if verbose)
        sim_config.display_sim_param()

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

        grass_interface.check_output_files(sim_config.output_map_names.values())
        msgr.debug("Output files OK")

        data_type = np.float32
        # Create the grass_interface object
        self.g_interface = grass_interface.GrassInterface(
            start_time=sim_config.start_time,
            end_time=sim_config.end_time,
            dtype=data_type,
            region_id=grass_params.region,
            raster_mask_id=grass_params.mask,
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

        sim_builder = (
            SimulationBuilder(sim_config, self.g_interface.get_npmask(), data_type)
            .with_input_provider(raster_input_provider)
            .with_raster_output_provider(raster_output_provider)
            .with_vector_output_provider(vector_output_provider)
        )
        if hotstart_path:
            sim_builder.with_hotstart(hotstart_path)
        self.sim: Simulation = sim_builder.build()
        # Initialize the simulation
        self.sim.initialize()

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

    def finalize(self):
        """Tear down the simulation and return to previous state."""
        self.sim.finalize()
        # Cleanup the grass interface object
        if hasattr(self, "g_interface"):
            self.g_interface.finalize()
            self.g_interface.cleanup()
        return self

    def step(self):
        """Do one simulation step."""
        self.sim.update()
        return self

    @property
    def origin(self):
        # Get origin from a TimedArray object
        tarr = next(iter(self.sim.timed_arrays.values()))
        return tarr.origin

    def __del__(self):
        # Cleanup the grass interface object
        if hasattr(self, "g_interface"):
            self.g_interface.finalize()
            self.g_interface.cleanup()


def sim_runner_worker(conf_file: str, hotstart_file: str | None):
    """Run one simulation"""
    msgr.raise_on_error = True
    msgr._itzi_logger.set_verbosity(msgr.verbosity())
    try:
        # Run the simulation
        msgr.message(f"Starting simulation of {os.path.basename(conf_file)}...")
        conf_data = ConfigReader(conf_file)
        sim_params = conf_data.get_sim_params()
        grass_params = conf_data.get_grass_params()
        with GrassSessionManager(grass_params):
            with profile_context():
                sim_runner = SimulationRunner(
                    sim_params,
                    grass_params,
                    hotstart_file,
                )
                sim_runner.run().finalize()
    except itzi_error.ItziError:
        # if an Itzï error, only print the last line of the traceback
        traceback_lines = traceback.format_exc().splitlines()
        msgr.warning("Error during execution: {}".format(traceback_lines[-1]))
    except Exception:
        msgr.warning("Error during execution: {}".format(traceback.format_exc()))


def itzi_run_one(conf_file: str, hotstart_file: str | None):
    """Run a simulation in a subprocess"""
    worker_args = (conf_file, hotstart_file)
    p = Process(target=sim_runner_worker, args=worker_args)
    p.start()
    p.join()
    if p.exitcode != 0:
        msgr.warning(("Execution of {} ended with an error").format(conf_file))
    p.close()


def reconcile_hotstart_commands(
    config_file_list: list[str],
    resume_from_list: list[tuple[str | None, str]],
) -> list[tuple[str, str | None]]:
    """Output a list of tuples in the form (config_file, hotstart_file)."""

    # No hotstart requested: run every config from scratch.
    if not resume_from_list:
        return [(config_file, None) for config_file in config_file_list]

    # An unnamed single hotstart only makes sense when there is a single config.
    if len(resume_from_list) == 1:
        config_key, hotstart_file = resume_from_list[0]
        if config_key is None:
            if len(config_file_list) != 1:
                msgr.fatal(
                    "A single unnamed --resume-from value can only be used with a single config file"
                )
            return [(config_file_list[0], hotstart_file)]

    # In batch mode every hotstart must be explicitly mapped.
    if any(config_key is None for config_key, _ in resume_from_list):
        msgr.fatal("When using multiple --resume-from values, each one must be CONFIG=PATH")

    # Accept a normalized config path first, then a unique basename like "a.ini".
    exact_lookup = {
        os.path.abspath(os.path.normpath(config_file)): config_file
        for config_file in config_file_list
    }
    basename_lookup: dict[str, str | None] = {}
    resolved_hotstarts = {config_file: None for config_file in config_file_list}

    # Build basename lookup; Store None when a basename is shared.
    for config_file in config_file_list:
        basename = os.path.basename(config_file)
        if basename not in basename_lookup:
            basename_lookup[basename] = config_file
        elif basename_lookup[basename] != config_file:
            basename_lookup[basename] = None

    # Resolve each CONFIG=PATH pair onto its target config file.
    for config_key, hotstart_file in resume_from_list:
        assert config_key is not None
        normalized_key = os.path.abspath(os.path.normpath(config_key))
        target_config = exact_lookup.get(normalized_key)
        if target_config is None:
            basename = os.path.basename(config_key)
            if basename not in basename_lookup:
                msgr.fatal(f"--resume-from config {config_key!r} does not match any config file")
            target_config = basename_lookup[basename]
            if target_config is None:
                msgr.fatal(f"--resume-from config {config_key!r} is ambiguous")

        if resolved_hotstarts[target_config] is not None:
            msgr.fatal(f"Multiple hotstart files were given for {target_config!r}")
        resolved_hotstarts[target_config] = hotstart_file

    return [(config_file, resolved_hotstarts[config_file]) for config_file in config_file_list]


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

    # start total time counter
    total_sim_start = time.time()
    # dictionary to store computation times
    times_list = []
    run_commands = reconcile_hotstart_commands(
        cli_args.config_file,
        getattr(cli_args, "resume_from", []),
    )
    for conf_file, hotstart_file in run_commands:
        sim_start = time.time()
        # Run the simulation
        itzi_run_one(conf_file, hotstart_file)
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
