#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NAME:      Itzï

AUTHOR(S): Laurent Courty

PURPOSE:   A distributed GIS computer model for flood simulation. See:
           Courty, L. G., Pedrozo-Acuña, A., and Bates, P. D.:
           Itzï (version 16.8): An open-source,
           distributed GIS model for dynamic flood simulation,
           Geosci. Model Dev. Discuss., doi:10.5194/gmd-2016-283, in review, 2016.

COPYRIGHT: (C) 2015-2017 by Laurent Courty

            This program is free software; you can redistribute it and/or
            modify it under the terms of the GNU General Public License
            as published by the Free Software Foundation; either version 2
            of the License, or (at your option) any later version.

            This program is distributed in the hope that it will be useful,
            but WITHOUT ANY WARRANTY; without even the implied warranty of
            MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
            GNU General Public License for more details.
"""


from __future__ import print_function, division
import sys
import os
import argparse
import time
import subprocess
import traceback
import numpy as np
from multiprocessing import Process
from pyinstrument import Profiler
from datetime import timedelta

from configreader import ConfigReader
import itzi_error
import messenger as msgr
from const import *

def main():
    args = parser.parse_args()
    args.func(args)


class SimulationRunner(object):
    """provide the necessary tools to run one simulation
    including reading configuration file, setting-up GRASS
    """
    def __init__(self, conf, grass_use_file, args):
        self.conf = conf
        self.grass_use_file = grass_use_file
        if args.p:
            self.prof = Profiler()
        else:
            self.prof = None

    def run(self):
        """
        """
        if self.prof:
            self.prof.start()
        # If run outside of grass, set it
        if self.grass_use_file:
            self.set_grass_session()
        import grass.script as gscript
        msgr.debug('gscript done')
        # return error if output files exist
        # (should be done once GRASS set up)
        self.conf.check_output_files()
        msgr.debug('files check')
        # stop program if location is latlong
        if gscript.locn_is_latlong():
            msgr.fatal(u"latlong location is not supported. "
                       u"Please use a projected location")
        # Run simulation (SimulationManager needs GRASS, so imported now)
        from simulation import SimulationManager
        sim = SimulationManager(sim_times=self.conf.sim_times,
                                stats_file=self.conf.stats_file,
                                dtype=np.float32,
                                input_maps=self.conf.input_map_names,
                                output_maps=self.conf.output_map_names,
                                sim_param=self.conf.sim_param,
                                drainage_params=self.conf.drainage_params)
        sim.run()
        # delete the rcfile
        if self.grass_use_file:
            os.remove(self.rcfile)
        # end profiling and print results
        if self.prof:
            self.prof.stop()
            print(self.prof.output_text(unicode=True, color=True))
        return self

    def set_grass_session(self):
        """Inspired by example on GRASS wiki
        """
        grassbin = self.conf.grass_params['grass_bin']
        gisdb = self.conf.grass_params['grassdata']
        location = self.conf.grass_params['location']
        mapset = self.conf.grass_params['mapset']

        # check if the given parameters exist and can be accessed
        error_msg = u"'{}' does not exist or do not have adequate permissions"
        if not os.access(gisdb, os.R_OK):
            msgr.fatal(error_msg.format(gisdb))
        elif not os.access(os.path.join(gisdb, location), os.R_OK):
            msgr.fatal(error_msg.format(location))
        elif not os.access(os.path.join(gisdb, location, mapset), os.W_OK):
            msgr.fatal(error_msg.format(mapset))

        # query GRASS 7 itself for its GISBASE
        gisbase = get_gisbase(grassbin)

        # Set GISBASE environment variable
        os.environ['GISBASE'] = gisbase

        # define GRASS Python environment
        sys.path.append(os.path.join(gisbase, "etc", "python"))

        # launch session
        import grass.script.setup as gsetup
        self.rcfile = gsetup.init(gisbase, gisdb, location, mapset)
        return self


def get_gisbase(grassbin):
    """query GRASS 7 itself for its GISBASE
    """
    startcmd = [grassbin, '--config', 'path']
    try:
        p = subprocess.Popen(startcmd, shell=False,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
    except OSError as error:
        msgr.fatal("Cannot find GRASS GIS binary"
                   " '{cmd}' {error}".format(cmd=startcmd[0], error=error))
    if p.returncode != 0:
        msgr.fatal("Error while running GRASS GIS start-up script"
                   " '{cmd}': {error}".format(cmd=' '.join(startcmd), error=stderr))
    return stdout.strip(os.linesep)


def set_ldpath(gisbase):
    """Add GRASS libraries to the dynamic library path
    And then re-execute the process to take the changes into account
    """
    # choose right path for each platform
    if (sys.platform.startswith('linux') or
            'bsd' in sys.platform or
            'solaris' in sys.platform):
        ldvar = 'LD_LIBRARY_PATH'
    elif sys.platform.startswith('win'):
        ldvar = 'PATH'
    elif sys.platform == 'darwin':
        ldvar = 'DYLD_LIBRARY_PATH'
    else:
        msgr.fatal("Platform not configured: {}".format(sys.platform))

    ld_base = os.path.join(gisbase, "lib")
    if not os.environ.get(ldvar):
        # if the path variable is not set
        msgr.debug("{} not set. Setting and restart".format(ldvar))
        os.environ[ldvar] = ld_base
        reexec()
    elif ld_base not in os.environ[ldvar]:
        msgr.debug("{} not in {}. Setting and restart".format(ld_base, ldvar))
        # if the variable exists but does not have the path
        os.environ[ldvar] += os.pathsep + ld_base
        reexec()


def reexec():
    """Re-execute the software with the same arguments
    """
    args = [sys.executable] + sys.argv
    try:
        os.execv(sys.executable, args)
    except Exception, exc:
        msgr.fatal(u"Failed to re-execute: {}".format(exc))


def sim_runner_worker(conf, grass_use_file, grassbin):
    msgr.raise_on_error = True
    try:
        sim_runner = SimulationRunner(conf, grass_use_file, grassbin)
        sim_runner.run()
    except itzi_error.ItziError:
        # if an Itzï error, only print the last line of the traceback
        traceback_lines = traceback.format_exc().splitlines()
        msgr.warning("Error during execution: {}".format(traceback_lines[-1]))
    except:
        msgr.warning("Error during execution: {}".format(traceback.format_exc()))


def itzi_run(args):
    # Check if being run within GRASS session
    try:
        import grass.script as gscript
    except ImportError:
        grass_use_file = True
    else:
        grass_use_file = False

    # set environment variables
    if args.o:
        os.environ['GRASS_OVERWRITE'] = '1'
    else:
        os.environ['GRASS_OVERWRITE'] = '0'
    # verbosity
    if args.q == 2:
        os.environ['ITZI_VERBOSE'] = str(SUPER_QUIET)
    elif args.q == 1:
        os.environ['ITZI_VERBOSE'] = str(QUIET)
    elif args.v == 1:
        os.environ['ITZI_VERBOSE'] = str(VERBOSE)
    elif args.v >= 2:
        os.environ['ITZI_VERBOSE'] = str(DEBUG)
    else:
        os.environ['ITZI_VERBOSE'] = str(MESSAGE)

    # setting GRASS verbosity (especially for maps registration)
    if args.q >= 1:
        # no warnings
        os.environ['GRASS_VERBOSE'] = '-1'
    elif args.v >= 1:
        # normal
        os.environ['GRASS_VERBOSE'] = '2'
    else:
        # only warnings
        os.environ['GRASS_VERBOSE'] = '0'

    # start total time counter
    total_sim_start = time.time()
    # dictionary to store computation times
    times_dict = {}
    for conf_file in args.config_file:
        # parsing configuration file
        conf = ConfigReader(conf_file)
        grassbin = conf.grass_params['grass_bin']
        # if outside from GRASS, set path to shared libraries and restart
        if grass_use_file and not grassbin:
            msgr.fatal(u"Please define [grass] section in parameter file")
        elif grass_use_file:
            set_ldpath(get_gisbase(grassbin))
        file_name = os.path.basename(conf_file)
        msgr.message(u"Starting simulation of {}...".format(file_name))
        # display parameters (if verbose)
        conf.display_sim_param()
        # run in a subprocess
        sim_start = time.time()
        worker_args = (conf, grass_use_file, args)
        p = Process(target=sim_runner_worker, args=worker_args)
        p.start()
        p.join()
        if p.exitcode != 0:
            msgr.warning((u"Execution of {} "
                          u"ended with an error").format(file_name))
        # store computational time
        comp_time = timedelta(seconds=int(time.time() - sim_start))
        times_dict[file_name] = comp_time

    # stop total time counter
    total_elapsed_time = timedelta(seconds=int(time.time() - total_sim_start))
    # display total computation duration
    msgr.message(u"Simulations complete. Elapsed times:")
    for f, t in times_dict.iteritems():
        msgr.message(u"{}: {}".format(f, t))
    msgr.message(u"Total: {}".format(total_elapsed_time))
    avg_time_s = int(total_elapsed_time.total_seconds() / len(times_dict))
    msgr.message(u"Average: {}".format(timedelta(seconds=avg_time_s)))


def itzi_version(args):
    """Display the software version number from a file
    """
    ROOT = os.path.dirname(__file__)
    F_VERSION = os.path.join(ROOT, 'data', 'VERSION')
    with open(F_VERSION, 'r') as f:
        print(f.readline().strip())


########################
# Parsing command line #
########################

DESCR = (u"A dynamic, fully distributed hydraulic and hydrologic model.")

parser = argparse.ArgumentParser(description=DESCR)
subparsers = parser.add_subparsers()

# running a simulation
run_parser = subparsers.add_parser("run", help=u"run a simulation",
                                   description="run a simulation")
run_parser.add_argument("config_file", nargs='+',
                        help=(u"an Itzï configuration file "
                              u"(if several given, run in batch mode)"))
run_parser.add_argument("-o", action='store_true', help=u"overwrite files if exist")
run_parser.add_argument("-p", action='store_true', help=u"activate profiler")
verbosity_parser = run_parser.add_mutually_exclusive_group()
verbosity_parser.add_argument("-v", action='count', help=u"increase verbosity")
verbosity_parser.add_argument("-q", action='count', help=u"decrease verbosity")
run_parser.set_defaults(func=itzi_run)

# display version
version_parser = subparsers.add_parser("version",
                                       help=u"display software version number")
version_parser.set_defaults(func=itzi_version)


if __name__ == "__main__":
    sys.exit(main())
