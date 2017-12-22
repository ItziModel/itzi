#!/usr/bin/env python
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
from __future__ import absolute_import
import sys
import os
import time
import subprocess
import traceback
from multiprocessing import Process
from datetime import timedelta
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
        """prepare the simulation, run it and clean up
        """
        if self.prof:
            self.prof.start()
        # If run outside of grass, set it
        if self.grass_use_file:
            self.set_grass_session()
        import itzi.gis as gis
        msgr.debug('GRASS session set')
        # return error if output files exist
        # (should be done once GRASS set up)
        gis.check_output_files(self.conf.output_map_names.itervalues())
        msgr.debug('Output files OK')
        # stop program if location is latlong
        if gis.is_latlon():
            msgr.fatal(u"latlong location is not supported. "
                       u"Please use a projected location")
        # set region
        if self.conf.grass_params['region']:
            gis.set_temp_region(self.conf.grass_params['region'])
        # set mask
        if self.conf.grass_params['mask']:
            gis.set_temp_mask(self.conf.grass_params['mask'])
        # Run simulation (SimulationManager needs GRASS, so imported now)
        from itzi.simulation import SimulationManager
        sim = SimulationManager(sim_times=self.conf.sim_times,
                                stats_file=self.conf.stats_file,
                                dtype=np.float32,
                                input_maps=self.conf.input_map_names,
                                output_maps=self.conf.output_map_names,
                                sim_param=self.conf.sim_param,
                                drainage_params=self.conf.drainage_params)
        sim.run()
        # return to previous region and mask
        if self.conf.grass_params['region']:
            gis.del_temp_region()
        if self.conf.grass_params['mask']:
            gis.del_temp_mask()
        # Delete the rcfile
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
        error_msg = u"'{}' does not exist or does not have adequate permissions"
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
        grass_python = os.path.join(gisbase, u"etc", u"python")
        sys.path.append(grass_python)

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
    return stdout.strip().decode(encoding='UTF-8')


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

    ld_base = os.path.join(gisbase, u"lib")
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
    except Exception as exc:
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
        import grass.script
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
    if args.q and args.q == 2:
        os.environ['ITZI_VERBOSE'] = str(VerbosityLevel.SUPER_QUIET)
    elif args.q == 1:
        os.environ['ITZI_VERBOSE'] = str(VerbosityLevel.QUIET)
    elif args.v == 1:
        os.environ['ITZI_VERBOSE'] = str(VerbosityLevel.VERBOSE)
    elif args.v and args.v >= 2:
        os.environ['ITZI_VERBOSE'] = str(VerbosityLevel.DEBUG)
    else:
        os.environ['ITZI_VERBOSE'] = str(VerbosityLevel.MESSAGE)

    # setting GRASS verbosity (especially for maps registration)
    if args.q and args.q >= 1:
        # no warnings
        os.environ['GRASS_VERBOSE'] = '-1'
    elif args.v and args.v >= 1:
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
    for f, t in times_dict.items():
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


if __name__ == "__main__":
    sys.exit(main())
