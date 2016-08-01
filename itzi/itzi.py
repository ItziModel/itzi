#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NAME:      Itzï

AUTHOR(S): Laurent Courty

PURPOSE:   Simulate dynamic superficial water flows using a simplified
           quasi-2D implementation of the Shallow Water Equations.
           See:
           De Almeida, G. & Bates, P., 2013. Applicability of the local
           inertial approximation of the shallow water equations to
           flood modeling. Water Resources Research, 49(8), pp.4833–4844.
           Sampson, C.C. et al., 2013. An automated routing methodology
           to enable direct rainfall in high resolution shallow water models.
           Hydrological Processes, 27(3), pp.467–476.

COPYRIGHT: (C) 2015-2016 by Laurent Courty

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

# exit with an error if run outside GRASS shell
try:
    import grass.script as grass
    from grass.pygrass.messages import Messenger
except ImportError:
    sys.exit("Please run from a GRASS GIS environment")

import os
import time
import argparse
import msgpack
import numpy as np
from pyinstrument import Profiler
from datetime import datetime, timedelta

import simulation
from configreader import ConfigReader
from resultsreader import ResultsReader


def main():
    args = parser.parse_args()
    args.func(args)


def itzi_run(args):
    # start profiler
    if args.p:
        prof = Profiler()
        prof.start()

    # set environment variables
    if args.o:
        os.environ['GRASS_OVERWRITE'] = '1'
    else:
        os.environ['GRASS_OVERWRITE'] = '0'
    if args.v:
        os.environ['GRASS_VERBOSE'] = '3'
    else:
        os.environ['GRASS_VERBOSE'] = '2'

    # start messenger
    msgr = Messenger()

    # stop program if location is latlong
    if grass.locn_is_latlong():
        msgr.fatal(_(u"latlong location is not supported. "
                     u"Please use a projected location"))

    # parsing configuration file
    conf = ConfigReader(args.config_file, msgr)
    # display parameters (if verbose)
    conf.display_sim_param()

    # Run simulation
    msgr.verbose(_(u"Starting simulation..."))
    sim_start = time.time()
    sim = simulation.SimulationManager(sim_times=conf.sim_times,
                                       stats_file=conf.stats_file,
                                       dtype=np.float32,
                                       input_maps=conf.input_map_names,
                                       output_maps=conf.output_map_names,
                                       sim_param=conf.sim_param,
                                       drainage_params=conf.drainage_params)
    sim.run()

    # end profiling and print results
    if args.p:
        prof.stop()
        print(prof.output_text(unicode=True, color=True))
    # display total computation duration
    elapsed_time = timedelta(seconds=int(time.time() - sim_start))
    grass.message(_(u"Simulation complete. "
                    u"Elapsed time: {}").format(elapsed_time))


def itzi_version(args):
    """Display the software version number from a file
    """
    ROOT = os.path.dirname(__file__)
    F_VERSION = os.path.join(ROOT, 'data', 'VERSION')
    with open(F_VERSION, 'r') as f:
        print(f.readline().strip())


def itzi_read(args):
    msgr = Messenger()
    # read input and affect variables
    with open(args.result_file, 'r') as infile:
        results = msgpack.load(infile)
    processor = ResultsReader(results, msgr)

    # perform actions
    if args.type == 'node':
        processor.verif_node_id(args.id)
        if args.action == 'plot':
            processor.plot_node_values(args.id, args.variables)
        elif args.action == 'csv':
            processor.node_values_to_csv(args.id, args.output)
    elif args.type == 'link':
        pass
    else:
        self.msgr.fatal(_(u"Unknown type: '{}'".format(args.type)))
    return None


# parsing command line
parser = argparse.ArgumentParser(description=u"A dynamic, fully distributed "
                                             u"hydraulic and hydrologic model")
subparsers = parser.add_subparsers()

# running a simple simulation
run_parser = subparsers.add_parser("run", help=u"run a simulation",
                                   description="run a simulation")
run_parser.add_argument("config_file", help=u"an Itzï configuration file")
run_parser.add_argument("-o", action='store_true', help=u"overwrite files if exist")
run_parser.add_argument("-p", action='store_true', help=u"activate profiler")
run_parser.add_argument("-v", action='store_true', help=u"verbose output")
run_parser.set_defaults(func=itzi_run)

# display version
version_parser = subparsers.add_parser("version",
                                       help=u"display software version number")
version_parser.set_defaults(func=itzi_version)

# read results
read_parser = subparsers.add_parser("read", help=u"read simulation results",
                                   description=u"read simulation results")
read_parser.add_argument("result_file", help=u"an Itzï results file")
read_parser.add_argument("--output",
                         help=u"CSV file. If not given, print to standard output")
read_parser.add_argument("action", choices=['plot', 'csv'],
                         help=u"action to perform")
read_parser.add_argument("type", choices=['node', 'link'],
                         help=u"Type of object to read")
read_parser.add_argument("id", help=u"ID of object")
read_parser.add_argument("variables", nargs='*',
                         help=u"list of variables")
read_parser.set_defaults(func=itzi_read)


if __name__ == "__main__":
    sys.exit(main())
