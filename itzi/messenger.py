# coding=utf8
"""
Copyright (C) 2015-2016 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

from __future__ import division
from __future__ import print_function
import sys
import os
from datetime import timedelta, datetime

from itzi_error import ItziFatal

OUTPUT = sys.stderr
FATAL = "ERROR: "
WARNING = "WARNING: "
PAD = " " * 30  # Necessary to print a clean line

raise_on_error = False

def verbosity():
    return int(os.environ['ITZI_VERBOSE'])


def percent(start_time, end_time, sim_time, sim_start_time):
    """Display progress of the simulation
    """
    sim_time_s = (sim_time - start_time).total_seconds()
    duration_s = (end_time-start_time).total_seconds()
    advance_perc = sim_time_s / duration_s

    if verbosity() == 1:
        print(u"{:.1%}".format(advance_perc), file=OUTPUT, end='\r')

    elif verbosity() >=2:
        now = datetime.now()
        elapsed_s = (now - sim_start_time).total_seconds()
        try:
            rate = elapsed_s / sim_time_s
        except ZeroDivisionError:
            rate = 0
        remaining = (end_time - sim_time).total_seconds()
        eta = timedelta(seconds=int(remaining * rate))
        txt = u"Time: {sim} Advance: {perc:.1%} ETA: {eta}"
        disp = txt.format(sim=sim_time.isoformat(" ").split(".")[0],
                          perc=advance_perc,
                          eta=eta)
        print(disp, file=OUTPUT, end='\r')


def message(msg):
    if verbosity() >= 2:
        print(msg + PAD, file=OUTPUT)


def verbose(msg):
    if verbosity() >= 3:
        print(msg + PAD, file=OUTPUT)


def debug(msg):
    if verbosity() >= 4:
        print(msg + PAD, file=OUTPUT)


def warning(msg):
    if verbosity() >= 0:
        print(WARNING + msg + PAD, file=OUTPUT)


def fatal(msg):
    if raise_on_error:
        raise ItziFatal(msg)
    else:
        sys.exit(FATAL + msg + PAD)
