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

OUTPUT = sys.stderr
FATAL = "ERROR: "
WARNING = "WARNING: "


def verbosity():
    return int(os.environ['GRASS_VERBOSE'])


def message(msg):
    if verbosity() >= 2:
        print(msg, file=OUTPUT)


def verbose(msg):
    if verbosity() >= 3:
        print(msg, file=OUTPUT)


def warning(msg):
    print(WARNING + msg, file=OUTPUT)


def fatal(msg):
    sys.exit(FATAL + msg)
