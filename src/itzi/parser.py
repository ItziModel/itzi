# -*- coding: utf-8 -*-
"""parse command line"""

import argparse


DESCR = "A dynamic, fully distributed hydraulic and hydrologic model."

arg_parser = argparse.ArgumentParser(description=DESCR)
subparsers = arg_parser.add_subparsers()

# run a simulation
run_parser = subparsers.add_parser("run", help="run a simulation")
run_parser.add_argument(
    "config_file",
    nargs="+",
    help=("an Itzï configuration file (if several given, run in batch mode)"),
)
run_parser.add_argument("-o", action="store_true", help="overwrite files if exist")
verbosity_parser = run_parser.add_mutually_exclusive_group()
verbosity_parser.add_argument("-v", action="count", help="increase verbosity")
verbosity_parser.add_argument("-q", action="count", help="decrease verbosity")


# display version
version_parser = subparsers.add_parser("version", help="display software version number")
