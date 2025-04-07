# -*- coding: utf-8 -*-
"""parse command line
"""
import argparse


DESCR = (u"A dynamic, fully distributed hydraulic and hydrologic model.")

arg_parser = argparse.ArgumentParser(description=DESCR)
subparsers = arg_parser.add_subparsers()

# run a simulation
run_parser = subparsers.add_parser("run", help=u"run a simulation")
run_parser.add_argument("config_file", nargs='+',
                        help=(u"an Itzï configuration file "
                              u"(if several given, run in batch mode)"))
run_parser.add_argument("-o", action='store_true', help=u"overwrite files if exist")
run_parser.add_argument("-p", action='store_true', help=u"activate profiler")
verbosity_parser = run_parser.add_mutually_exclusive_group()
verbosity_parser.add_argument("-v", action='count', help=u"increase verbosity")
verbosity_parser.add_argument("-q", action='count', help=u"decrease verbosity")


# display version
version_parser = subparsers.add_parser("version",
                                       help=u"display software version number")
