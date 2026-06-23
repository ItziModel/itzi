"""parse command line"""

from __future__ import annotations

import argparse


DESCR = "A dynamic, fully distributed hydraulic and hydrologic model."


def parse_resume_from(arg_value: str) -> tuple[str | None, str]:
    if "=" not in arg_value:
        return None, arg_value

    key, path = arg_value.split("=", 1)
    if not key or not path:
        raise argparse.ArgumentTypeError("--resume-from must be PATH or CONFIG=PATH")
    return key, path


def build_parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(description=DESCR)
    subparsers = arg_parser.add_subparsers(dest="command", required=True)

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
    run_parser.add_argument(
        "--resume-from",
        action="append",
        type=parse_resume_from,
        metavar="PATH|CONFIG=PATH",
        default=[],
        help="resume a simulation from a hotstart file",
    )

    # display version
    subparsers.add_parser("version", help="display software version number")

    return arg_parser
