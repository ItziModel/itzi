"""
Copyright (C) 2015-2025 Laurent Courty

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

from typing import TYPE_CHECKING, Callable, NoReturn
import sys
import logging
import os
from datetime import timedelta, datetime

if TYPE_CHECKING:
    from itzi_core.data_containers import SimulationConfig

raise_on_error = True


class VerbosityLevel:
    """Messenger verbosity levels"""

    SUPER_QUIET = 0
    QUIET = 1
    MESSAGE = 2
    VERBOSE = 3
    DEBUG = 4


def verbosity():
    """Return the current verbosity as integer"""
    try:
        return int(os.environ.get("ITZI_VERBOSE"))
    except TypeError:
        return VerbosityLevel.QUIET


class ItziLogger:
    """Custom logger wrapper maintaining backward compatibility"""

    VERBOSE_LEVEL = 15
    logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")

    def __init__(self):
        self.logger = logging.getLogger("itzi")
        self.raise_on_error: bool = True
        self._setup_handlers()

    def _setup_handlers(self):
        """Configure console and optional file handlers"""
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        if any(
            getattr(handler, "_itzi_console_handler", False) for handler in self.logger.handlers
        ):
            return

        # Console handler (stderr)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler._itzi_console_handler = True
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(console_handler)

    def add_file_handler(self, filepath, level=logging.DEBUG):
        """Add file logging capability"""
        file_handler = logging.FileHandler(filepath)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(file_handler)

    def set_verbosity(self, verbosity_level: int):
        """Map verbosity to logging level"""
        mapping = {
            VerbosityLevel.SUPER_QUIET: logging.ERROR,
            VerbosityLevel.QUIET: logging.WARNING,
            VerbosityLevel.MESSAGE: logging.INFO,
            VerbosityLevel.VERBOSE: self.VERBOSE_LEVEL,
            VerbosityLevel.DEBUG: logging.DEBUG,
        }
        level = mapping.get(verbosity_level, logging.INFO)
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setLevel(level)

    def fatal(self, msg: str) -> NoReturn:
        """Log fatal error and raise or exit"""
        self.logger.error(f"ERROR: {msg}")
        if raise_on_error:
            raise RuntimeError(msg)
        else:
            sys.exit(f"ERROR: {msg}")

    def warning(self, msg: str) -> None:
        self.logger.warning(f"WARNING: {msg}")

    def message(self, msg: str) -> None:
        self.logger.info(msg)

    def verbose(self, msg: str) -> None:
        self.logger.log(self.VERBOSE_LEVEL, msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)


# Global instance
_itzi_logger = ItziLogger()

# Backward-compatible module-level interface
raise_on_error: bool = _itzi_logger.raise_on_error
fatal: Callable[[str], NoReturn] = _itzi_logger.fatal
warning: Callable[[str], None] = _itzi_logger.warning
message: Callable[[str], None] = _itzi_logger.message
verbose: Callable[[str], None] = _itzi_logger.verbose
debug: Callable[[str], None] = _itzi_logger.debug


def percent(start_time, end_time, sim_time, sim_start_time):
    """Display progress of the simulation"""
    sim_time_s = (sim_time - start_time).total_seconds()
    duration_s = (end_time - start_time).total_seconds()
    advance_perc = sim_time_s / duration_s

    if verbosity() == VerbosityLevel.QUIET:
        print(f"{advance_perc:.1%}", file=sys.stderr, end="\r")

    elif verbosity() >= VerbosityLevel.MESSAGE:
        elapsed_s = (datetime.now() - sim_start_time).total_seconds()
        try:
            rate = elapsed_s / sim_time_s
        except ZeroDivisionError:
            rate = 0
        remaining = (end_time - sim_time).total_seconds()
        eta = timedelta(seconds=int(remaining * rate))
        txt = "Time: {sim} Advance: {perc:.1%} ETA: {eta}{pad}"
        disp = txt.format(
            sim=sim_time.isoformat(" ").split(".")[0],
            perc=advance_perc,
            eta=eta,
            pad=" " * 10,
        )
        print(disp, file=sys.stderr, end="\r")


def display_sim_param(sim_config: SimulationConfig) -> None:
    """Display simulation parameters if verbose."""
    inter_txt = "#" * 50
    txt_template = "{:<24s} {}"
    verbose(inter_txt)
    verbose("Input maps:")
    for key, value in sim_config.input_map_names.items():
        verbose(txt_template.format(key, value))
    verbose(inter_txt)
    verbose("Output maps:")
    for key, value in sim_config.output_map_names.items():
        verbose(txt_template.format(key, value))
    verbose(inter_txt)
    verbose("Simulation parameters:")
    sim_params = {
        **sim_config.surface_flow_parameters.model_dump(),
        "dtinf": sim_config.dtinf,
        "inf_model": sim_config.infiltration_model,
    }
    for key, value in sim_params.items():
        verbose(txt_template.format(key, value))
    verbose(inter_txt)
    verbose("Simulation times:")
    txt_start_time = sim_config.start_time.isoformat(" ").split(".")[0]
    txt_end_time = sim_config.end_time.isoformat(" ").split(".")[0]
    verbose(txt_template.format("start", txt_start_time))
    verbose(txt_template.format("end", txt_end_time))
    verbose(txt_template.format("duration", sim_config.end_time - sim_config.start_time))
    verbose(txt_template.format("record_step", sim_config.record_step))
    verbose(inter_txt)
