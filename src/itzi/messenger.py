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

import sys
import logging
import os
from datetime import timedelta, datetime

from itzi.itzi_error import ItziFatal
from itzi.const import VerbosityLevel

raise_on_error = True


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
        self.raise_on_error = True
        self._setup_handlers()

    def _setup_handlers(self):
        """Configure console and optional file handlers"""
        # Console handler (stderr)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG)

    def add_file_handler(self, filepath, level=logging.DEBUG):
        """Add file logging capability"""
        file_handler = logging.FileHandler(filepath)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(file_handler)

    def set_verbosity(self, verbosity_level):
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

    def fatal(self, msg):
        """Log fatal error and raise or exit"""
        self.logger.error(f"ERROR: {msg}")
        if raise_on_error:
            raise ItziFatal(msg)
        else:
            sys.exit(f"ERROR: {msg}")

    def warning(self, msg):
        self.logger.warning(f"WARNING: {msg}")

    def message(self, msg):
        self.logger.info(msg)

    def verbose(self, msg):
        self.logger.log(self.VERBOSE_LEVEL, msg)

    def debug(self, msg):
        self.logger.debug(msg)


# Global instance
_itzi_logger = ItziLogger()

# Backward-compatible module-level interface
raise_on_error = _itzi_logger.raise_on_error
fatal = _itzi_logger.fatal
warning = _itzi_logger.warning
message = _itzi_logger.message
verbose = _itzi_logger.verbose
debug = _itzi_logger.debug


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
