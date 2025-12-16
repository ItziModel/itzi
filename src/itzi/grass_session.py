"""
Copyright (C) 2025 Laurent Courty

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
from typing import TYPE_CHECKING
import sys
import os
import subprocess
import importlib

import itzi.messenger as msgr

if TYPE_CHECKING:
    from itzi.data_containers import GrassParams


class GrassSessionManager:
    """Manages GRASS session lifecycle."""

    def __init__(self, grass_params: GrassParams):
        self.grass_params = grass_params
        self.session = None
        if importlib.util.find_spec("grass"):
            self._is_active = True
        else:
            self._is_active = False

    def open(self):
        """Open a GRASS session if needed."""
        if self._is_active:
            return  # Already started

        # Check if mandatory GRASS parameters are present
        if not all(
            [self.grass_params.grassdata, self.grass_params.location, self.grass_params.mapset]
        ):
            msgr.fatal("No GRASS parameters to create a session.")

        gisdb = self.grass_params.grassdata
        location = self.grass_params.location
        mapset = self.grass_params.mapset

        # Check if the given parameters exist and can be accessed
        error_msg = "'{}' does not exist or does not have adequate permissions"
        if not os.access(gisdb, os.R_OK):
            msgr.fatal(error_msg.format(gisdb))
        elif not os.access(os.path.join(gisdb, location), os.R_OK):
            msgr.fatal(error_msg.format(location))
        elif not os.access(os.path.join(gisdb, location, mapset), os.W_OK):
            msgr.fatal(error_msg.format(mapset))

        # Set GRASS python path
        if self.grass_params.grass_bin:
            grassbin = self.grass_params.grass_bin
        else:
            grassbin = "grass"
        grass_cmd = [grassbin, "--config", "python_path"]
        grass_python_path = subprocess.check_output(grass_cmd, text=True).strip()
        sys.path.append(grass_python_path)
        # Now we can import grass modules
        import grass.script as gscript

        # set up session
        self.grass_session = gscript.setup.init(
            path=gisdb, location=location, mapset=mapset, grass_path=grassbin
        )

        self._is_active = True

    def close(self):
        """Stop GRASS session."""
        if self.session is not None and self._is_active:
            try:
                self.session.finish()
            except Exception as e:
                print(f"Warning: Error cleaning up GRASS session: {e}")
        self._is_active = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()
