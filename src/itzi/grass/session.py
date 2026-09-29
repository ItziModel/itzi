"""
Copyright (C) 2025-2026 Laurent Courty

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

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict

import itzi.messenger as msgr

_initialized_temporal_session: tuple[int, str, tuple[str, str, str]] | None = None


class GrassParams(BaseModel):
    """Parameters for GRASS GIS session."""

    model_config = ConfigDict(frozen=True)

    grassdata: str | None = None
    location: str | None = None
    mapset: str | None = None
    region: str | None = None
    mask: str | None = None
    grass_bin: str | None = None


class GrassSessionManager:
    """Manages GRASS session lifecycle."""

    def __init__(self, grass_params: GrassParams):
        self.grass_params = grass_params
        self.grass_session = None
        self._owns_session = False

    @staticmethod
    def current_context() -> tuple[str, str, str] | None:
        """Return the active GRASS context, or ``None`` outside a session."""
        # Importing PyGRASS without a GISRC can terminate through GRASS' C API.
        # The environment guard keeps configuration parsing and external launches
        # independent from GRASS imports.
        if not os.environ.get("GISRC"):
            return None
        if importlib.util.find_spec("grass") is None and "grass" not in sys.modules:
            return None
        try:
            from grass.pygrass.utils import getenv

            values = (getenv("GISDBASE"), getenv("LOCATION_NAME"), getenv("MAPSET"))
        except (ImportError, RuntimeError, SystemExit):
            return None
        if not all(values):
            return None
        return values

    def _validate_requested_context(self) -> None:
        """Ensure an explicit request matches the session selected for this process."""
        active = self.current_context()
        if active is None:
            msgr.fatal("Unable to determine the active GRASS context")
        gisdb, location, mapset = (
            self.grass_params.grassdata,
            self.grass_params.location,
            self.grass_params.mapset,
        )
        if not any((gisdb, location, mapset)):
            return
        if not gisdb or not location or not mapset:
            msgr.fatal("GRASS database, location, and mapset must be supplied together")
        requested_database = str(Path(gisdb).expanduser().resolve())
        active_database = str(Path(active[0]).expanduser().resolve())
        if (requested_database, location, mapset) != (
            active_database,
            active[1],
            active[2],
        ):
            msgr.fatal(
                "Requested GRASS context does not match the active session "
                f"({active_database}/{active[1]}/{active[2]})"
            )

    @staticmethod
    def ensure_temporal_initialized() -> None:
        """Initialize the temporal framework once for the active GRASS session."""
        global _initialized_temporal_session

        context = GrassSessionManager.current_context()
        if context is None:
            msgr.fatal("No active GRASS session for temporal initialization")
        import grass.script as gscript
        import grass.temporal as tgis

        gscript.set_raise_on_error(True)
        session = (os.getpid(), os.environ["GISRC"], context)
        if _initialized_temporal_session != session:
            tgis.init(raise_fatal_error=True)
            _initialized_temporal_session = session
        tgis.set_raise_on_error(True)

    def open(self) -> None:
        """Open a GRASS session if needed."""
        if self.current_context() is not None:
            self._validate_requested_context()
            self.ensure_temporal_initialized()
            return

        gisdb = self.grass_params.grassdata
        location = self.grass_params.location
        mapset = self.grass_params.mapset
        # Check if mandatory GRASS parameters are present
        if not gisdb or not location or not mapset:
            msgr.fatal("No GRASS parameters to create a session.")

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
        self._owns_session = True
        if not os.environ.get("GISRC"):
            session_env = getattr(self.grass_session, "env", None)
            if session_env and session_env.get("GISRC"):
                os.environ.update(session_env)
        try:
            self._validate_requested_context()
            self.ensure_temporal_initialized()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Stop GRASS session."""
        global _initialized_temporal_session
        if self.grass_session is not None and self._owns_session:
            try:
                self.grass_session.finish()
            except Exception as e:
                print(f"Warning: Error cleaning up GRASS session: {e}")
            _initialized_temporal_session = None
        self.grass_session = None
        self._owns_session = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()
