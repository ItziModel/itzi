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

import os
import subprocess
import sys
from contextlib import ExitStack
from ctypes import byref
from dataclasses import dataclass
from multiprocessing import get_start_method, set_start_method
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Self

import itzi.messenger as msgr

if TYPE_CHECKING:
    from grass.lib.gis import Cell_head
    from grass.script.setup import SessionHandle

    from itzi.ensemble.models import EffectiveMask

MIN_GRASS_VERSION = (8, 5)


@dataclass(frozen=True, slots=True)
class _GrassContext:
    grassdata: str
    location: str
    mapset: str


@dataclass(frozen=True, slots=True)
class _SessionIdentity:
    pid: int
    gisrc: str
    context: _GrassContext


_initialized_temporal_session: _SessionIdentity | None = None
_synchronized_session: _SessionIdentity | None = None


@dataclass(frozen=True, slots=True)
class GrassParams:
    """Parameters for GRASS GIS session."""

    grassdata: str | None = None
    location: str | None = None
    mapset: str | None = None
    region: str | None = None
    mask: str | None = None
    grass_bin: str | None = None

    def __post_init__(self) -> None:
        context = (self.grassdata, self.location, self.mapset)
        if any(value is not None for value in context) and not all(
            value is not None for value in context
        ):
            raise ValueError("GRASS database, location, and mapset must be supplied together")


def ensure_min_version() -> None:
    """Check compatibility before importing session-dependent GRASS libraries."""
    import grass.script as gscript

    full = gscript.parse_command("g.version", flags="g")["version"]
    major, minor = (int(part) for part in full.split(".")[:2])
    if (major, minor) < MIN_GRASS_VERSION:
        required = f"{MIN_GRASS_VERSION[0]}.{MIN_GRASS_VERSION[1]}"
        raise RuntimeError(
            f"itzi requires at least GRASS {required}, found version {major}.{minor}"
        )


def _restore_environment(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _validate_target(params: GrassParams) -> None:
    assert params.grassdata and params.location and params.mapset
    database = Path(params.grassdata).expanduser().resolve()
    for path, permission in (
        (database, os.R_OK),
        (database / params.location, os.R_OK),
        (database / params.location / params.mapset, os.W_OK),
    ):
        if not path.is_dir() or not os.access(path, permission):
            msgr.fatal(f"'{path}' does not exist or does not have adequate permissions")


def _finish_session(session: SessionHandle) -> None:
    global _initialized_temporal_session, _synchronized_session

    try:
        if _initialized_temporal_session is not None:
            _stop_temporal()
    finally:
        _initialized_temporal_session = None
        _synchronized_session = None
        session.finish()


def _stop_temporal() -> None:
    global _initialized_temporal_session
    import grass.temporal as tgis

    try:
        tgis.stop_subprocesses()
    finally:
        # Stopped RPC objects retain closed pipes and can race their checker threads.
        tgis.core.c_library_interface = None
        tgis.core.message_interface = None
        _initialized_temporal_session = None


def _acquire_session(params: GrassParams, contexts: ExitStack) -> SessionHandle | None:
    """Acquire a bare runtime, also used by multi-member validation workers."""
    try:
        import grass.script as gscript
    except ImportError:
        python_path = subprocess.check_output(
            [params.grass_bin or "grass", "--config", "python_path"], text=True
        ).strip()
        sys.path.append(python_path)
        import grass.script as gscript

    if GrassSessionManager.current_context() is not None:
        return None
    if not params.grassdata:
        msgr.fatal("No GRASS parameters to create a session.")
    _validate_target(params)
    previous = {name: os.environ.get(name) for name in ("GISRC", "GIS_LOCK")}
    contexts.callback(_restore_environment, previous)
    session = gscript.setup.init(
        path=str(Path(params.grassdata).expanduser().resolve()),
        location=params.location,
        mapset=params.mapset,
        grass_path=params.grass_bin or "grass",
    )
    contexts.callback(_finish_session, session)
    return session


def _refresh_session() -> None:
    """Refresh C-library session paths and caches without rewriting the GISRC."""
    global _synchronized_session
    from grass.lib import gis as libgis, raster as libraster

    libgis.G__read_gisrc_path()
    names = []
    index = 0
    while name := libgis.G_get_env_name(index):
        names.append(name)
        index += 1
    for source in (libgis.G_VAR_MAPSET, libgis.G_VAR_GISRC):
        for name in names:
            libgis.G_setenv_nogisrc2(name, None, source)
    libgis.G__read_gisrc_env()
    libgis.G__read_mapset_env()
    libgis.G_reset_mapsets()
    libgis.G_unset_window()
    libraster.Rast_unset_window()
    context = GrassSessionManager.current_context()
    assert context is not None
    _synchronized_session = _SessionIdentity(
        pid=os.getpid(), gisrc=os.environ["GISRC"], context=_GrassContext(*context)
    )


def _restore_windows(gis: Cell_head, window: Cell_head) -> None:
    from grass.lib import gis as libgis, raster as libraster

    # The normal window API closes the retained automatic-mask descriptor first.
    libgis.G_set_window(byref(window))
    try:
        libraster.Rast_set_window(byref(window))
    finally:
        libgis.G_set_window(byref(gis))


class GrassSessionManager:
    """Own one fixed work region/mask and restore its caller's GRASS context.

    All Itzi raster handles must be closed before closing the manager. An attached
    session remains available; only sessions acquired by this manager are finished.
    """

    def __init__(self, grass_params: GrassParams) -> None:
        self.grass_params = grass_params
        self.grass_session: SessionHandle | None = None
        self.effective_mask: EffectiveMask
        self._contexts: ExitStack | None = None

    @staticmethod
    def current_context() -> tuple[str, str, str] | None:
        """Return the active GRASS context, or ``None`` outside a session."""
        if not os.environ.get("GISRC"):
            return None
        try:
            import grass.script as gscript

            environment = gscript.gisenv()
            values = tuple(
                environment.get(name) for name in ("GISDBASE", "LOCATION_NAME", "MAPSET")
            )
        except (ImportError, RuntimeError, SystemExit, OSError):
            return None
        if not all(values):
            return None
        return values

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
        session = _SessionIdentity(
            pid=os.getpid(), gisrc=os.environ["GISRC"], context=_GrassContext(*context)
        )
        if _initialized_temporal_session != session:
            _stop_temporal()
            # Python's forkserver retains the first session's environment for later helpers.
            method = get_start_method()
            if method == "forkserver":
                set_start_method("spawn", force=True)
            try:
                tgis.init(raise_fatal_error=True)
            except BaseException:
                _stop_temporal()
                raise
            finally:
                if method == "forkserver":
                    set_start_method(method, force=True)
            _initialized_temporal_session = session
        tgis.set_raise_on_error(True)

    def open(self) -> None:
        """Acquire a session, select the job context, and synchronize its windows."""
        if self._contexts is not None:
            return
        self._contexts = contexts = ExitStack()
        try:
            self.grass_session = _acquire_session(self.grass_params, contexts)
            ensure_min_version()
            self._select_context(contexts)
            self.ensure_temporal_initialized()
        except BaseException:
            self.close()
            raise

    def _select_context(self, contexts: ExitStack) -> None:
        import grass.script as gscript
        from grass.lib import gis as libgis, raster as libraster

        from itzi.grass.utils import read_domain, resolve_effective_mask

        active = self.current_context()
        if active is None:
            msgr.fatal("Unable to determine the active GRASS context")
        identity = _SessionIdentity(
            pid=os.getpid(), gisrc=os.environ["GISRC"], context=_GrassContext(*active)
        )
        if _synchronized_session != identity:
            _refresh_session()
        gis = libgis.Cell_head()
        window = libgis.Cell_head()
        libgis.G_get_set_window(byref(gis))
        libraster.Rast_get_window(byref(window))
        params = self.grass_params
        requested = (
            (str(Path(params.grassdata).expanduser().resolve()), params.location, params.mapset)
            if params.grassdata
            else active
        )
        changed = requested != (str(Path(active[0]).resolve()), active[1], active[2])
        previous = {
            name: os.environ.get(name)
            for name in ("GISRC", "GRASS_REGION", "WIND_OVERRIDE", "GRASS_MASK")
        }
        if changed:
            _validate_target(params)
            target = gscript.setup.init(
                path=requested[0],
                location=params.location,
                mapset=params.mapset,
                grass_path=params.grass_bin or "grass",
                env=os.environ.copy(),
            )
            contexts.callback(target.finish)
            contexts.callback(self.ensure_temporal_initialized)
        contexts.callback(_restore_windows, gis, window)
        if changed:
            contexts.callback(_refresh_session)
        contexts.callback(_restore_environment, previous)
        if changed:
            os.environ["GISRC"] = target.env["GISRC"]
            for name in ("GRASS_REGION", "WIND_OVERRIDE", "GRASS_MASK"):
                os.environ.pop(name, None)
            _refresh_session()
        self.effective_mask = resolve_effective_mask(params.mask)
        # Explicit regions must not consult caller-only saved-region overrides.
        if params.region is not None:
            os.environ.pop("GRASS_REGION", None)
            os.environ.pop("WIND_OVERRIDE", None)
            contexts.enter_context(gscript.RegionManagerEnv(region=params.region))
        else:
            # RegionManagerEnv() without arguments reads WIND, ignoring inherited overrides.
            snapshot = gscript.region()
            contexts.enter_context(
                gscript.RegionManagerEnv(
                    **{key: snapshot[key] for key in ("n", "s", "e", "w", "rows", "cols")}
                )
            )
        contexts.enter_context(
            gscript.MaskManager(mask_name=self.effective_mask.source, remove=False)
        )
        libgis.G_unset_window()
        selected = libgis.Cell_head()
        libgis.G_get_window(byref(selected))
        libgis.G_set_window(byref(selected))
        libraster.Rast_set_window(byref(selected))
        read_domain()

    def close(self) -> None:
        """Restore process state and release owned resources, once."""
        contexts, self._contexts = self._contexts, None
        try:
            if contexts is not None:
                contexts.close()
        finally:
            self.grass_session = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
