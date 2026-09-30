import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from itzi.grass.session import GrassParams, GrassSessionManager


@pytest.mark.parametrize(
    "values",
    (
        {"grassdata": "/grassdata"},
        {"grassdata": "/grassdata", "location": "project"},
        {"mapset": "PERMANENT"},
    ),
)
def test_grass_params_requires_complete_context(values: dict[str, str]) -> None:
    with pytest.raises(
        ValueError, match="GRASS database, location, and mapset must be supplied together"
    ):
        GrassParams(**values)

    assert GrassParams(region="region").grassdata is None
    assert GrassParams("/grassdata", "project", "PERMANENT").mapset == "PERMANENT"


class FakeGrassSession:
    def __init__(self, env: dict[str, str]) -> None:
        self.env = env
        self.finished = False

    def finish(self) -> None:
        self.finished = True


def test_manager_activates_and_finishes_created_session(monkeypatch) -> None:
    session = FakeGrassSession({"GISRC": "/tmp/worker-gisrc"})
    grass_script = ModuleType("grass.script")
    grass_script.__dict__["setup"] = SimpleNamespace(init=lambda **_: session)
    grass_package = ModuleType("grass")
    grass_package.__dict__["script"] = grass_script

    monkeypatch.setattr("itzi.grass.session.os.access", lambda *_: True)
    monkeypatch.setattr(GrassSessionManager, "ensure_temporal_initialized", lambda self: None)
    monkeypatch.setattr(
        "itzi.grass.session.subprocess.check_output", lambda *_args, **_kwargs: "/tmp"
    )
    monkeypatch.delenv("GISRC", raising=False)
    monkeypatch.setattr(
        GrassSessionManager,
        "current_context",
        staticmethod(
            lambda: (
                ("/grassdata", "location", "mapset")
                if os.environ.get("GISRC") == session.env["GISRC"]
                else None
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "grass", grass_package)
    monkeypatch.setitem(sys.modules, "grass.script", grass_script)

    grass_params = GrassParams(
        grassdata="/grassdata",
        location="location",
        mapset="mapset",
        grass_bin="grass",
    )
    manager = GrassSessionManager(grass_params)

    manager.open()
    manager.close()

    assert os.environ["GISRC"] == "/tmp/worker-gisrc"
    assert session.finished


def test_temporal_initialization_once_per_active_session(monkeypatch) -> None:
    calls: list[str] = []
    script = ModuleType("grass.script")
    script.__dict__["set_raise_on_error"] = lambda _: calls.append("script")
    temporal = ModuleType("grass.temporal")
    temporal.__dict__["init"] = lambda **_: calls.append("init")
    temporal.__dict__["set_raise_on_error"] = lambda _: calls.append("temporal")
    package = ModuleType("grass")
    package.__dict__["script"] = script
    package.__dict__["temporal"] = temporal
    monkeypatch.setitem(sys.modules, "grass", package)
    monkeypatch.setitem(sys.modules, "grass.script", script)
    monkeypatch.setitem(sys.modules, "grass.temporal", temporal)
    monkeypatch.setattr("itzi.grass.session._initialized_temporal_session", None)
    monkeypatch.setattr(
        GrassSessionManager,
        "current_context",
        staticmethod(lambda: ("/grassdata", "location", "mapset")),
    )
    monkeypatch.setenv("GISRC", "/tmp/first-gisrc")

    GrassSessionManager(GrassParams()).open()
    GrassSessionManager.ensure_temporal_initialized()
    assert calls.count("init") == 1

    monkeypatch.setenv("GISRC", "/tmp/second-gisrc")
    GrassSessionManager(GrassParams()).open()
    assert calls.count("init") == 2
