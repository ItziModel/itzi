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


@pytest.mark.parametrize("setup_fails", [False, True])
def test_manager_activates_and_finishes_created_session(
    monkeypatch, tmp_path, setup_fails
) -> None:
    session = FakeGrassSession({"GISRC": "/tmp/worker-gisrc"})
    grass_script = ModuleType("grass.script")

    def init(**kwargs):
        os.environ.update(session.env)
        return session

    grass_script.__dict__["setup"] = SimpleNamespace(init=init)
    grass_package = ModuleType("grass")
    grass_package.__dict__["script"] = grass_script

    monkeypatch.setattr("itzi.grass.session.os.access", lambda *_: True)
    monkeypatch.setattr("itzi.grass.session._validate_target", lambda _: None)
    monkeypatch.setattr("itzi.grass.session.ensure_min_version", lambda: None)
    monkeypatch.setattr(GrassSessionManager, "ensure_temporal_initialized", lambda self: None)
    monkeypatch.setattr("itzi.grass.session._initialized_temporal_session", None)
    restored = []

    def select_context(self, stack):
        assert os.environ["GISRC"] == session.env["GISRC"]
        stack.callback(restored.append, "restored")
        if setup_fails:
            raise ValueError("setup failed")

    monkeypatch.setattr(GrassSessionManager, "_select_context", select_context)
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

    if setup_fails:
        with pytest.raises(ValueError, match="setup failed"):
            manager.open()
        assert session.finished
    else:
        manager.open()
        manager.open()
    manager.close()
    manager.close()

    assert "GISRC" not in os.environ
    assert session.finished
    assert restored == ["restored"]


def test_temporal_initialization_once_per_active_session(monkeypatch) -> None:
    calls: list[str] = []
    script = ModuleType("grass.script")
    script.__dict__["set_raise_on_error"] = lambda _: calls.append("script")
    temporal = ModuleType("grass.temporal")
    temporal.__dict__["init"] = lambda **_: calls.append("init")
    temporal.__dict__["set_raise_on_error"] = lambda _: calls.append("temporal")
    temporal.__dict__["stop_subprocesses"] = lambda: calls.append("stop")
    temporal.__dict__["core"] = SimpleNamespace()
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

    GrassSessionManager.ensure_temporal_initialized()
    GrassSessionManager.ensure_temporal_initialized()
    assert calls.count("init") == 1

    monkeypatch.setenv("GISRC", "/tmp/second-gisrc")
    GrassSessionManager.ensure_temporal_initialized()
    assert calls.count("init") == 2
    assert calls.count("stop") == 2


def test_attached_cleanup_runs_all_callbacks_after_restoration_error(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("itzi.grass.session._acquire_session", lambda *_: None)
    monkeypatch.setattr("itzi.grass.session.ensure_min_version", lambda: None)
    monkeypatch.setattr(GrassSessionManager, "ensure_temporal_initialized", lambda self: None)

    def fail():
        calls.append("failure")
        raise RuntimeError("restore failed")

    def select_context(self, stack):
        stack.callback(calls.append, "last")
        stack.callback(fail)
        stack.callback(calls.append, "first")

    monkeypatch.setattr(GrassSessionManager, "_select_context", select_context)
    manager = GrassSessionManager(GrassParams())
    manager.open()
    with pytest.raises(RuntimeError, match="restore failed"):
        manager.close()
    manager.close()
    assert calls == ["first", "failure", "last"]


def test_unsupported_version_unwinds_owned_session_before_context_setup(monkeypatch) -> None:
    calls = []

    def acquire(params, stack):
        stack.callback(calls.append, "finished")

    def version():
        raise RuntimeError("requires at least GRASS 8.5")

    monkeypatch.setattr("itzi.grass.session._acquire_session", acquire)
    monkeypatch.setattr("itzi.grass.session.ensure_min_version", version)
    monkeypatch.setattr(
        GrassSessionManager, "_select_context", lambda *_: pytest.fail("context setup")
    )
    manager = GrassSessionManager(GrassParams(region="missing"))
    with pytest.raises(RuntimeError, match="requires at least GRASS 8.5"):
        manager.open()
    manager.close()
    assert calls == ["finished"]
