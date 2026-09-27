import os
import sys
from types import ModuleType, SimpleNamespace

from itzi.grass_session import GrassSessionManager


class FakeGrassSession:
    def __init__(self, env: dict[str, str]) -> None:
        self.env = env
        self.finished = False

    def finish(self) -> None:
        self.finished = True


def test_manager_activates_and_finishes_created_session(monkeypatch) -> None:
    session = FakeGrassSession({"GISRC": "/tmp/worker-gisrc"})
    grass_script = ModuleType("grass.script")
    grass_script.setup = SimpleNamespace(init=lambda **_: session)
    grass_package = ModuleType("grass")
    grass_package.script = grass_script

    monkeypatch.setattr("itzi.grass_session.importlib.util.find_spec", lambda _: None)
    monkeypatch.setattr("itzi.grass_session.os.access", lambda *_: True)
    monkeypatch.setattr(
        "itzi.grass_session.subprocess.check_output", lambda *_args, **_kwargs: "/tmp"
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

    grass_params = SimpleNamespace(
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
