import sys
from types import ModuleType, SimpleNamespace

from itzi.grass_session import GrassSessionManager


class FakeGrassSession:
    def __init__(self) -> None:
        self.finished = False

    def finish(self) -> None:
        self.finished = True


def test_close_finishes_session_created_by_manager(monkeypatch) -> None:
    session = FakeGrassSession()
    grass_script = ModuleType("grass.script")
    grass_script.setup = SimpleNamespace(init=lambda **_: session)
    grass_package = ModuleType("grass")
    grass_package.script = grass_script

    monkeypatch.setattr("itzi.grass_session.importlib.util.find_spec", lambda _: None)
    monkeypatch.setattr("itzi.grass_session.os.access", lambda *_: True)
    monkeypatch.setattr(
        "itzi.grass_session.subprocess.check_output", lambda *_args, **_kwargs: "/tmp"
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

    assert session.finished
