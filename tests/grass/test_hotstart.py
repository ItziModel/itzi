"""Application-owned checkpoint cadence and persistence."""

from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import NoReturn

import pytest
from itzi_core import SimulationConfig, SurfaceFlowParameters, TemporalType
from itzi_core.hotstart import HotstartLoader

from itzi.configreader import HotstartRunConfig
from itzi.grass.session import GrassParams, GrassSessionManager
from itzi.simulation_runner import SimulationRunner


@pytest.fixture
def checkpoint_runner(
    grass_5by5: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[SimulationRunner, list[float], Path]]:
    clock = [1000.0]
    monkeypatch.setattr("itzi.simulation_runner.monotonic", lambda: clock[0])
    path = tmp_path / "checkpoint.zip"
    config = SimulationConfig(
        start_time=datetime.min,
        end_time=datetime.min + timedelta(minutes=1),
        record_step=timedelta(seconds=30),
        temporal_type=TemporalType.RELATIVE,
        surface_flow_parameters=SurfaceFlowParameters(),
        input_map_names={"ground_elevation": "z", "friction": "n"},
        output_map_names={},
    )
    with GrassSessionManager(GrassParams()):
        runner = SimulationRunner(
            config,
            hotstart_config=HotstartRunConfig(
                wallclock_step=timedelta(seconds=10), save_file_name=path
            ),
        ).initialize()
        try:
            yield runner, clock, path
        finally:
            runner.finalize()


def test_checkpoints_follow_wallclock_cadence_after_completed_steps(
    checkpoint_runner: tuple[SimulationRunner, list[float], Path],
) -> None:
    runner, clock, path = checkpoint_runner
    clock[0] = 1009.0
    runner.step()
    assert not path.exists()

    clock[0] = 1010.0
    runner.step()
    checkpoint = path.read_bytes()
    checkpoint_time = HotstartLoader.from_file(path).get_simulation_state().sim_time
    assert checkpoint_time == runner.sim.sim_time

    clock[0] = 1019.0
    runner.step()
    assert path.read_bytes() == checkpoint

    clock[0] = 1020.0
    runner.step()
    saved_time = HotstartLoader.from_file(path).get_simulation_state().sim_time
    assert saved_time == runner.sim.sim_time
    assert saved_time > checkpoint_time
    assert not tuple(path.parent.glob(f".{path.name}.*"))


@pytest.mark.parametrize("enabled", [False, True])
def test_runs_without_a_due_checkpoint_do_not_write_one(
    checkpoint_runner: tuple[SimulationRunner, list[float], Path], enabled: bool
) -> None:
    runner, clock, path = checkpoint_runner
    if enabled:
        clock[0] = 1001.0
    else:
        runner.hotstart_config = None
        clock[0] = 1100.0
    runner.run()
    assert not path.exists()


def test_failed_step_does_not_write_a_checkpoint(
    checkpoint_runner: tuple[SimulationRunner, list[float], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, clock, path = checkpoint_runner
    clock[0] = 1010.0

    def fail() -> NoReturn:
        raise RuntimeError("step failed")

    monkeypatch.setattr(runner.sim, "update", fail)
    with pytest.raises(RuntimeError, match="step failed"):
        runner.step()
    assert not path.exists()


@pytest.mark.parametrize("failure_point", ["archive", "sync", "replace"])
def test_failed_checkpoint_preserves_previous_file_and_can_be_retried(
    checkpoint_runner: tuple[SimulationRunner, list[float], Path],
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    runner, clock, path = checkpoint_runner
    with runner.sim.create_hotstart() as archive:
        path.write_bytes(archive.getbuffer())
    previous_checkpoint = path.read_bytes()
    clock[0] = 1010.0

    def fail(*_args: Path | int) -> NoReturn:
        raise OSError("checkpoint failed")

    with monkeypatch.context() as failure:
        if failure_point == "archive":
            failure.setattr(runner.sim, "create_hotstart", fail)
        elif failure_point == "sync":
            failure.setattr("itzi.simulation_runner.os.fsync", fail)
        else:
            failure.setattr(Path, "replace", fail)
        with pytest.raises(OSError, match="checkpoint failed"):
            runner.step()

    assert path.read_bytes() == previous_checkpoint
    assert not tuple(path.parent.glob(f".{path.name}.*"))
    runner.step()
    assert HotstartLoader.from_file(path).get_simulation_state().sim_time == runner.sim.sim_time
