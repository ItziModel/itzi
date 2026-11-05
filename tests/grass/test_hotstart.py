"""Application-owned checkpoint cadence and persistence."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import IO, NoReturn

import numpy as np
import pytest
from itzi_core import (
    HotstartData,
    InfiltrationModelType,
    SimulationConfig,
    SurfaceFlowParameters,
    TemporalType,
)

from itzi.configreader import HotstartRunConfig
from itzi.grass.session import GrassParams, GrassSessionManager
from itzi.hotstart import load_hotstart
from itzi.providers.grass_output import GrassRasterOutputProvider, GrassVectorOutputProvider
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
    checkpoint_time = load_hotstart(path).metadata.simulation_state.sim_time
    assert checkpoint_time == runner.sim.sim_time

    clock[0] = 1019.0
    runner.step()
    assert path.read_bytes() == checkpoint

    clock[0] = 1020.0
    runner.step()
    saved_time = load_hotstart(path).metadata.simulation_state.sim_time
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


@pytest.mark.parametrize("failure_point", ["capture", "archive", "sync", "replace"])
def test_failed_checkpoint_preserves_previous_file_and_can_be_retried(
    checkpoint_runner: tuple[SimulationRunner, list[float], Path],
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    runner, clock, path = checkpoint_runner
    runner._write_hotstart(path)
    previous_checkpoint = path.read_bytes()
    clock[0] = 1010.0

    def fail(*_args: Path | int) -> NoReturn:
        raise OSError("checkpoint failed")

    with monkeypatch.context() as failure:
        if failure_point == "capture":
            failure.setattr(runner.sim, "capture_hotstart", fail)
        elif failure_point == "archive":

            def fail_write(_hotstart: HotstartData, destination: IO[bytes]) -> NoReturn:
                destination.write(b"partial archive")
                raise OSError("checkpoint failed")

            failure.setattr("itzi.simulation_runner.write_hotstart", fail_write)
        elif failure_point == "sync":
            failure.setattr("itzi.simulation_runner.os.fsync", fail)
        else:
            failure.setattr(Path, "replace", fail)
        with pytest.raises(OSError, match="checkpoint failed"):
            runner.step()

    assert path.read_bytes() == previous_checkpoint
    assert not tuple(path.parent.glob(f".{path.name}.*"))
    runner.step()
    assert load_hotstart(path).metadata.simulation_state.sim_time == runner.sim.sim_time


@pytest.mark.parametrize("temporal_type", [TemporalType.RELATIVE, TemporalType.ABSOLUTE])
@pytest.mark.parametrize(
    "case",
    [
        pytest.param((23, 10, 25), id="clipped-final-interval"),
        pytest.param((23, 1, 25), id="changed-cadence"),
        pytest.param((12, 7, 25), id="changed-cadence-before-next-report"),
        pytest.param((23, 10, 40), id="extended-end"),
        pytest.param((3, 30, 25), id="before-first-scheduled-report"),
    ],
)
def test_persisted_resume_preserves_reporting_intervals(
    grass_5by5: None,
    tmp_path: Path,
    temporal_type: TemporalType,
    case: tuple[int, int, int],
) -> None:
    capture_s, record_s, end_s = case
    start = datetime.min if temporal_type == TemporalType.RELATIVE else datetime(2020, 1, 1)
    means = {"mean_rainfall": 10.0, "mean_infiltration": 2.0, "mean_losses": 1.5}
    config = SimulationConfig(
        start_time=start,
        end_time=start + timedelta(seconds=25),
        record_step=timedelta(seconds=10),
        temporal_type=temporal_type,
        dtinf=1.0,
        infiltration_model=InfiltrationModelType.CONSTANT,
        input_map_names={
            "ground_elevation": "z",
            "friction": "n",
            "water_depth": "checkpoint_depth",
            "rainfall_rate": "rainfall",
            "infiltration": "infiltration_rate",
            "losses": "loss_rate",
        },
        output_map_names={key: f"reference_{key}" for key in means},
        surface_flow_parameters=SurfaceFlowParameters(dtmax=1.0),
    )
    path = tmp_path / "checkpoint.zip"
    reference_csv = tmp_path / "reference.csv"
    resumed_csv = tmp_path / "resumed.csv"
    with GrassSessionManager(GrassParams()):
        # Native GRASS libraries need the active session's library search path.
        from itzi.grass.interface import GrassInterface

        interface = GrassInterface(start, config.end_time, np.float32)
        interface.write_raster_map(
            np.full(interface.get_domain_data().shape, 0.1, dtype=np.float32),
            "checkpoint_depth@5by5",
            "water_depth",
            hmin=0.0,
        )
        source = SimulationRunner(config, stats_file=str(reference_csv)).initialize()
        try:
            source.sim.update_until(timedelta(seconds=capture_s))
            source._write_hotstart(path)
            checkpoint = load_hotstart(path)
            previous_report_time = source.sim.report.last_step
            source.run()
            reference_depth = source.sim.get_array("water_depth").copy()
        finally:
            source.finalize()

        resumed_config = config.model_copy(
            update={
                "record_step": timedelta(seconds=record_s),
                "end_time": start + timedelta(seconds=end_s),
                "output_map_names": {},
            }
        )
        immediate = SimulationRunner(resumed_config, hotstart_path=str(path)).initialize()
        try:
            assert immediate.sim.report.last_step == previous_report_time
            assert (
                immediate.sim.time_steps_counters
                == checkpoint.metadata.simulation_state.time_steps_counters
            )
            for key, array in checkpoint.raster_state.padded_arrays.items():
                np.testing.assert_array_equal(immediate.sim.raster_domain.get_padded(key), array)
            immediate._write_hotstart(path)
            assert (
                load_hotstart(path).metadata.simulation_state.last_report_time
                == previous_report_time
            )
        finally:
            immediate.finalize()

        resumed_config = resumed_config.model_copy(
            update={"output_map_names": {key: f"resumed_{key}" for key in means}}
        )
        resumed = SimulationRunner(
            resumed_config, hotstart_path=str(path), stats_file=str(resumed_csv)
        ).initialize()
        try:
            assert resumed.sim.report.last_step == previous_report_time
            resumed.run()
            if end_s == 25:
                np.testing.assert_array_equal(
                    resumed.sim.get_array("water_depth"), reference_depth
                )
            provider = resumed.sim.report.raster_provider
            assert isinstance(provider, GrassRasterOutputProvider)
            for key, expected in means.items():
                for map_id, _ in provider.output_maplist[key]:
                    np.testing.assert_allclose(
                        resumed.g_interface.read_raster_map(map_id), expected, rtol=1e-6
                    )
            area = resumed.sim.domain_data.cells * resumed.sim.domain_data.cell_area
        finally:
            resumed.finalize()

    with resumed_csv.open() as file_obj:
        reports = list(csv.DictReader(file_obj))
    assert reports
    for report in reports:
        if temporal_type == TemporalType.RELATIVE:
            hours, minutes, seconds = report["simulation_time"].split(":")
            report_time = start + timedelta(
                hours=int(hours), minutes=int(minutes), seconds=float(seconds)
            )
        else:
            report_time = datetime.fromisoformat(report["simulation_time"])
        interval = (report_time - previous_report_time).total_seconds()
        assert interval > 0
        assert int(report["timesteps"]) == interval
        assert float(report["average_timestep"]) == 1.0
        for column, rate in (
            ("rainfall_volume", 10.0),
            ("infiltration_volume", -2.0),
            ("losses_volume", -1.5),
        ):
            assert float(report[column]) == pytest.approx(
                rate / 3_600_000 * interval * area, abs=0.00051
            )
        previous_report_time = report_time
    if record_s == 10 and end_s == 25:
        with reference_csv.open() as file_obj:
            assert reports[-1] == list(csv.DictReader(file_obj))[-1]


def test_persisted_drainage_resume(grass_5by5: None, tmp_path: Path, test_data_path: str) -> None:
    inp = tmp_path / "drainage.inp"
    template = (Path(test_data_path) / "tutorial_files" / "tutorial_drainage.inp").read_text()
    inp.write_text(
        template.replace("638752   220262", "15   25")
        .replace("638769   220233", "35   25")
        .replace("J0        112", "J0        -1")
        .replace("J1        111.4", "J1        -1.5")
        .replace("O2         1000", "O2         -2")
    )
    config = SimulationConfig(
        start_time=datetime.min,
        end_time=datetime.min + timedelta(seconds=10),
        record_step=timedelta(seconds=2),
        temporal_type=TemporalType.RELATIVE,
        input_map_names={"ground_elevation": "z", "friction": "n", "rainfall_rate": "rainfall"},
        output_map_names={},
        drainage_output="reference_drainage",
        swmm_inp=inp,
        surface_flow_parameters=SurfaceFlowParameters(dtmax=0.5),
    )
    path = tmp_path / "drainage.zip"
    with GrassSessionManager(GrassParams()):
        source = SimulationRunner(config).initialize()
        try:
            source.sim.update_until(timedelta(seconds=5))
            source._write_hotstart(path)
            checkpoint = load_hotstart(path)
            assert checkpoint.swmm_hotstart_bytes
            swmm_elapsed = checkpoint.metadata.simulation_state.swmm_elapsed_time
            assert swmm_elapsed is not None and swmm_elapsed > 0
            assert source.sim.node_id_to_loc
            source.run()
            reference_depth = source.sim.get_array("water_depth").copy()
        finally:
            source.finalize()

        resumed = SimulationRunner(
            config.model_copy(update={"drainage_output": "resumed_drainage"}),
            hotstart_path=str(path),
        ).initialize()
        try:
            assert resumed.sim.sim_time == checkpoint.metadata.simulation_state.sim_time
            assert resumed.sim.drainage_model is not None
            assert resumed.sim.drainage_model.swmm_sim.start_time == (
                config.start_time + timedelta(seconds=swmm_elapsed)
            )
            np.testing.assert_array_equal(
                resumed.sim.raster_domain.get_padded("water_depth"),
                checkpoint.raster_state.padded_arrays["water_depth"],
            )
            resumed.run()
            np.testing.assert_allclose(
                resumed.sim.get_array("water_depth"), reference_depth, rtol=1e-5, atol=1e-6
            )
            vector_provider = resumed.sim.report.vector_provider
            assert isinstance(vector_provider, GrassVectorOutputProvider)
            assert vector_provider.record_counter > 0
        finally:
            resumed.finalize()
