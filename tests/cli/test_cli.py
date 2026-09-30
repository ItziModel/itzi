"""Test the CLI"""

import argparse
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import itzi.messenger as msgr
from itzi.cli_parser import build_parser
from itzi.ensemble.models import ResolvedEnsemble, ResolvedSimulation
from itzi.itzi import (
    VerbosityLevel,
    _preflight_ensemble,
    _run_ensemble_batch,
    _run_one_ensemble,
    _select_members,
    itzi_run,
    itzi_run_one,
    main,
    preflight_ensemble_worker,
    reconcile_hotstart_commands,
    sim_runner_worker,
)


def test_run_parser_accepts_multiple_config_files():
    args = build_parser().parse_args(["run", "a.ini", "b.ini", "-o", "-vv"])
    assert args.config_file == ["a.ini", "b.ini"]
    assert args.o is True
    assert args.v == 2
    assert args.q is None


def test_run_parser_accepts_resume_from_args():
    args = build_parser().parse_args(
        [
            "run",
            "a.ini",
            "b.ini",
            "--resume-from",
            "a.ini=restart_a.zip",
            "--resume-from",
            "b.ini=restart_b.zip",
        ]
    )
    assert args.resume_from == [("a.ini", "restart_a.zip"), ("b.ini", "restart_b.zip")]


@pytest.mark.parametrize("dry_option", ["--dry-run", "-d"])
def test_run_parser_accepts_yaml_dry_run_and_member_selection(dry_option):
    args = build_parser().parse_args(
        ["run", "study.yaml", dry_option, "--member", "study#sim-a", "--member", "study#sim-b"]
    )

    assert args.dry is True
    assert args.member == ["study#sim-a", "study#sim-b"]


def test_ensemble_members_are_resolved_in_one_spawn(monkeypatch):
    simulations = (object(), object())
    ensemble = SimpleNamespace(
        ensemble_id="study",
        simulations=simulations,
        source=SimpleNamespace(path=Path("study.yaml")),
        manifest_template=None,
    )
    calls = []

    monkeypatch.setattr("itzi.itzi.load_batch", lambda _: ((ensemble,), ()))
    monkeypatch.setattr(
        "itzi.itzi._resolve_ensemble_in_subprocess",
        lambda received: calls.append(received) or (),
    )
    monkeypatch.setattr("itzi.itzi._display_dry_plan", lambda *_, **__: None)

    _run_ensemble_batch(
        SimpleNamespace(config_file=["study.yaml"], resume_from=[], member=[], dry=True)
    )

    assert calls == [simulations]


@pytest.mark.parametrize("dry", [False, True])
def test_ensembles_are_checked_and_run_in_order_without_cross_ensemble_validation(
    tmp_path, monkeypatch, dry
):
    ensembles = tuple(
        SimpleNamespace(
            ensemble_id=name,
            simulations=(SimpleNamespace(simulation_id=name),),
            source=SimpleNamespace(path=tmp_path / f"{name}.yaml"),
            manifest_template=None,
        )
        for name in ("first", "second")
    )
    members = {
        name: ResolvedSimulation(
            simulation_id=name,
            coordinates=(),
            grass_params=None,
            domain_data=None,
            effective_mask=None,
            input_kinds=(),
            simulation_config=SimpleNamespace(
                input_map_names={"friction": "first_water_depth@PERMANENT"}
                if name == "second"
                else {}
            ),
            artifacts=SimpleNamespace(
                output_map_names=(("water_depth", "first_water_depth"),)
                if name == "first"
                else (),
            ),
            normalized_payload=name,
        )
        for name in ("first", "second")
    }
    calls = []
    monkeypatch.setattr("itzi.itzi.load_batch", lambda _: (ensembles, ()))

    def resolve(expanded):
        name = expanded[0].simulation_id
        calls.append(("resolve", name))
        return (members[name],)

    def validate(simulations, _sources):
        calls.append(("validate", simulations[0].simulation_id))
        assert len(simulations) == 1

    def preflight(simulations):
        calls.append(("preflight", simulations[0].simulation_id))
        return {}

    def run(resolved, selected_ids, *_args, **_kwargs):
        calls.append(("run", resolved.ensemble.ensemble_id))
        assert selected_ids == {resolved.ensemble.ensemble_id}
        return 0

    def display(resolved, *_args, **_kwargs):
        calls.append(("display", resolved.ensemble.ensemble_id))

    monkeypatch.setattr("itzi.itzi._resolve_ensemble_in_subprocess", resolve)
    monkeypatch.setattr("itzi.itzi.validate_resolved_ensemble", validate)
    monkeypatch.setattr("itzi.itzi._validate_manifest_destination", lambda *_, **__: None)
    monkeypatch.setattr("itzi.itzi._preflight_ensemble_in_subprocess", preflight)
    monkeypatch.setattr("itzi.itzi._run_one_ensemble", run)
    monkeypatch.setattr("itzi.itzi._display_dry_plan", display)

    _run_ensemble_batch(
        SimpleNamespace(config_file=["first.yaml", "second.yaml"], member=[], dry=dry, o=True)
    )

    assert calls == [
        (step, name)
        for name in ("first", "second")
        for step in ("resolve", "validate", "preflight", "display" if dry else "run")
    ]


def test_preflight_only_checks_selected_members(tmp_path, monkeypatch):
    selected_simulation = SimpleNamespace(simulation_id="sim-selected")
    selected_with_failure = SimpleNamespace(simulation_id="sim-failed")
    unselected_simulation = SimpleNamespace(simulation_id="sim-unselected")
    ensemble = SimpleNamespace(
        ensemble_id="study",
        manifest_template=None,
        source=SimpleNamespace(path=tmp_path / "study.yaml"),
    )
    calls = []
    monkeypatch.setattr("itzi.itzi._validate_manifest_destination", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "itzi.itzi._preflight_ensemble_in_subprocess",
        lambda simulations: (
            calls.append(tuple(s.simulation_id for s in simulations))
            or {"sim-failed": "invalid input"}
        ),
    )

    member_failures, manifest_failure = _preflight_ensemble(
        ResolvedEnsemble(
            ensemble, (selected_simulation, selected_with_failure, unselected_simulation), ()
        ),
        {"sim-selected", "sim-failed"},
        has_selectors=True,
        overwrite=False,
    )

    assert calls == [("sim-selected", "sim-failed")]
    assert member_failures == {"sim-failed": {"phase": "preflight", "detail": "invalid input"}}
    assert manifest_failure is None


def test_select_members_only_matches_current_ensemble():
    resolved = ResolvedEnsemble(
        SimpleNamespace(ensemble_id="first"), (SimpleNamespace(simulation_id="member"),), ()
    )
    assert _select_members(resolved, [("second#member", "second#member")]) == set()
    assert _select_members(resolved, [("first#member", "first#member")]) == {"member"}


def test_unqualified_selector_requires_one_ensemble(monkeypatch):
    ensembles = tuple(SimpleNamespace(ensemble_id=name) for name in ("first", "second"))
    monkeypatch.setattr("itzi.itzi.load_batch", lambda _: (ensembles, ()))

    with pytest.raises(RuntimeError, match="Unqualified --member IDs"):
        _run_ensemble_batch(
            SimpleNamespace(config_file=["first.yaml", "second.yaml"], member=["member"])
        )


def test_one_ensemble_runs_only_planned_members_and_counts_failures(tmp_path, monkeypatch):
    ensemble = SimpleNamespace(
        ensemble_id="study",
        source=SimpleNamespace(path=tmp_path / "study.yaml"),
        manifest_template=None,
    )
    artifacts = SimpleNamespace(output_map_names=(), drainage_output=None, statistics_file=None)
    simulations = tuple(
        SimpleNamespace(simulation_id=name, coordinates=(), artifacts=artifacts)
        for name in ("preflight", "unselected", "good", "bad")
    )
    runs = []
    updates = []
    preflights = []
    monkeypatch.setattr("itzi.itzi._create_manifest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "itzi.itzi._update_manifest",
        lambda _path, _ensemble, states: updates.append(
            {key: value["status"] for key, value in states.items()}
        ),
    )
    monkeypatch.setattr(
        "itzi.itzi._preflight_in_subprocess",
        lambda simulations: preflights.append(simulations[0].simulation_id) or {},
    )
    monkeypatch.setattr(
        "itzi.itzi._run_simulation_in_subprocess",
        lambda simulation: (
            runs.append(simulation.simulation_id)
            or (
                ("execution_failed", "failed")
                if simulation.simulation_id == "bad"
                else ("completed", None)
            )
        ),
    )

    count = _run_one_ensemble(
        ResolvedEnsemble(ensemble, simulations, ()),
        {"good", "bad", "preflight"},
        {"preflight": {"phase": "preflight", "detail": "invalid input"}},
        None,
        has_selectors=True,
        overwrite=False,
    )

    assert count == 2
    assert runs == ["good", "bad"]
    assert preflights == ["bad"]
    assert updates[-1] == {
        "good": "completed",
        "bad": "execution_failed",
        "preflight": "validation_failed",
        "unselected": "not_selected",
    }


def test_preflight_ensemble_worker_keeps_member_failures(monkeypatch):
    calls = []

    class FakeGrassSessionManager:
        def __init__(self, params):
            assert params == "shared"

        def __enter__(self):
            calls.append("open")

        def __exit__(self, *_):
            calls.append("close")

    def preflight(simulation):
        calls.append(simulation.simulation_id)
        if simulation.simulation_id == "bad":
            raise ValueError("invalid input")

    monkeypatch.setattr("itzi.itzi.GrassSessionManager", FakeGrassSessionManager)
    fake_preflight = ModuleType("itzi.ensemble.preflight")
    fake_preflight.preflight_simulation = preflight
    monkeypatch.setitem(sys.modules, "itzi.ensemble.preflight", fake_preflight)
    simulations = tuple(
        SimpleNamespace(simulation_id=simulation_id, grass_params="shared")
        for simulation_id in ("good", "bad", "another")
    )

    assert preflight_ensemble_worker(simulations) == {"bad": "ValueError: invalid input"}
    assert calls == ["open", "good", "bad", "another", "close"]


def test_run_parser_rejects_v_and_q_together():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "a.ini", "-v", "-q"])


def test_prints_version(monkeypatch, capsys):
    monkeypatch.setattr("itzi.itzi.version", lambda _: "22.2")
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == "22.2"


def test_main_returns_error_status_for_fatal_error(monkeypatch, itzi_stderr):
    def fail(_):
        msgr.fatal("expected failure")

    monkeypatch.setattr("itzi.itzi.itzi_run", fail)

    assert main(["run", "a.ini"]) == 1
    stderr = itzi_stderr.getvalue()
    assert stderr.count("ERROR: expected failure") == 1
    assert "Traceback" not in stderr


def test_main_propagates_unexpected_error(monkeypatch):
    def fail(_):
        raise ValueError("unexpected")

    monkeypatch.setattr("itzi.itzi.itzi_run", fail)

    with pytest.raises(ValueError, match="unexpected"):
        main(["run", "a.ini"])


def test_worker_does_not_format_fatal_error_as_traceback(monkeypatch, itzi_stderr):
    def fail(_):
        msgr.fatal("expected worker failure")

    monkeypatch.setenv("ITZI_VERBOSE", str(VerbosityLevel.QUIET))
    monkeypatch.setattr("itzi.itzi.ConfigReader", fail)

    with pytest.raises(SystemExit) as error:
        sim_runner_worker("a.ini", None)

    assert error.value.code == 1
    stderr = itzi_stderr.getvalue()
    assert stderr.count("ERROR: expected worker failure") == 1
    assert "Traceback" not in stderr
    assert "WARNING: Error during execution" not in stderr


def test_worker_reports_unexpected_error_without_traceback(monkeypatch, itzi_stderr):
    def fail(_):
        raise ValueError("unexpected worker failure")

    monkeypatch.setenv("ITZI_VERBOSE", str(VerbosityLevel.QUIET))
    monkeypatch.setattr("itzi.itzi.ConfigReader", fail)

    with pytest.raises(SystemExit) as error:
        sim_runner_worker("a.ini", None)

    assert error.value.code == 1
    stderr = itzi_stderr.getvalue()
    assert "WARNING: Error during execution: ValueError: unexpected worker failure" in stderr
    assert "Traceback" not in stderr


def test_worker_reports_system_exit_without_traceback(monkeypatch, itzi_stderr):
    def fail(_):
        raise SystemExit("GRASS failure")

    monkeypatch.setenv("ITZI_VERBOSE", str(VerbosityLevel.QUIET))
    monkeypatch.setattr("itzi.itzi.ConfigReader", fail)

    with pytest.raises(SystemExit) as error:
        sim_runner_worker("a.ini", None)

    assert error.value.code == 1
    stderr = itzi_stderr.getvalue()
    assert "WARNING: Simulation terminated with GRASS failure" in stderr
    assert "Traceback" not in stderr


def test_worker_formats_numeric_system_exit_as_status(monkeypatch, itzi_stderr):
    def fail(_):
        raise SystemExit(1)

    monkeypatch.setenv("ITZI_VERBOSE", str(VerbosityLevel.QUIET))
    monkeypatch.setattr("itzi.itzi.ConfigReader", fail)

    with pytest.raises(SystemExit):
        sim_runner_worker("a.ini", None)

    assert "WARNING: Simulation terminated with exit status 1" in itzi_stderr.getvalue()


def test_worker_passes_statistics_file_to_simulation_runner(monkeypatch):
    sim_params = object()
    grass_params = object()
    runner_arguments = {}

    class FakeConfigReader:
        def __init__(self, _):
            self.sim_config = sim_params
            self.grass_params = grass_params
            self.stats_file = "statistics.csv"

    class FakeGrassSessionManager:
        def __init__(self, received_grass_params):
            assert received_grass_params is grass_params

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    class FakeSimulationRunner:
        def __init__(self, *args, **kwargs):
            runner_arguments["args"] = args
            runner_arguments["kwargs"] = kwargs

        def initialize(self):
            return self

        def run(self):
            return self

        def finalize(self):
            return self

    monkeypatch.setattr("itzi.itzi.ConfigReader", FakeConfigReader)
    monkeypatch.setattr("itzi.itzi.GrassSessionManager", FakeGrassSessionManager)
    monkeypatch.setattr("itzi.itzi.SimulationRunner", FakeSimulationRunner)

    sim_runner_worker("a.ini", "hotstart.zip")

    assert runner_arguments == {
        "args": (sim_params, grass_params),
        "kwargs": {
            "hotstart_path": "hotstart.zip",
            "stats_file": "statistics.csv",
        },
    }


def test_run_one_reports_worker_signal(monkeypatch, itzi_stderr):
    class FailedProcess:
        exitcode = -11

        def start(self):
            pass

        def join(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr("itzi.itzi.Process", lambda **_: FailedProcess())

    assert itzi_run_one("a.ini", None) is False
    assert "WARNING: Execution of a.ini ended with an error (signal 11)" in (
        itzi_stderr.getvalue()
    )


def test_reconcile_hotstart_commands_accepts_single_resume_for_single_config():
    assert reconcile_hotstart_commands(["/tmp/a.ini"], [(None, "restart_a.zip")]) == [
        ("/tmp/a.ini", "restart_a.zip"),
    ]


def test_reconcile_hotstart_commands_matches_multiple_named_values():
    config_file_list = ["/tmp/a.ini", "/tmp/b.ini", "/tmp/c.ini"]
    resume_from_list = [("c.ini", "restart_c.zip"), ("a.ini", "restart_a.zip")]

    assert reconcile_hotstart_commands(config_file_list, resume_from_list) == [
        ("/tmp/a.ini", "restart_a.zip"),
        ("/tmp/b.ini", None),
        ("/tmp/c.ini", "restart_c.zip"),
    ]


def test_reconcile_hotstart_commands_accepts_duplicate_basenames_with_paths():
    config_file_list = ["./sim1/config.ini", "sim2/config.ini"]
    resume_from_list = [
        ("./sim1/config.ini", "sim1/hotstart.zip"),
        ("sim2/config.ini", "sim2/hotstart.zip"),
    ]

    assert reconcile_hotstart_commands(config_file_list, resume_from_list) == [
        ("./sim1/config.ini", "sim1/hotstart.zip"),
        ("sim2/config.ini", "sim2/hotstart.zip"),
    ]


def test_reconcile_hotstart_commands_rejects_single_resume_for_multiple_configs():
    with pytest.raises(RuntimeError):
        reconcile_hotstart_commands(["/tmp/a.ini", "/tmp/b.ini"], [(None, "restart.zip")])


def test_reconcile_hotstart_commands_accepts_single_named_resume_for_multiple_configs():
    assert reconcile_hotstart_commands(
        ["/tmp/a.ini", "/tmp/b.ini"],
        [("a.ini", "restart_a.zip")],
    ) == [
        ("/tmp/a.ini", "restart_a.zip"),
        ("/tmp/b.ini", None),
    ]


def test_reconcile_hotstart_commands_rejects_unnamed_values_in_batch_mode():
    with pytest.raises(RuntimeError):
        reconcile_hotstart_commands(
            ["/tmp/a.ini", "/tmp/b.ini"],
            [("a.ini", "restart_a.zip"), (None, "restart_b.zip")],
        )


def test_reconcile_hotstart_commands_rejects_unknown_config_key():
    with pytest.raises(RuntimeError):
        reconcile_hotstart_commands(
            ["/tmp/a.ini", "/tmp/b.ini"],
            [("missing.ini", "restart.zip"), ("b.ini", "restart_b.zip")],
        )


def test_itzi_run_sets_env_and_dispatches(monkeypatch):
    calls = []
    messages = []

    def record_run(conf_file, hotstart_file):
        calls.append((conf_file, hotstart_file))
        return True

    monkeypatch.setattr("itzi.itzi.itzi_run_one", record_run)
    monkeypatch.setattr("itzi.itzi.msgr.message", messages.append)

    args = argparse.Namespace(
        config_file=["a.ini", "b.ini"],
        o=True,
        v=1,
        q=None,
        resume_from=[("a.ini", "restart_a.zip"), ("b.ini", "restart_b.zip")],
    )

    itzi_run(args)

    assert calls == [("a.ini", "restart_a.zip"), ("b.ini", "restart_b.zip")]
    assert os.environ["GRASS_OVERWRITE"] == "1"
    assert os.environ["ITZI_VERBOSE"] == str(VerbosityLevel.VERBOSE)
    assert os.environ["GRASS_VERBOSE"] == "2"
    assert any("Simulation(s) complete" in m for m in messages)


def test_main_returns_error_status_when_simulation_fails(monkeypatch, itzi_stderr):
    monkeypatch.setattr("itzi.itzi.itzi_run_one", lambda *_: False)
    msgr._itzi_logger.set_verbosity(VerbosityLevel.MESSAGE)

    assert main(["run", "a.ini"]) == 1
    stderr = itzi_stderr.getvalue()
    assert "Simulation run(s) finished with errors" in stderr
    assert "ERROR: 1 simulation(s) failed" in stderr
    assert "Traceback" not in stderr
