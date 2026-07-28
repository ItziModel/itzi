"""Test the CLI"""

import argparse
import os

import pytest

import itzi.messenger as msgr
from itzi.cli_parser import build_parser
from itzi.itzi import (
    VerbosityLevel,
    itzi_run,
    itzi_run_one,
    main,
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
            pass

        def get_sim_params(self):
            return sim_params

        def get_grass_params(self):
            return grass_params

        def get_stats_file(self):
            return "statistics.csv"

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
