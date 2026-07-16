"""Test the CLI"""

import argparse
import os

import pytest

from itzi.itzi import main, itzi_run, reconcile_hotstart_commands, VerbosityLevel
from itzi.cli_parser import build_parser


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
    assert main(["version"]) is None
    assert capsys.readouterr().out.strip() == "22.2"


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

    monkeypatch.setattr(
        "itzi.itzi.itzi_run_one", lambda conf, hotstart: calls.append((conf, hotstart))
    )
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
