"""Test the CLI"""

import argparse
import os

import pytest

from itzi.const import VerbosityLevel
from itzi.itzi import main, itzi_run
from itzi.parser import build_parser


def test_run_parser_accepts_multiple_config_files():
    args = build_parser().parse_args(["run", "a.ini", "b.ini", "-o", "-vv"])
    assert args.config_file == ["a.ini", "b.ini"]
    assert args.o is True
    assert args.v == 2
    assert args.q is None


def test_run_parser_rejects_v_and_q_together():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "a.ini", "-v", "-q"])


def test_prints_version(monkeypatch, capsys):
    monkeypatch.setattr("itzi.itzi.version", lambda _: "22.2")
    assert main(["version"]) is None
    assert capsys.readouterr().out.strip() == "22.2"


def test_itzi_run_sets_env_and_dispatches(monkeypatch):
    calls = []
    messages = []

    monkeypatch.setattr("itzi.itzi.itzi_run_one", calls.append)
    monkeypatch.setattr("itzi.itzi.msgr.message", messages.append)

    args = argparse.Namespace(
        config_file=["a.ini", "b.ini"],
        o=True,
        v=1,
        q=None,
    )

    itzi_run(args)

    assert calls == ["a.ini", "b.ini"]
    assert os.environ["GRASS_OVERWRITE"] == "1"
    assert os.environ["ITZI_VERBOSE"] == str(VerbosityLevel.VERBOSE)
    assert os.environ["GRASS_VERBOSE"] == "2"
    assert any("Simulation(s) complete" in m for m in messages)
