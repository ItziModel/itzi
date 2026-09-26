"""Pure YAML ensemble loader and expansion tests."""

from __future__ import annotations

from datetime import datetime, timedelta

from itzi_core import InfiltrationModelType, TemporalType

from itzi.ensemble import load_yaml_stream
from itzi.ensemble_models import (
    MAX_ENSEMBLE_MEMBERS,
    format_iso_duration,
    parse_iso_duration,
    render_template,
)


def write_yaml(tmp_path, content: str):
    path = tmp_path / "study.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def document(*, ensemble_id: str = "study", extra: str = "") -> str:
    body = (
        extra
        or """\
input:
  ground_elevation: "elevation"
  friction: "manning"
options: {}
outputs:
  rasters:
    prefix: "results_{ensemble}_{simulation}"
    variables: [water_depth]
"""
    )
    return f"""\
schema_version: 1
ensemble:
  id: {ensemble_id}
domain: {{}}
time:
  duration: "PT2H"
  record_step: "PT5M"
{body}"""


def test_scalar_document_expands_to_one_member(tmp_path):
    stream = load_yaml_stream(write_yaml(tmp_path, document()))

    assert stream.failures == ()
    assert len(stream.ensembles) == 1
    simulation = stream.ensembles[0].simulations[0]
    assert simulation.coordinates == ()
    assert simulation.time.temporal_type == TemporalType.RELATIVE
    assert simulation.time.start is None
    assert simulation.time.end is None
    assert simulation.time.duration == timedelta(hours=2)


def test_cartesian_sweeps_use_canonical_coordinate_order(tmp_path):
    content = document(
        extra="""\
input:
  ground_elevation: ["elevation_a", "elevation_b"]
  friction: "manning"
  rainfall_rate: ["rain_a", "rain_b"]
options:
  cfl: [0.5, 0.7]
outputs:
  rasters:
    prefix: "out_{simulation}"
    variables: [water_depth]
"""
    )

    stream = load_yaml_stream(write_yaml(tmp_path, content))

    assert stream.failures == ()
    simulations = stream.ensembles[0].simulations
    assert len(simulations) == 8
    assert tuple(path for path, _ in simulations[0].coordinates) == (
        "input.ground_elevation",
        "input.rainfall_rate",
        "options.cfl",
    )
    assert dict(simulations[0].coordinates)["options.cfl"] == 0.5
    assert dict(simulations[-1].coordinates)["options.cfl"] == 0.7


def test_loader_continues_after_failed_explicit_document(tmp_path):
    content = """\
schema_version: 1
ensemble:
  id: first
domain: {}
time:
  duration: "PT1H"
  record_step: "PT5M"
input:
  ground_elevation: elevation
  friction: manning
options:
  cfl: "0.7"
outputs: {}
---
""" + document(ensemble_id="second")

    stream = load_yaml_stream(write_yaml(tmp_path, content))

    assert len(stream.failures) == 1
    assert stream.failures[0].document_index == 0
    assert stream.failures[0].phase == "schema"
    assert [ensemble.ensemble_id for ensemble in stream.ensembles] == ["second"]


def test_loader_rejects_duplicate_keys_and_keeps_later_document(tmp_path):
    content = """\
schema_version: 1
schema_version: 1
---
""" + document(ensemble_id="second")

    stream = load_yaml_stream(write_yaml(tmp_path, content))

    assert stream.failures[0].phase == "parse"
    assert stream.failures[0].line == 2
    assert [ensemble.ensemble_id for ensemble in stream.ensembles] == ["second"]


def test_loader_rejects_empty_documents_and_merge_keys(tmp_path):
    content = """\
---
---
defaults: &defaults
  cfl: 0.7
schema_version: 1
ensemble:
  id: invalid-merge
domain: {}
time:
  duration: "PT1H"
  record_step: "PT5M"
input:
  ground_elevation: elevation
  friction: manning
options:
  <<: *defaults
outputs: {}
---
""" + document(ensemble_id="valid")

    stream = load_yaml_stream(write_yaml(tmp_path, content))

    assert [failure.phase for failure in stream.failures] == ["parse", "parse"]
    assert [ensemble.ensemble_id for ensemble in stream.ensembles] == ["valid"]


def test_infiltration_alternatives_are_explicit(tmp_path):
    content = document(
        extra="""\
input:
  ground_elevation: elevation
  friction: manning
  infiltration:
    - type: none
    - type: constant
      rate: infiltration_rate
options: {}
outputs: {}
"""
    )
    stream = load_yaml_stream(write_yaml(tmp_path, content))

    assert stream.failures == ()
    simulations = stream.ensembles[0].simulations
    assert [simulation.infiltration.model for simulation in simulations] == [
        InfiltrationModelType.NULL,
        InfiltrationModelType.CONSTANT,
    ]

    labelled = content.replace("    - type: none", '    - type: none\n      label: "first"')
    labelled_stream = load_yaml_stream(write_yaml(tmp_path, labelled))
    assert labelled_stream.failures[0].phase == "schema"


def test_yaml_rejects_unsupported_hotstart(tmp_path):
    content = (
        document()
        + """\
hotstart:
  wallclock_step: "PT1M"
  file: "checkpoint.zip"
"""
    )

    stream = load_yaml_stream(write_yaml(tmp_path, content))

    assert stream.ensembles == ()
    assert stream.failures[0].phase == "schema"


def test_absolute_offsets_keep_wall_clock_values(tmp_path):
    content = document().replace(
        '  duration: "PT2H"\n  record_step: "PT5M"',
        '  start: "2026-09-01T00:00:00+02:00"\n  duration: "PT2H"\n  record_step: "PT5M"',
    )

    stream = load_yaml_stream(write_yaml(tmp_path, content))

    simulation = stream.ensembles[0].simulations[0]
    assert simulation.time.temporal_type == TemporalType.ABSOLUTE
    assert simulation.time.start == datetime(2026, 9, 1, 0, 0)  # noqa: DTZ001
    assert simulation.time.end == datetime(2026, 9, 1, 2, 0)  # noqa: DTZ001
    assert simulation.time.had_timezone_offset is True


def test_expansion_limit_is_checked_before_materialization(tmp_path):
    values = ", ".join(f"{0.01 + index / 1000:.3f}" for index in range(MAX_ENSEMBLE_MEMBERS + 1))
    content = document(
        extra=f"""\
input:
  ground_elevation: elevation
  friction: manning
options:
  cfl: [{values}]
outputs: {{}}
"""
    )

    stream = load_yaml_stream(write_yaml(tmp_path, content))

    assert stream.ensembles == ()
    assert stream.failures[0].phase == "expansion"
    assert "exceeding the 100 limit" in stream.failures[0].detail


def test_duration_and_template_helpers_are_strict():
    assert parse_iso_duration("P1DT2H3M4.5S") == timedelta(days=1, hours=2, minutes=3, seconds=4.5)
    assert format_iso_duration(timedelta(minutes=5)) == "PT5M"
    assert (
        render_template("{{{ensemble}}}-{simulation}", ensemble="study", simulation="sim-a")
        == "{study}-sim-a"
    )
