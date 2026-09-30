"""Pure YAML ensemble loader and expansion tests."""

from __future__ import annotations

from datetime import datetime, timedelta
import sys
from types import SimpleNamespace

import pytest
from itzi_core import InfiltrationModelType, SurfaceFlowParameters, TemporalType
from itzi_core.const import DefaultValues
from pydantic import ValidationError

from itzi.ensemble import load_yaml_stream
from itzi.ensemble.models import (
    MAX_ENSEMBLE_MEMBERS,
    format_duration,
    parse_duration,
    render_template,
)
from itzi.ensemble.schema import (
    EnsembleMetadata,
    InputSweepConfig,
    ManifestOutputs,
    OptionSweepConfig,
    StatisticsOutputs,
    TimeConfig,
)
from itzi.ensemble.resolution import (
    _last_record_index,
    _render_artifacts,
    _resolve_inputs,
    validate_resolved_ensemble,
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
  duration: "02:00:00"
  record_step: "00:05:00"
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
    assert dict(simulation.options) == SurfaceFlowParameters().model_dump() | {
        "dtinf": DefaultValues.DTINF
    }


def test_time_config_accepts_only_supported_combinations():
    start = "2026-09-01T00:00:00"
    end = "2026-09-01T02:00:00"
    duration = "02:00:00"
    for fields in (
        {"duration": duration},
        {"start": start, "duration": duration},
        {"start": start, "end": end},
    ):
        TimeConfig.model_validate({"record_step": "00:05:00", **fields})
    for fields in (
        {},
        {"start": start},
        {"end": end},
        {"end": end, "duration": duration},
        {"start": start, "end": end, "duration": duration},
    ):
        with pytest.raises(ValidationError):
            TimeConfig.model_validate({"record_step": "00:05:00", **fields})


@pytest.mark.parametrize(
    ("model", "value"),
    (
        (EnsembleMetadata, {"id": "invalid!"}),
        (InputSweepConfig, {"ground_elevation": [], "friction": "manning"}),
        (InputSweepConfig, {"ground_elevation": ["elevation"], "friction": [["manning"]]}),
        (OptionSweepConfig, {"cfl": [float("nan")]}),
        (OptionSweepConfig, {"cfl": [0.5, 0.5]}),
        (StatisticsOutputs, {"file": ""}),
        (ManifestOutputs, {"file": ""}),
    ),
)
def test_schema_constraints_reject_invalid_values(model, value):
    with pytest.raises(ValidationError):
        model.model_validate(value)


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
  duration: "01:00:00"
  record_step: "00:05:00"
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
  duration: "01:00:00"
  record_step: "00:05:00"
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


def test_loader_rejects_non_json_yaml_and_keeps_later_document(tmp_path):
    content = (
        document(ensemble_id="binary")
        + "extra: !!binary AQI=\n---\n"
        + document(ensemble_id="valid")
    )

    stream = load_yaml_stream(write_yaml(tmp_path, content))

    assert [failure.phase for failure in stream.failures] == ["parse"]
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

    duplicate = content.replace("    - type: none\n", "    - type: none\n    - type: none\n")
    assert load_yaml_stream(write_yaml(tmp_path, duplicate)).failures[0].phase == "schema"


def test_yaml_rejects_unsupported_hotstart(tmp_path):
    content = (
        document()
        + """\
hotstart:
  wallclock_step: "00:01:00"
  file: "checkpoint.zip"
"""
    )

    stream = load_yaml_stream(write_yaml(tmp_path, content))

    assert stream.ensembles == ()
    assert stream.failures[0].phase == "schema"


def test_absolute_offsets_keep_wall_clock_values(tmp_path):
    content = document().replace(
        '  duration: "02:00:00"\n  record_step: "00:05:00"',
        '  start: "2026-09-01T00:00:00+02:00"\n  duration: "02:00:00"\n  record_step: "00:05:00"',
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
    for str_value, expected in (
        ("25:02:03", timedelta(hours=25, minutes=2, seconds=3)),
        ("123:55:20", timedelta(hours=123, minutes=55, seconds=20)),
    ):
        assert parse_duration(str_value) == expected
    assert format_duration(timedelta(days=1, hours=2, minutes=3, seconds=4)) == "26:03:04"
    for value in (
        "PT1H",
        "01:00",
        "01:60:00",
        "00:00:00",
        "-01:00:00",
        "2:31:00:00",
        "une:heure:trente",
        "00:90:00",
    ):
        with pytest.raises(ValueError):
            parse_duration(value)
    assert (
        render_template("{{{ensemble}}}-{simulation}", ensemble="study", simulation="sim-a")
        == "{study}-sim-a"
    )


def test_statistics_file_template_is_rendered(tmp_path, monkeypatch):
    expanded = SimpleNamespace(
        ensemble_id="study",
        source=SimpleNamespace(path=tmp_path / "study.yaml"),
        outputs=SimpleNamespace(
            raster_prefix=None,
            raster_variables=(),
            statistics_file="results/{simulation}.csv",
            drainage_dataset=None,
        ),
        drainage=None,
        time=SimpleNamespace(duration=timedelta(hours=1), record_step=timedelta(minutes=5)),
    )
    monkeypatch.setattr("itzi.ensemble.resolution._validate_output_names", lambda *_: None)

    artifacts = _render_artifacts(expanded, "sim-a", "PERMANENT")

    assert artifacts.statistics_file == tmp_path / "results/sim-a.csv"


def test_generated_output_inventory_includes_final_partial_record():
    assert _last_record_index(timedelta(seconds=60), timedelta(seconds=30)) == 2
    assert _last_record_index(timedelta(seconds=61), timedelta(seconds=30)) == 3


def test_later_generated_output_cannot_alias_an_input():
    start = datetime(2026, 1, 1)  # noqa: DTZ001
    config = SimpleNamespace(
        start_time=start,
        end_time=start + timedelta(minutes=1),
        record_step=timedelta(seconds=30),
        input_map_names={},
        swmm_inp=None,
    )
    output = SimpleNamespace(
        simulation_id="sim-output",
        normalized_payload="output",
        grass_params=SimpleNamespace(mapset="PERMANENT"),
        effective_mask=SimpleNamespace(source=None),
        simulation_config=config,
        artifacts=SimpleNamespace(
            output_map_names=(("water_depth", "result@PERMANENT"),),
            drainage_output=None,
            statistics_file=None,
        ),
    )
    input_simulation = SimpleNamespace(
        simulation_id="sim-input",
        normalized_payload="input",
        grass_params=SimpleNamespace(mapset="PERMANENT"),
        effective_mask=SimpleNamespace(source=None),
        simulation_config=SimpleNamespace(
            start_time=start,
            end_time=start + timedelta(minutes=1),
            record_step=timedelta(seconds=30),
            input_map_names={"ground_elevation": "result_0002@PERMANENT"},
            swmm_inp=None,
        ),
        artifacts=SimpleNamespace(
            output_map_names=(),
            drainage_output=None,
            statistics_file=None,
        ),
    )

    with pytest.raises(ValueError, match="result_0002@PERMANENT.*aliases"):
        validate_resolved_ensemble((output, input_simulation))


def test_input_resolution_cache_reuses_shared_identifiers(monkeypatch):
    calls = []

    def resolve(identifier):
        calls.append(identifier)
        return f"{identifier}@PERMANENT", "raster"

    cache = {}
    monkeypatch.setitem(
        sys.modules, "itzi.grass.utils", SimpleNamespace(resolve_input_identifier=resolve)
    )

    _resolve_inputs(
        {"ground_elevation": "shared", "rainfall_rate": "first"},
        cache=cache,
    )
    resolved, _ = _resolve_inputs(
        {"ground_elevation": "shared", "rainfall_rate": "second"},
        cache=cache,
    )

    assert calls == ["shared", "first", "second"]
    assert resolved["ground_elevation"] == "shared@PERMANENT"
