"""GRASS integration coverage for basic spawned YAML ensemble execution."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import grass.script as gscript
import grass.temporal as tgis
import pytest
import yaml
from grass.pygrass import utils as gutils
from itzi_core import TemporalType

from itzi.cli_parser import build_parser
from itzi.ensemble_models import EnsembleError
from itzi.grass_session import GrassSessionManager
from itzi.itzi import itzi_run
from itzi.preflight import _validate_grass_outputs


@pytest.fixture
def stop_temporal_subprocesses() -> Iterator[None]:
    yield
    tgis.stop_subprocesses()


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
def test_yaml_ensemble_runs_in_spawned_resolver_and_worker(test_data_temp_path):
    config_path = Path(test_data_temp_path) / "source-file-name.yaml"
    config_path.write_text(
        """\
schema_version: 1
ensemble:
  id: basic
domain: {}
time:
  duration: "00:01:00"
  record_step: "00:00:30"
input:
  ground_elevation: "z"
  friction: "n"
  infiltration:
    - type: "constant"
      rate: "infiltration_rate"
options:
  cfl: [0.2, 0.3]
  dtmax: 0.3
outputs:
  rasters:
    prefix: "basic_{simulation}"
    variables: [water_depth]
""",
        encoding="utf-8",
    )
    args = build_parser().parse_args(["run", str(config_path), "-o"])

    itzi_run(args)

    manifest_path = Path(test_data_temp_path) / "results" / "basic.manifest.yaml"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert not manifest_text.lstrip().startswith("{")
    assert manifest_text.splitlines()[1].startswith("last_updated_at: ")
    manifest = yaml.safe_load(manifest_text)
    last_updated_at = manifest["last_updated_at"]
    assert isinstance(last_updated_at, str)
    assert datetime.fromisoformat(last_updated_at).utcoffset() is not None
    assert len(manifest["members"]) == 2
    assert all(member["status"] == "completed" for member in manifest["members"])
    assert [member["coordinates"] for member in manifest["members"]] == [
        {
            "input.infiltration": {"rate": "infiltration_rate", "type": "constant"},
            "options.cfl": 0.2,
        },
        {
            "input.infiltration": {"rate": "infiltration_rate", "type": "constant"},
            "options.cfl": 0.3,
        },
    ]
    assert all(
        member["artifacts"]["rasters"]["water_depth"].startswith("basic_sim-")
        for member in manifest["members"]
    )


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
def test_explicit_grass_context_runs_four_members_without_parent_session(test_data_temp_path):
    context = gscript.gisenv()
    config_path = Path(test_data_temp_path) / "explicit.yaml"
    config_path.write_text(
        f"""\
schema_version: 1
ensemble:
  id: explicit
domain:
  grass:
    database: "{context["GISDBASE"]}"
    project: "{context["LOCATION_NAME"]}"
    mapset: "{context["MAPSET"]}"
    executable: "grass"
time:
  duration: "00:01:00"
  record_step: "00:00:30"
input:
  ground_elevation: "z"
  friction: "n"
  water_depth: "start_h"
options:
  cfl: [0.2, 0.3]
  theta: [0.8, 0.9]
  dtmax: 0.3
outputs:
  rasters:
    prefix: "explicit_{{simulation}}"
    variables: [water_depth]
  manifest:
    file: "explicit-manifest.yaml"
""",
        encoding="utf-8",
    )
    args = build_parser().parse_args(["run", str(config_path), "-o"])

    gisrc = os.environ.pop("GISRC")
    try:
        itzi_run(args)
    finally:
        os.environ["GISRC"] = gisrc

    manifest_path = Path(test_data_temp_path) / "explicit-manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["members"]) == 4
    assert all(member["status"] == "completed" for member in manifest["members"])
    mapset_path = (
        Path(context["GISDBASE"]) / context["LOCATION_NAME"] / context["MAPSET"] / "cellhd"
    )
    assert {
        f"{member['artifacts']['rasters']['water_depth']}_0000" for member in manifest["members"]
    } <= {path.name for path in mapset_path.iterdir()}


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
def test_dry_run_preflights_multiple_members_in_one_session(tmp_path):
    prefix = f"dry_multi_{uuid4().hex[:8]}"
    config_path = tmp_path / "dry-multi.yaml"
    config_path.write_text(
        f"""\
schema_version: 1
ensemble:
  id: dry-multi
domain: {{}}
time:
  duration: "00:01:00"
  record_step: "00:00:30"
input:
  ground_elevation: "z"
  friction: "n"
options:
  cfl: [0.2, 0.3]
outputs:
  rasters:
    prefix: "{prefix}_{{simulation}}"
    variables: [water_depth]
""",
        encoding="utf-8",
    )

    itzi_run(build_parser().parse_args(["run", str(config_path), "--dry-run"]))

    assert not (tmp_path / "results" / "dry-multi.manifest.yaml").exists()
    assert not gscript.list_strings(type="raster", pattern=f"{prefix}_*")


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
def test_dry_run_rejects_null_elevation_without_creating_outputs(tmp_path):
    gscript.mapcalc("dry_null_dem=null()")
    config_path = tmp_path / "dry.yaml"
    config_path.write_text(
        """\
schema_version: 1
ensemble:
  id: dry
domain: {}
time:
  duration: "00:01:00"
  record_step: "00:00:30"
input:
  ground_elevation: "dry_null_dem"
  friction: "n"
options: {}
outputs:
  rasters:
    prefix: "dry_output"
    variables: [water_depth]
  statistics:
    file: "dry-results/statistics.csv"
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="YAML batch validation failed"):
        itzi_run(build_parser().parse_args(["run", str(config_path), "--dry-run"]))

    assert not (tmp_path / "results" / "dry.manifest.yaml").exists()
    assert not (tmp_path / "dry-results").exists()
    assert not gscript.find_file(name="dry_output_water_depth_0000", element="cell").get("file")

    with pytest.raises(RuntimeError, match="failed validation or execution"):
        itzi_run(build_parser().parse_args(["run", str(config_path)]))

    manifest = yaml.safe_load(
        (tmp_path / "results" / "dry.manifest.yaml").read_text(encoding="utf-8")
    )
    assert manifest["members"][0]["status"] == "validation_failed"
    assert manifest["members"][0]["failure"]["phase"] == "preflight"
    assert not (tmp_path / "dry-results").exists()
    assert not gscript.find_file(name="dry_output_water_depth_0000", element="cell").get("file")


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
def test_dry_run_does_not_read_rainfall_cells(tmp_path):
    gscript.mapcalc("dry_null_rain_0=null()")
    gscript.mapcalc("dry_null_rain_1=1")
    gscript.run_command(
        "t.create",
        output="dry_rain_series",
        type="strds",
        temporaltype="relative",
        semantictype="mean",
        title="dry_rain_series",
        description="dry_rain_series",
    )
    gscript.run_command(
        "t.register",
        flags="i",
        input="dry_rain_series",
        type="raster",
        maps="dry_null_rain_0,dry_null_rain_1",
        start="0",
        increment="30",
        unit="seconds",
    )
    config_path = tmp_path / "dry-rain.yaml"
    config_path.write_text(
        """\
schema_version: 1
ensemble:
  id: dry-rain
domain: {}
time:
  duration: "00:01:00"
  record_step: "00:00:30"
input:
  ground_elevation: "z"
  friction: "n"
  rainfall_rate: "dry_rain_series"
options: {}
outputs:
  rasters:
    prefix: "dry_rain_output"
    variables: [water_depth]
""",
        encoding="utf-8",
    )

    itzi_run(build_parser().parse_args(["run", str(config_path), "--dry-run"]))
    assert not (tmp_path / "results" / "dry-rain.manifest.yaml").exists()


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
def test_dry_run_checks_every_generated_output_record(test_data_temp_path):
    gscript.mapcalc("existing_output_water_depth_0002=1")
    config_path = Path(test_data_temp_path) / "existing-output.yaml"
    config_path.write_text(
        """\
schema_version: 1
ensemble:
  id: existing-output
domain: {}
time:
  duration: "00:01:00"
  record_step: "00:00:30"
input:
  ground_elevation: "z"
  friction: "n"
options: {}
outputs:
  rasters:
    prefix: "existing_output"
    variables: [water_depth]
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="YAML batch validation failed"):
        itzi_run(build_parser().parse_args(["run", str(config_path), "--dry-run"]))

    assert not (Path(test_data_temp_path) / "results" / "existing-output.manifest.yaml").exists()
    assert not gscript.find_file(name="existing_output_water_depth_0000", element="cell").get(
        "file"
    )


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5", "stop_temporal_subprocesses")
def test_preflight_checks_existing_drainage_tables():
    GrassSessionManager.ensure_temporal_initialized()
    mapset = gutils.getenv("MAPSET")
    database = (
        Path(gutils.getenv("GISDBASE"))
        / gutils.getenv("LOCATION_NAME")
        / mapset
        / "sqlite"
        / "sqlite.db"
    )
    name = f"drainage_collision_{uuid4().hex[:8]}"
    simulation = SimpleNamespace(
        simulation_config=SimpleNamespace(
            start_time=datetime(2020, 1, 1),
            end_time=datetime(2020, 1, 1) + timedelta(minutes=1),
            record_step=timedelta(seconds=30),
            output_map_names={},
            drainage_output=name,
            temporal_type=TemporalType.RELATIVE,
        )
    )
    interface = SimpleNamespace(
        overwrite=False,
        get_current_mapset=lambda: mapset,
        format_id=lambda value: f"{value}@{mapset}",
        validate_output_stds_temporal_type=lambda *_: None,
    )

    database_existed = database.exists()
    _validate_grass_outputs(simulation, interface)
    assert database.exists() == database_existed
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(f"CREATE TABLE {name}_0002_link (cat integer)")
        connection.commit()
    with pytest.raises(EnsembleError, match=f"drainage table <{name}_0002_link> already exists"):
        _validate_grass_outputs(simulation, interface)
