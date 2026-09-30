"""GRASS integration coverage for basic spawned YAML ensemble execution."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import grass.script as gscript
import numpy as np
import pytest
import yaml
from itzi_core import TemporalType

from itzi.cli_parser import build_parser
from itzi.ensemble.models import ResolvedSimulation, ValidationFailure
from itzi.grass.session import GrassSessionManager
from itzi.itzi import _resolve_ensemble_in_subprocess, itzi_run, run_validation_worker
from itzi.run_plan import load_batch


def write_study(tmp_path: Path, **changes: dict) -> Path:
    """Write a minimal runnable ensemble with optional nested YAML overrides."""
    document = {
        "schema_version": 1,
        "ensemble": {"id": "stage-one"},
        "grass": {},
        "time": {"duration": "00:01:00", "record_step": "00:00:30"},
        "input": {"ground_elevation": "z", "friction": "n"},
        "parameters": {"dtmax": 0.3},
        "outputs": {},
    }
    document["parameters"] = changes.pop("parameters", document["parameters"])
    for section, fields in changes.items():
        values = document[section]
        assert isinstance(values, dict)
        values.update(fields)
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def read_manifest(tmp_path: Path) -> dict:
    return yaml.safe_load(
        (tmp_path / "results" / "stage-one.manifest.yaml").read_text(encoding="utf-8")
    )


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
def test_input_resolution_qualifies_rasters_and_strds_and_rejects_ambiguity(
    tmp_path: Path,
) -> None:
    from grass.pygrass import utils as gutils

    GrassSessionManager.ensure_temporal_initialized()
    mapset = gutils.getenv("MAPSET")
    gscript.mapcalc("stage_rain=1")
    gscript.run_command(
        "t.create",
        output="stage_series",
        type="strds",
        temporaltype="relative",
        semantictype="mean",
        title="stage_series",
        description="stage_series",
    )
    gscript.run_command(
        "t.register",
        flags="i",
        input="stage_series",
        type="raster",
        maps="stage_rain",
        start="0",
        increment="60",
        unit="seconds",
    )
    path = write_study(
        tmp_path,
        input={"ground_elevation": f"z@{mapset}", "rainfall_rate": "stage_series"},
    )
    expanded = load_batch([str(path)])[0][0].simulations
    resolved = _resolve_ensemble_in_subprocess(expanded)[0]
    assert isinstance(resolved, ResolvedSimulation)
    assert resolved.simulation_config.input_map_names == {
        "ground_elevation": f"z@{mapset}",
        "friction": f"n@{mapset}",
        "rainfall_rate": f"stage_series@{mapset}",
    }
    assert dict(resolved.input_kinds)["rainfall_rate"] == "strds"
    assert run_validation_worker((resolved,)).get(resolved.simulation_id) is None

    path = write_study(tmp_path, input={"rainfall_rate": f"stage_series@{mapset}"})
    qualified = _resolve_ensemble_in_subprocess(load_batch([str(path)])[0][0].simulations)[0]
    assert isinstance(qualified, ResolvedSimulation)
    assert qualified.simulation_id == resolved.simulation_id

    gscript.mapcalc("stage_series=1")
    ambiguous = _resolve_ensemble_in_subprocess(load_batch([str(path)])[0][0].simulations)[0]
    assert isinstance(ambiguous, ValidationFailure)
    assert f"raster stage_series@{mapset}" in ambiguous.detail
    assert f"STRDS stage_series@{mapset}" in ambiguous.detail


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
def test_temporal_run_validation_failure_does_not_stop_next_member(tmp_path: Path) -> None:
    from grass.pygrass import utils as gutils

    GrassSessionManager.ensure_temporal_initialized()
    gscript.mapcalc("stage_short_rain=1")
    gscript.run_command(
        "t.create",
        output="stage_short_series",
        type="strds",
        temporaltype="relative",
        semantictype="mean",
        title="stage_short_series",
        description="stage_short_series",
    )
    gscript.run_command(
        "t.register",
        flags="i",
        input="stage_short_series",
        type="raster",
        maps="stage_short_rain",
        start="0",
        increment="30",
        unit="seconds",
    )
    path = write_study(
        tmp_path,
        input={"water_depth": "start_h", "rainfall_rate": ["stage_short_series", "rainfall"]},
        outputs={"rasters": {"prefix": "stage_{simulation}", "variables": ["water_depth"]}},
    )
    with pytest.raises(RuntimeError, match="failed validation or execution"):
        itzi_run(build_parser().parse_args(["run", str(path)]))

    members = read_manifest(tmp_path)["members"]
    assert sorted(member["status"] for member in members) == ["completed", "validation_failed"]
    failed = next(member for member in members if member["status"] == "validation_failed")
    completed = next(member for member in members if member["status"] == "completed")
    assert failed["failure"]["phase"] == "run_validation"
    assert (
        "ends before simulation" in failed["failure"]["detail"]
        or "inadequate temporal" in failed["failure"]["detail"]
    )
    output = completed["artifacts"]["rasters"]["water_depth"]
    assert output.endswith(f"@{gutils.getenv('MAPSET')}")
    assert gutils.get_mapset_raster(f"{output.partition('@')[0]}_0000", gutils.getenv("MAPSET"))


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
def test_worker_rejects_changed_mask_region_and_inputs_after_resolution(tmp_path: Path) -> None:
    from grass.pygrass import utils as gutils

    from itzi.grass.interface import GrassInterface

    gscript.mapcalc("stage_mask_values=if(col()==1, null(), if(col()==2, 0, 1))")
    gscript.run_command("g.copy", raster="stage_mask_values,MASK")
    try:
        path = write_study(tmp_path, grass={"mask": "stage_mask_values"})
        expanded = load_batch([str(path)])[0][0].simulations
        resolved = _resolve_ensemble_in_subprocess(expanded)[0]
        assert isinstance(resolved, ResolvedSimulation)
        assert resolved.effective_mask.mode == "explicit"

        config = resolved.simulation_config
        interface = GrassInterface(
            config.start_time,
            config.end_time,
            np.float32,
            None,
            "stage_mask_values",
            resolved.effective_mask,
        )
        try:
            explicit = interface.get_npmask()
            assert explicit[:, 0].all()
            assert not explicit[:, 1:].any()  # zero is inside an explicit mask
            assert np.isfinite(
                interface.read_raster_map(f"n@{gutils.getenv('MAPSET')}")
            ).all()  # bypass active MASK
            interface.mask_mode, interface.mask_source = (
                "active",
                f"MASK@{gutils.getenv('MAPSET')}",
            )
            active = interface.get_npmask()
            assert active[:, :2].all()  # zero is outside an active CELL mask
            assert not active[:, 2:].any()
        finally:
            interface.cleanup()

        assert run_validation_worker((resolved,)).get(resolved.simulation_id) is None
        gscript.run_command("r.mask", flags="r")
        assert run_validation_worker((resolved,)).get(resolved.simulation_id) is None

        active_path = write_study(tmp_path, grass={})
        active_resolved = _resolve_ensemble_in_subprocess(
            load_batch([str(active_path)])[0][0].simulations
        )[0]
        assert isinstance(active_resolved, ResolvedSimulation)
        assert active_resolved.effective_mask.mode == "none"
        gscript.run_command("g.copy", raster="stage_mask_values,MASK")
        assert "effective GRASS mask changed" in (
            run_validation_worker((active_resolved,)).get(active_resolved.simulation_id) or ""
        )
        assert gutils.get_mapset_raster("MASK", gutils.getenv("MAPSET"))
    finally:
        gscript.run_command("r.mask", flags="r")

    gscript.run_command("g.copy", raster="n,stage_worker_friction")
    path = write_study(tmp_path, input={"friction": "stage_worker_friction"})
    resolved = _resolve_ensemble_in_subprocess(load_batch([str(path)])[0][0].simulations)[0]
    assert isinstance(resolved, ResolvedSimulation)

    gscript.run_command("g.region", n=60)
    assert "GRASS domain changed" in (
        run_validation_worker((resolved,)).get(resolved.simulation_id) or ""
    )
    gscript.run_command("g.region", n=50)
    gscript.run_command("g.remove", flags="f", type="raster", name="stage_worker_friction")
    assert "stage_worker_friction" in (
        run_validation_worker((resolved,)).get(resolved.simulation_id) or ""
    )
    assert not (tmp_path / "results").exists()


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
def test_dry_run_rejects_raster_outside_domain_without_writing(tmp_path: Path) -> None:
    from grass.pygrass import utils as gutils

    gscript.use_temp_region()
    try:
        gscript.run_command("g.region", n=150, s=100, e=150, w=100, res=10)
        gscript.mapcalc("stage_outside=1")
    finally:
        gscript.del_temp_region()
    path = write_study(
        tmp_path,
        input={"friction": "stage_outside"},
        outputs={
            "rasters": {"prefix": "stage_spatial", "variables": ["water_depth"]},
            "statistics": {"file": "stats/results.csv"},
        },
    )
    with pytest.raises(RuntimeError, match="YAML batch validation failed"):
        itzi_run(build_parser().parse_args(["run", str(path), "--dry-run"]))
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "stats").exists()
    assert not gutils.get_mapset_raster("stage_spatial_water_depth_0000", gutils.getenv("MAPSET"))


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
def test_ensemble_output_collision_with_input_rejects_all_members(tmp_path: Path) -> None:
    from grass.pygrass import utils as gutils

    gscript.mapcalc("stage_alias_water_depth_0002=1")
    path = write_study(
        tmp_path,
        input={"rainfall_rate": ["rainfall", "stage_alias_water_depth_0002"]},
        outputs={"rasters": {"prefix": "stage_alias", "variables": ["water_depth"]}},
    )
    with pytest.raises(RuntimeError, match="failed validation or execution"):
        itzi_run(build_parser().parse_args(["run", str(path), "-o"]))
    members = read_manifest(tmp_path)["members"]
    assert len(members) == 2
    assert all(member["status"] == "validation_failed" for member in members)
    assert all(member["failure"]["phase"] == "artifact_validation" for member in members)
    assert not gutils.get_mapset_raster("stage_alias_water_depth_0000", gutils.getenv("MAPSET"))


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
def test_yaml_ensemble_runs_in_spawned_resolver_and_worker(test_data_temp_path):
    config_path = Path(test_data_temp_path) / "source-file-name.yaml"
    config_path.write_text(
        """\
schema_version: 1
ensemble:
  id: basic
grass: {}
time:
  duration: "00:01:00"
  record_step: "00:00:30"
input:
  ground_elevation: "z"
  friction: "n"
  infiltration:
    - type: "constant"
      rate: "infiltration_rate"
parameters:
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
            "parameters.cfl": 0.2,
        },
        {
            "input.infiltration": {"rate": "infiltration_rate", "type": "constant"},
            "parameters.cfl": 0.3,
        },
    ]
    assert all(
        member["artifacts"]["rasters"]["water_depth"].startswith("basic_sim-")
        and member["artifacts"]["rasters"]["water_depth"].endswith("@5by5")
        for member in manifest["members"]
    )


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
def test_explicit_grass_context_runs_two_members_without_parent_session(test_data_temp_path):
    context = gscript.gisenv()
    config_path = Path(test_data_temp_path) / "explicit.yaml"
    config_path.write_text(
        f"""\
schema_version: 1
ensemble:
  id: explicit
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
parameters:
  cfl: 0.2
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
    assert len(manifest["members"]) == 2
    assert all(member["status"] == "completed" for member in manifest["members"])
    mapset_path = (
        Path(context["GISDBASE"]) / context["LOCATION_NAME"] / context["MAPSET"] / "cellhd"
    )
    assert {
        f"{member['artifacts']['rasters']['water_depth'].partition('@')[0]}_0000"
        for member in manifest["members"]
    } <= {path.name for path in mapset_path.iterdir()}


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
def test_dry_run_validates_multiple_members_in_one_session(tmp_path):
    prefix = f"dry_multi_{uuid4().hex[:8]}"
    config_path = write_study(
        tmp_path,
        ensemble={"id": "dry-multi"},
        parameters={"cfl": [0.2, 0.3]},
        outputs={"rasters": {"prefix": f"{prefix}_{{simulation}}", "variables": ["water_depth"]}},
    )

    itzi_run(build_parser().parse_args(["run", str(config_path), "--dry-run"]))

    assert not (tmp_path / "results" / "dry-multi.manifest.yaml").exists()
    assert not gscript.list_strings(type="raster", pattern=f"{prefix}_*")


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
def test_dry_run_rejects_null_elevation_without_creating_outputs(tmp_path):
    gscript.mapcalc("dry_null_dem=null()")
    config_path = write_study(
        tmp_path,
        ensemble={"id": "dry"},
        input={"ground_elevation": "dry_null_dem"},
        parameters={},
        outputs={
            "rasters": {"prefix": "dry_output", "variables": ["water_depth"]},
            "statistics": {"file": "dry-results/statistics.csv"},
        },
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
    assert manifest["members"][0]["failure"]["phase"] == "run_validation"
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
    config_path = write_study(
        tmp_path,
        ensemble={"id": "dry-rain"},
        input={"rainfall_rate": "dry_rain_series"},
        parameters={},
        outputs={"rasters": {"prefix": "dry_rain_output", "variables": ["water_depth"]}},
    )

    itzi_run(build_parser().parse_args(["run", str(config_path), "--dry-run"]))
    assert not (tmp_path / "results" / "dry-rain.manifest.yaml").exists()


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
def test_dry_run_checks_every_generated_output_record(test_data_temp_path):
    gscript.mapcalc("existing_output_water_depth_0002=1")
    config_path = write_study(
        Path(test_data_temp_path),
        ensemble={"id": "existing-output"},
        parameters={},
        outputs={"rasters": {"prefix": "existing_output", "variables": ["water_depth"]}},
    )

    with pytest.raises(RuntimeError, match="YAML batch validation failed"):
        itzi_run(build_parser().parse_args(["run", str(config_path), "--dry-run"]))

    assert not (Path(test_data_temp_path) / "results" / "existing-output.manifest.yaml").exists()
    assert not gscript.find_file(name="existing_output_water_depth_0000", element="cell").get(
        "file"
    )


@pytest.mark.forked
@pytest.mark.usefixtures("grass_xy_session")
def test_run_validation_accepts_unoccupied_drainage_id():
    from itzi.grass.utils import get_current_mapset
    from itzi.ensemble.run_validation import _validate_grass_outputs
    from itzi.grass.interface import GrassInterface

    GrassSessionManager.ensure_temporal_initialized()
    name = f"drainage_collision_{uuid4().hex[:8]}"
    simulation = SimpleNamespace(
        simulation_config=SimpleNamespace(
            start_time=datetime(2020, 1, 1),
            end_time=datetime(2020, 1, 1) + timedelta(minutes=1),
            record_step=timedelta(seconds=30),
            output_map_names={},
            drainage_output=f"{name}@{get_current_mapset()}",
            temporal_type=TemporalType.RELATIVE,
        )
    )
    interface = SimpleNamespace(
        overwrite=False,
        validate_output_stds_temporal_type=lambda *_: None,
    )

    _validate_grass_outputs(cast(ResolvedSimulation, simulation), cast(GrassInterface, interface))
