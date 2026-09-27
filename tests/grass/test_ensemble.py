"""GRASS integration coverage for basic spawned YAML ensemble execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import grass.script as gscript
import pytest

from itzi.itzi import itzi_run


@pytest.mark.forked
@pytest.mark.usefixtures("grass_5by5")
def test_yaml_ensemble_runs_in_spawned_resolver_and_worker(test_data_temp_path):
    config_path = Path(test_data_temp_path) / "basic.yaml"
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
options:
  cfl: [0.2, 0.3]
  dtmax: 0.3
outputs:
  rasters:
    prefix: "basic_{simulation}"
    variables: [water_depth]
  manifest:
    file: "basic.manifest.json"
""",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config_file=[str(config_path)],
        o=True,
        v=None,
        q=None,
        resume_from=[],
        member=[],
        dry=False,
    )

    itzi_run(args)

    manifest_path = Path(test_data_temp_path) / "basic.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["members"]) == 2
    assert all(member["status"] == "completed" for member in manifest["members"])
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
    file: "explicit.manifest.json"
""",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config_file=[str(config_path)],
        o=True,
        v=None,
        q=None,
        resume_from=[],
        member=[],
        dry=False,
    )

    gisrc = os.environ.pop("GISRC")
    try:
        itzi_run(args)
    finally:
        os.environ["GISRC"] = gisrc

    manifest_path = Path(test_data_temp_path) / "explicit.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["members"]) == 4
    assert all(member["status"] == "completed" for member in manifest["members"])
    mapset_path = (
        Path(context["GISDBASE"]) / context["LOCATION_NAME"] / context["MAPSET"] / "cellhd"
    )
    assert {
        f"{member['artifacts']['rasters']['water_depth']}_0000" for member in manifest["members"]
    } <= {path.name for path in mapset_path.iterdir()}
