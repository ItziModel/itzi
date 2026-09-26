"""GRASS integration coverage for basic spawned YAML ensemble execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
  duration: "PT1M"
  record_step: "PT30S"
input:
  ground_elevation: "z"
  friction: "n"
options:
  cfl: 0.2
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
    member = manifest["members"][0]
    assert member["status"] == "completed"
    assert member["artifacts"]["rasters"]["water_depth"].startswith("basic_sim-")
