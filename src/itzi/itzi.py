"""
NAME:      Itzï

AUTHOR(S): Laurent Courty

PURPOSE:   A distributed GIS computer model for flood simulation. See:
           Courty, L. G., Pedrozo-Acuña, A., & Bates, P. D. (2017).
           Itzï (version 17.1): an open-source,
           distributed GIS model for dynamic flood simulation.
           Geoscientific Model Development, 10(4), 1835–1847.
           https://doi.org/10.5194/gmd-10-1835-2017

COPYRIGHT: (C) 2015-2025 by Laurent Courty

            This program is free software; you can redistribute it and/or
            modify it under the terms of the GNU General Public License
            as published by the Free Software Foundation; either version 2
            of the License, or (at your option) any later version.

            This program is distributed in the hope that it will be useful,
            but WITHOUT ANY WARRANTY; without even the implied warranty of
            MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
            GNU General Public License for more details.
"""

from __future__ import annotations

import os
import sys
import time
from argparse import Namespace
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from importlib.metadata import version
from multiprocessing import Process, get_context
from pathlib import Path

import itzi.messenger as msgr
from itzi.cli_parser import build_parser
from itzi.configreader import ConfigReader
from itzi.ensemble.models import (
    EnsembleError,
    ExpandedSimulation,
    ResolvedEnsemble,
    ResolvedSimulation,
    ValidationFailure,
)
from itzi.ensemble.resolution import resolve_ensemble, validate_resolved_ensemble
from itzi.grass.session import GrassSessionManager
from itzi.manifest import (
    _create_manifest,
    _initial_member_states,
    _manifest_path,
    _update_manifest,
    _validate_manifest_destination,
)
from itzi.messenger import VerbosityLevel
from itzi.run_plan import load_batch
from itzi.simulation_runner import SimulationRunner


def main(argv: list[str] | None = None) -> int:
    """argv: alternative CLI arguments, used for testing (default to sys.argv)"""
    args = build_parser().parse_args(argv)

    command_mapper: dict[str, Callable] = {
        "run": itzi_run,
        "version": itzi_version,
    }

    try:
        # args.command is the name of the subcommand
        command_mapper[args.command](args)
    except msgr.FatalError:
        return 1
    return 0


def sim_runner_worker(conf_file: str, hotstart_file: str | None) -> None:
    """Run one simulation"""
    msgr.raise_on_error = True
    msgr._itzi_logger.set_verbosity(msgr.verbosity())
    try:
        # Run the simulation
        msgr.message(f"Starting simulation of {os.path.basename(conf_file)}...")
        conf_data = ConfigReader(conf_file)
        sim_params = conf_data.sim_config
        grass_params = conf_data.grass_params
        with GrassSessionManager(grass_params):
            sim_runner = SimulationRunner(
                sim_params,
                grass_params,
                hotstart_path=hotstart_file,
                stats_file=conf_data.stats_file,
            )
            sim_runner.initialize().run().finalize()
    except msgr.FatalError:
        raise SystemExit(1) from None
    except SystemExit as error:
        detail = error.code if isinstance(error.code, str) else f"exit status {error.code}"
        msgr.warning(f"Simulation terminated with {detail}")
        raise SystemExit(1) from None
    except Exception as error:
        msgr.warning(f"Error during execution: {type(error).__name__}: {error}")
        raise SystemExit(1) from None


def resolved_sim_runner_worker(simulation: ResolvedSimulation) -> tuple[str, str | None]:
    """Run one resolved member and return its status."""
    msgr.raise_on_error = True
    msgr._itzi_logger.set_verbosity(msgr.verbosity())
    runner: SimulationRunner | None = None
    try:
        with GrassSessionManager(simulation.grass_params):
            runner = SimulationRunner(
                simulation.simulation_config,
                simulation.grass_params,
                stats_file=str(simulation.artifacts.statistics_file)
                if simulation.artifacts.statistics_file is not None
                else None,
                effective_mask=simulation.effective_mask,
                input_kinds=dict(simulation.input_kinds),
            )
            runner.initialize().run().finalize()
        return "completed", None
    except Exception as error:
        if runner is not None:
            try:
                runner.finalize()
            except Exception:
                pass
        return "execution_failed", f"{type(error).__name__}: {error}"


def resolver_worker(
    expanded: tuple[ExpandedSimulation, ...],
) -> tuple[ResolvedSimulation | ValidationFailure, ...]:
    """Resolve one ensemble in a short-lived spawned GRASS process."""
    msgr.raise_on_error = True
    msgr._itzi_logger.set_verbosity(msgr.verbosity())
    return resolve_ensemble(expanded)


def preflight_worker(simulations: tuple[ResolvedSimulation, ...]) -> dict[str, str]:
    """Preflight selected members in one GRASS session, retaining individual failures."""
    msgr.raise_on_error = True
    msgr._itzi_logger.set_verbosity(msgr.verbosity())
    failures: dict[str, str] = {}
    with GrassSessionManager(simulations[0].grass_params):
        from itzi.ensemble.preflight import preflight_simulation

        for simulation in simulations:
            try:
                preflight_simulation(simulation)
            except Exception as error:
                failures[simulation.simulation_id] = f"{type(error).__name__}: {error}"
    return failures


def itzi_run_one(conf_file: str, hotstart_file: str | None) -> bool:
    """Run a simulation in a subprocess"""
    worker_args = (conf_file, hotstart_file)
    p = Process(target=sim_runner_worker, args=worker_args)
    p.start()
    p.join()
    exitcode = p.exitcode
    if exitcode is None:
        msgr.warning(f"Execution of {conf_file} did not report an exit status")
        return False
    p.close()
    if exitcode == 0:
        return True

    reason = f"signal {-exitcode}" if exitcode < 0 else f"exit status {exitcode}"
    msgr.warning(f"Execution of {conf_file} ended with an error ({reason})")
    return False


def reconcile_hotstart_commands(
    config_file_list: list[str],
    resume_from_list: list[tuple[str | None, str]],
) -> list[tuple[str, str | None]]:
    """Output a list of tuples in the form (config_file, hotstart_file)."""

    # No hotstart requested: run every config from scratch.
    if not resume_from_list:
        return [(config_file, None) for config_file in config_file_list]

    # An unnamed single hotstart only makes sense when there is a single config.
    if len(resume_from_list) == 1:
        config_key, hotstart_file = resume_from_list[0]
        if config_key is None:
            if len(config_file_list) != 1:
                msgr.fatal(
                    "A single unnamed --resume-from value can only be used with a single config file"
                )
            return [(config_file_list[0], hotstart_file)]

    # In batch mode every hotstart must be explicitly mapped.
    if any(config_key is None for config_key, _ in resume_from_list):
        msgr.fatal("When using multiple --resume-from values, each one must be CONFIG=PATH")

    # Accept a normalized config path first, then a unique basename like "a.ini".
    exact_lookup = {
        os.path.abspath(os.path.normpath(config_file)): config_file
        for config_file in config_file_list
    }
    basename_lookup: dict[str, str | None] = {}
    resolved_hotstarts = {config_file: None for config_file in config_file_list}

    # Build basename lookup; Store None when a basename is shared.
    for config_file in config_file_list:
        basename = os.path.basename(config_file)
        if basename not in basename_lookup:
            basename_lookup[basename] = config_file
        elif basename_lookup[basename] != config_file:
            basename_lookup[basename] = None

    # Resolve each CONFIG=PATH pair onto its target config file.
    for config_key, hotstart_file in resume_from_list:
        assert config_key is not None
        normalized_key = os.path.abspath(os.path.normpath(config_key))
        target_config = exact_lookup.get(normalized_key)
        if target_config is None:
            basename = os.path.basename(config_key)
            if basename not in basename_lookup:
                msgr.fatal(f"--resume-from config {config_key!r} does not match any config file")
            target_config = basename_lookup[basename]
            if target_config is None:
                msgr.fatal(f"--resume-from config {config_key!r} is ambiguous")

        if resolved_hotstarts[target_config] is not None:
            msgr.fatal(f"Multiple hotstart files were given for {target_config!r}")
        resolved_hotstarts[target_config] = hotstart_file

    return [(config_file, resolved_hotstarts[config_file]) for config_file in config_file_list]


def itzi_run(cli_args):
    """Run one or multiple simulations from the command line."""
    # set environment variables
    if cli_args.o:
        os.environ["GRASS_OVERWRITE"] = "1"
    else:
        os.environ["GRASS_OVERWRITE"] = "0"
    # verbosity
    if cli_args.q and cli_args.q == 2:
        os.environ["ITZI_VERBOSE"] = str(VerbosityLevel.SUPER_QUIET)
    elif cli_args.q == 1:
        os.environ["ITZI_VERBOSE"] = str(VerbosityLevel.QUIET)
    elif cli_args.v == 1:
        os.environ["ITZI_VERBOSE"] = str(VerbosityLevel.VERBOSE)
    elif cli_args.v and cli_args.v >= 2:
        os.environ["ITZI_VERBOSE"] = str(VerbosityLevel.DEBUG)
    else:
        os.environ["ITZI_VERBOSE"] = str(VerbosityLevel.MESSAGE)

    # setting GRASS verbosity (especially for maps registration)
    if cli_args.q and cli_args.q >= 1:
        # no warnings
        os.environ["GRASS_VERBOSE"] = "-1"
    elif cli_args.v and cli_args.v >= 1:
        # normal
        os.environ["GRASS_VERBOSE"] = "2"
    else:
        # only warnings
        os.environ["GRASS_VERBOSE"] = "0"

    if _uses_legacy_execution_path(cli_args):
        _run_legacy_batch(cli_args)
        return
    _run_ensemble_batch(cli_args)


def _uses_legacy_execution_path(cli_args) -> bool:
    """Keep the one-simulation INI execution API during its deprecation window."""
    return (
        all(Path(path).suffix != ".yaml" for path in cli_args.config_file)
        and not getattr(cli_args, "member", [])
        and not getattr(cli_args, "dry", False)
    )


def _run_legacy_batch(cli_args) -> None:
    """Execute the retained pre-ensemble INI path."""
    for config_file in cli_args.config_file:
        msgr.warning(f"INI configuration <{config_file}> is deprecated; use YAML instead.")
    # start total time counter
    total_sim_start = time.time()
    # dictionary to store computation times
    times_list = []
    failed_files = []
    run_commands = reconcile_hotstart_commands(
        cli_args.config_file,
        getattr(cli_args, "resume_from", []),
    )
    for conf_file, hotstart_file in run_commands:
        sim_start = time.time()
        # Run the simulation
        if not itzi_run_one(conf_file, hotstart_file):
            failed_files.append(conf_file)
        # store computational time
        comp_time = timedelta(seconds=int(time.time() - sim_start))
        list_elem = (os.path.basename(conf_file), comp_time)
        times_list.append(list_elem)

    # stop total time counter
    total_elapsed_time = timedelta(seconds=int(time.time() - total_sim_start))
    # display total computation duration
    if failed_files:
        msgr.message("Simulation run(s) finished with errors. Elapsed times:")
    else:
        msgr.message("Simulation(s) complete. Elapsed times:")
    for f, t in times_list:
        msgr.message(f"{f}: {t}")
    msgr.message(f"Total: {total_elapsed_time}")
    avg_time_s = int(total_elapsed_time.total_seconds() / len(times_list))
    msgr.message(f"Average: {timedelta(seconds=avg_time_s)}")
    if failed_files:
        msgr.fatal(f"{len(failed_files)} simulation(s) failed")


def _run_ensemble_batch(cli_args: Namespace) -> None:
    """Resolve and execute each ensemble in order through spawn boundaries."""
    if getattr(cli_args, "resume_from", []):
        msgr.fatal("Resume is not available for YAML ensembles.")
    ensembles, document_failures = load_batch(cli_args.config_file)
    for failure in document_failures:
        msgr.warning(failure.format())

    selectors = getattr(cli_args, "member", [])
    has_selectors = bool(selectors)
    if has_selectors and len(ensembles) != 1 and any("#" not in s for s in selectors):
        msgr.fatal("Unqualified --member IDs require a batch with exactly one ensemble")
    qualified_selectors = [
        (selector, selector if "#" in selector else f"{ensembles[0].ensemble_id}#{selector}")
        for selector in selectors
    ]
    ensemble_ids = {ensemble.ensemble_id for ensemble in ensembles}
    for selector, qualified in qualified_selectors:
        if qualified.partition("#")[0] not in ensemble_ids:
            msgr.fatal(f"--member {selector!r} does not match a successfully resolved simulation")

    dry = getattr(cli_args, "dry", False)
    dry_failed = bool(document_failures)
    failure_count = len(document_failures)
    for ensemble in ensembles:
        successful: list[ResolvedSimulation] = []
        member_failures: list[ValidationFailure] = []
        for result in _resolve_ensemble_in_subprocess(ensemble.simulations):
            if isinstance(result, ResolvedSimulation):
                successful.append(result)
            else:
                member_failures.append(result)
        artifact_failure = None
        try:
            validate_resolved_ensemble(tuple(successful), (ensemble.source.path,))
        except EnsembleError as error:
            artifact_failure = str(error)
        resolved = ResolvedEnsemble(
            ensemble, tuple(successful), tuple(member_failures), artifact_failure
        )
        selected_ids = _select_members(resolved, qualified_selectors)
        preflight_failures, manifest_failure = _preflight_ensemble(
            resolved,
            selected_ids,
            has_selectors=has_selectors,
            overwrite=bool(getattr(cli_args, "o", False)),
        )
        if dry:
            _display_dry_plan(
                resolved,
                selected_ids,
                preflight_failures,
                manifest_failure,
                has_selectors=has_selectors,
            )
            dry_failed |= bool(
                (resolved.failures and not has_selectors) or preflight_failures or manifest_failure
            )
            continue

        failure_count += _run_one_ensemble(
            resolved,
            selected_ids,
            preflight_failures,
            manifest_failure,
            has_selectors=has_selectors,
            overwrite=bool(cli_args.o),
        )

    if dry:
        if dry_failed:
            msgr.fatal("YAML batch validation failed")
        return

    if failure_count:
        msgr.fatal(f"{failure_count} ensemble simulation(s) failed validation or execution")
    msgr.message("Simulation(s) complete.")


def _run_one_ensemble(
    resolved: ResolvedEnsemble,
    selected_ids: set[str],
    preflight_failures: dict[str, dict[str, str]],
    manifest_failure: str | None,
    *,
    has_selectors: bool,
    overwrite: bool,
) -> int:
    """Run selected members and return the number of failures in this ensemble."""
    ensemble = resolved.ensemble
    if not selected_ids and not (not has_selectors and resolved.failures):
        return 0
    manifest_path = _manifest_path(ensemble)
    states = _initial_member_states(
        resolved.simulations, resolved.failures, selected_ids, has_selectors
    )
    for simulation_id, failure in preflight_failures.items():
        states[simulation_id]["status"] = "validation_failed"
        states[simulation_id]["failure"] = failure
    if manifest_failure is not None:
        msgr.warning(manifest_failure)
        return max(1, len(selected_ids))
    try:
        _create_manifest(manifest_path, ensemble, states, overwrite=overwrite)
    except OSError as error:
        msgr.warning(f"Cannot create manifest <{manifest_path}>: {error}")
        return max(1, len(selected_ids))

    failure_count = 0
    member_started = False
    for simulation in resolved.simulations:
        if simulation.simulation_id not in selected_ids:
            continue
        if states[simulation.simulation_id]["status"] != "planned":
            continue
        if member_started:
            detail = _preflight_in_subprocess((simulation,)).get(simulation.simulation_id)
            if detail is not None:
                states[simulation.simulation_id]["status"] = "validation_failed"
                states[simulation.simulation_id]["failure"] = {
                    "phase": "preflight",
                    "detail": detail,
                }
                _update_manifest(manifest_path, ensemble, states)
                continue
        states[simulation.simulation_id]["status"] = "running"
        try:
            _update_manifest(manifest_path, ensemble, states)
        except OSError as error:
            msgr.warning(f"Cannot update manifest <{manifest_path}>: {error}")
            failure_count += 1
            break
        started = time.monotonic()
        member_started = True
        status, detail = _run_simulation_in_subprocess(simulation)
        state = states[simulation.simulation_id]
        state["status"] = status
        state["elapsed_seconds"] = time.monotonic() - started
        if detail is not None:
            states[simulation.simulation_id]["failure"] = {
                "phase": "execution",
                "detail": detail,
            }
            failure_count += 1
        try:
            _update_manifest(manifest_path, ensemble, states)
        except OSError as error:
            msgr.warning(f"Cannot update manifest <{manifest_path}>: {error}")
            failure_count += 1
            break
    failure_count += sum(
        1
        for state in states.values()
        if state["status"] == "validation_failed" and state.get("selected", False)
    )
    return failure_count


def _resolve_ensemble_in_subprocess(
    expanded: tuple[ExpandedSimulation, ...],
) -> tuple[ResolvedSimulation | ValidationFailure, ...]:
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as executor:
            return executor.submit(resolver_worker, expanded).result()
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
    return tuple(
        ValidationFailure(
            coordinates=simulation.coordinates,
            phase="input_resolution",
            detail=detail,
        )
        for simulation in expanded
    )


def _run_simulation_in_subprocess(simulation: ResolvedSimulation) -> tuple[str, str | None]:
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as executor:
            return executor.submit(resolved_sim_runner_worker, simulation).result()
    except Exception as error:
        return "execution_failed", f"{type(error).__name__}: {error}"


def _preflight_in_subprocess(
    simulations: tuple[ResolvedSimulation, ...],
) -> dict[str, str]:
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as executor:
            return executor.submit(preflight_worker, simulations).result()
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        return {simulation.simulation_id: detail for simulation in simulations}


def _preflight_ensemble(
    resolved: ResolvedEnsemble,
    selected_ids: set[str],
    *,
    has_selectors: bool,
    overwrite: bool,
) -> tuple[dict[str, dict[str, str]], str | None]:
    """Collect one ensemble's selected-member and manifest failures."""
    if not selected_ids and has_selectors:
        return {}, None
    ensemble = resolved.ensemble
    simulations = resolved.simulations
    failures: dict[str, dict[str, str]] = {}
    if resolved.artifact_failure is not None:
        for simulation_id in selected_ids:
            failures[simulation_id] = {
                "phase": "artifact_validation",
                "detail": resolved.artifact_failure,
            }
    manifest_failure = None
    try:
        _validate_manifest_destination(
            _manifest_path(ensemble), ensemble, simulations, overwrite=overwrite
        )
    except EnsembleError as error:
        manifest_failure = str(error)
    to_preflight = tuple(
        simulation
        for simulation in simulations
        if simulation.simulation_id in selected_ids and simulation.simulation_id not in failures
    )
    if to_preflight:
        for simulation_id, detail in _preflight_in_subprocess(to_preflight).items():
            failures[simulation_id] = {"phase": "preflight", "detail": detail}
    return failures, manifest_failure


def _select_members(
    resolved: ResolvedEnsemble,
    selectors: list[tuple[str, str]],
) -> set[str]:
    """Select successfully resolved members of one ensemble."""
    all_members = {simulation.simulation_id for simulation in resolved.simulations}
    if not selectors:
        return all_members
    selected: set[str] = set()
    for selector, qualified in selectors:
        ensemble_id, _, simulation_id = qualified.partition("#")
        if ensemble_id != resolved.ensemble.ensemble_id:
            continue
        if simulation_id not in all_members:
            msgr.fatal(f"--member {selector!r} does not match a successfully resolved simulation")
        selected.add(simulation_id)
    return selected


def _display_dry_plan(
    resolved: ResolvedEnsemble,
    selected_ids: set[str],
    preflight_failures: dict[str, dict[str, str]],
    manifest_failure: str | None,
    *,
    has_selectors: bool,
) -> None:
    ensemble = resolved.ensemble
    msgr.message(f"Ensemble {ensemble.ensemble_id}:")
    for simulation in resolved.simulations:
        selection = "selected" if simulation.simulation_id in selected_ids else "not selected"
        msgr.message(f"  {simulation.simulation_id} ({selection})")
        if simulation.simulation_id in preflight_failures:
            failure = preflight_failures[simulation.simulation_id]
            msgr.warning(f"    {failure['phase']} failed: {failure['detail']}")
        for variable, name in simulation.artifacts.output_map_names:
            msgr.message(f"    raster {variable}: {name}")
        if simulation.artifacts.drainage_output is not None:
            msgr.message(f"    drainage: {simulation.artifacts.drainage_output}")
        if simulation.artifacts.statistics_file is not None:
            msgr.message(f"    statistics: {simulation.artifacts.statistics_file}")
    for failure in resolved.failures:
        status = "not selected" if has_selectors else "validation failed"
        msgr.warning(f"  {status}: {failure.detail}")
    if manifest_failure is not None:
        msgr.warning(f"  manifest validation failed: {manifest_failure}")


def itzi_version(cli_args):
    """Display the software version number from the installed version"""
    print(version("itzi"))


if __name__ == "__main__":
    sys.exit(main())
