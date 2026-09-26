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
from collections.abc import Callable
from datetime import timedelta
from importlib.metadata import version
from multiprocessing import Process, get_context
from pathlib import Path
from queue import Empty

import itzi.messenger as msgr
from itzi.cli_parser import build_parser
from itzi.configreader import ConfigReader
from itzi.ensemble_models import (
    DocumentFailure,
    EnsembleError,
    ExpandedEnsemble,
    ExpandedSimulation,
    ResolvedSimulation,
    ValidationFailure,
    render_template,
)
from itzi.grass_session import GrassSessionManager
from itzi.messenger import VerbosityLevel
from itzi.resolution import (
    resolve_simulation,
    validate_resolved_ensemble,
    verify_resolved_simulation,
)
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
        sim_params = conf_data.get_sim_params()
        grass_params = conf_data.get_grass_params()
        with GrassSessionManager(grass_params):
            sim_runner = SimulationRunner(
                sim_params,
                grass_params,
                hotstart_path=hotstart_file,
                stats_file=conf_data.get_stats_file(),
            )
            sim_runner.run().finalize()
    except msgr.FatalError:
        raise SystemExit(1) from None
    except SystemExit as error:
        detail = error.code if isinstance(error.code, str) else f"exit status {error.code}"
        msgr.warning(f"Simulation terminated with {detail}")
        raise SystemExit(1) from None
    except Exception as error:
        msgr.warning(f"Error during execution: {type(error).__name__}: {error}")
        raise SystemExit(1) from None


def resolved_sim_runner_worker(simulation: ResolvedSimulation, result_queue) -> None:
    """Run one resolved member and return a small structured worker result."""
    msgr.raise_on_error = True
    msgr._itzi_logger.set_verbosity(msgr.verbosity())
    runner: SimulationRunner | None = None
    try:
        with GrassSessionManager(simulation.grass_params):
            verify_resolved_simulation(simulation)
            runner = SimulationRunner(
                simulation.simulation_config,
                simulation.grass_params,
                stats_file=str(simulation.artifacts.statistics_file)
                if simulation.artifacts.statistics_file is not None
                else None,
                effective_mask=simulation.effective_mask,
                input_kinds=dict(simulation.input_kinds),
                exclusive_stats=True,
            )
            runner.run().finalize()
        result_queue.put(("completed", None))
    except Exception as error:
        if runner is not None:
            try:
                runner.finalize()
            except Exception:
                pass
        result_queue.put(("execution_failed", f"{type(error).__name__}: {error}"))


def resolver_worker(expanded: ExpandedSimulation, result_queue) -> None:
    """Resolve one scalar member in a short-lived spawned GRASS process."""
    msgr.raise_on_error = True
    msgr._itzi_logger.set_verbosity(msgr.verbosity())
    try:
        result_queue.put(("resolved", resolve_simulation(expanded)))
    except Exception as error:
        result_queue.put(("validation_failed", f"{type(error).__name__}: {error}"))


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
        msgr.message("{}: {}".format(f, t))
    msgr.message("Total: {}".format(total_elapsed_time))
    avg_time_s = int(total_elapsed_time.total_seconds() / len(times_list))
    msgr.message("Average: {}".format(timedelta(seconds=avg_time_s)))
    if failed_files:
        msgr.fatal(f"{len(failed_files)} simulation(s) failed")


def _run_ensemble_batch(cli_args) -> None:
    """Resolve and execute an ordered YAML/INI batch through spawn boundaries."""
    if getattr(cli_args, "resume_from", []):
        msgr.fatal("Resume is not available for YAML ensembles until Stage 2")
    ensembles, document_failures = load_batch(cli_args.config_file)
    for failure in document_failures:
        msgr.warning(failure.format())

    resolved: list[
        tuple[ExpandedEnsemble, tuple[ResolvedSimulation, ...], tuple[ValidationFailure, ...]]
    ] = []
    for ensemble in ensembles:
        successful: list[ResolvedSimulation] = []
        failures: list[ValidationFailure] = []
        for expanded in ensemble.simulations:
            result = _resolve_in_spawn(expanded)
            if isinstance(result, ResolvedSimulation):
                successful.append(result)
            else:
                failures.append(result)
        try:
            validate_resolved_ensemble(tuple(successful))
        except EnsembleError as error:
            failures.extend(
                ValidationFailure(
                    coordinates=simulation.coordinates,
                    phase="artifact_validation",
                    detail=str(error),
                )
                for simulation in successful
            )
            successful = []
        resolved.append((ensemble, tuple(successful), tuple(failures)))

    selected = _select_members(resolved, getattr(cli_args, "member", []))
    if getattr(cli_args, "dry", False):
        _display_dry_plan(resolved, selected, document_failures)
        if document_failures or any(failures for _, _, failures in resolved):
            msgr.fatal("YAML batch validation failed")
        return

    failure_count = len(document_failures)
    for ensemble, simulations, failures in resolved:
        selected_ids = selected.get(ensemble.ensemble_id, set())
        if not selected_ids and not (not getattr(cli_args, "member", []) and failures):
            continue
        manifest_path = _manifest_path(ensemble)
        states = _initial_member_states(
            simulations, failures, selected_ids, bool(getattr(cli_args, "member", []))
        )
        try:
            _validate_manifest_destination(manifest_path, ensemble, simulations)
        except EnsembleError as error:
            msgr.warning(str(error))
            failure_count += max(1, len(selected_ids))
            continue
        try:
            _create_manifest(manifest_path, ensemble, states, overwrite=bool(cli_args.o))
        except OSError as error:
            msgr.warning(f"Cannot create manifest <{manifest_path}>: {error}")
            failure_count += max(1, len(selected_ids))
            continue

        for simulation in simulations:
            if simulation.simulation_id not in selected_ids:
                continue
            if states[simulation.simulation_id]["status"] != "planned":
                continue
            if (
                simulation.artifacts.statistics_file is not None
                and simulation.artifacts.statistics_file.exists()
                and not cli_args.o
            ):
                states[simulation.simulation_id]["status"] = "validation_failed"
                states[simulation.simulation_id]["failure"] = {
                    "phase": "artifact_validation",
                    "detail": f"statistics file <{simulation.artifacts.statistics_file}> already exists",
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
            status, detail = _run_resolved_in_spawn(simulation)
            states[simulation.simulation_id]["status"] = status
            states[simulation.simulation_id]["elapsed_seconds"] = time.monotonic() - started
            if detail is not None:
                states[simulation.simulation_id]["failure"] = detail
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

    if failure_count:
        msgr.fatal(f"{failure_count} ensemble simulation(s) failed validation or execution")
    msgr.message("Simulation(s) complete.")


def _resolve_in_spawn(expanded: ExpandedSimulation) -> ResolvedSimulation | ValidationFailure:
    context = get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=resolver_worker, args=(expanded, result_queue))
    process.start()
    process.join()
    try:
        status, payload = result_queue.get(timeout=1)
    except Empty:
        status = "validation_failed"
        payload = f"resolver exited with status {process.exitcode} without returning a result"
    finally:
        result_queue.close()
        process.close()
    if status == "resolved" and isinstance(payload, ResolvedSimulation):
        return payload
    return ValidationFailure(
        coordinates=expanded.coordinates,
        phase="input_resolution",
        detail=str(payload),
    )


def _run_resolved_in_spawn(simulation: ResolvedSimulation) -> tuple[str, str | None]:
    context = get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=resolved_sim_runner_worker, args=(simulation, result_queue))
    process.start()
    process.join()
    try:
        status, detail = result_queue.get(timeout=1)
    except Empty:
        reason = (
            f"signal {-process.exitcode}"
            if process.exitcode and process.exitcode < 0
            else str(process.exitcode)
        )
        status, detail = (
            "execution_failed",
            f"worker exited with {reason} without returning a result",
        )
    finally:
        result_queue.close()
        process.close()
    return status, detail


def _select_members(
    resolved: list[
        tuple[ExpandedEnsemble, tuple[ResolvedSimulation, ...], tuple[ValidationFailure, ...]]
    ],
    selectors: list[str],
) -> dict[str, set[str]]:
    """Resolve member selectors only against successfully resolved members."""
    selected = {ensemble.ensemble_id: set() for ensemble, _, _ in resolved}
    all_members = {
        f"{ensemble.ensemble_id}#{simulation.simulation_id}": simulation.simulation_id
        for ensemble, simulations, _ in resolved
        for simulation in simulations
    }
    if not selectors:
        for ensemble, simulations, _ in resolved:
            selected[ensemble.ensemble_id] = {
                simulation.simulation_id for simulation in simulations
            }
        return selected
    if any("#" not in selector for selector in selectors) and len(resolved) != 1:
        msgr.fatal("Unqualified --member IDs require a batch with exactly one ensemble")
    for selector in selectors:
        qualified = selector if "#" in selector else f"{resolved[0][0].ensemble_id}#{selector}"
        simulation_id = all_members.get(qualified)
        if simulation_id is None:
            msgr.fatal(f"--member {selector!r} does not match a successfully resolved simulation")
        ensemble_id, _separator, _member = qualified.partition("#")
        selected[ensemble_id].add(simulation_id)
    return selected


def _manifest_path(ensemble: ExpandedEnsemble) -> Path:
    if ensemble.manifest_template is not None:
        rendered = render_template(
            ensemble.manifest_template, ensemble=ensemble.ensemble_id, simulation=None
        )
        path = Path(rendered).expanduser()
        return (path if path.is_absolute() else ensemble.source.path.parent / path).resolve()
    return (
        ensemble.source.path.parent
        / f"{ensemble.source.path.stem}.{ensemble.ensemble_id}.manifest.json"
    )


def _validate_manifest_destination(
    manifest_path: Path,
    ensemble: ExpandedEnsemble,
    simulations: tuple[ResolvedSimulation, ...],
) -> None:
    """Keep the parent-owned manifest away from files required by the run."""
    protected = {ensemble.source.path.resolve()}
    for simulation in simulations:
        if simulation.simulation_config.swmm_inp is not None:
            protected.add(simulation.simulation_config.swmm_inp.resolve())
        if simulation.artifacts.statistics_file is not None:
            protected.add(simulation.artifacts.statistics_file.resolve())
    if manifest_path in protected:
        raise EnsembleError(
            f"manifest <{manifest_path}> aliases a protected input or member artifact"
        )
    for path in protected:
        if manifest_path.exists() and path.exists() and manifest_path.samefile(path):
            raise EnsembleError(
                f"manifest <{manifest_path}> aliases a protected input or member artifact"
            )


def _initial_member_states(
    simulations: tuple[ResolvedSimulation, ...],
    failures: tuple[ValidationFailure, ...],
    selected_ids: set[str],
    has_selectors: bool,
) -> dict[str, dict]:
    states: dict[str, dict] = {}
    for simulation in simulations:
        selected = simulation.simulation_id in selected_ids
        states[simulation.simulation_id] = {
            "simulation_id": simulation.simulation_id,
            "coordinates": simulation.coordinates,
            "status": "planned" if selected else "not_selected",
            "selected": selected,
            "artifacts": {
                "rasters": dict(simulation.artifacts.output_map_names),
                "drainage": simulation.artifacts.drainage_output,
                "statistics": str(simulation.artifacts.statistics_file)
                if simulation.artifacts.statistics_file is not None
                else None,
            },
        }
    for index, failure in enumerate(failures):
        key = f"validation-{index}"
        states[key] = {
            "simulation_id": None,
            "coordinates": failure.coordinates,
            "status": "not_selected" if has_selectors else "validation_failed",
            "selected": not has_selectors,
            "failure": {"phase": failure.phase, "detail": failure.detail},
        }
    return states


def _manifest_document(ensemble: ExpandedEnsemble, states: dict[str, dict]) -> dict:
    return {
        "manifest_version": 1,
        "ensemble": {"id": ensemble.ensemble_id, "name": ensemble.ensemble_name},
        "source": {
            "path": str(ensemble.source.path),
            "document_index": ensemble.source.document_index,
            "file_digest": ensemble.source.file_digest,
            "document_digest": ensemble.source.document_digest,
        },
        "members": list(states.values()),
    }


def _create_manifest(
    path: Path, ensemble: ExpandedEnsemble, states: dict[str, dict], *, overwrite: bool
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as file_obj:
        import json

        json.dump(
            _manifest_document(ensemble, states),
            file_obj,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        file_obj.write("\n")


def _update_manifest(path: Path, ensemble: ExpandedEnsemble, states: dict[str, dict]) -> None:
    import json
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as file_obj:
        temporary_path = Path(file_obj.name)
        json.dump(
            _manifest_document(ensemble, states),
            file_obj,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        file_obj.write("\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())
    try:
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _display_dry_plan(
    resolved: list[
        tuple[ExpandedEnsemble, tuple[ResolvedSimulation, ...], tuple[ValidationFailure, ...]]
    ],
    selected: dict[str, set[str]],
    document_failures: tuple[DocumentFailure, ...],
) -> None:
    for ensemble, simulations, failures in resolved:
        msgr.message(f"Ensemble {ensemble.ensemble_id}:")
        for simulation in simulations:
            selection = (
                "selected"
                if simulation.simulation_id in selected[ensemble.ensemble_id]
                else "not selected"
            )
            msgr.message(f"  {simulation.simulation_id} ({selection})")
        for failure in failures:
            msgr.warning(f"  validation failed: {failure.detail}")
    for failure in document_failures:
        msgr.warning(failure.format())


def itzi_version(cli_args):
    """Display the software version number from the installed version"""
    print(version("itzi"))


if __name__ == "__main__":
    sys.exit(main())
