"""Pure ordered batch loading for YAML streams and deprecated INI files."""

from __future__ import annotations

import hashlib
from pathlib import Path

from itzi_core import TemporalType

import itzi.messenger as msgr
from itzi.configreader import ConfigReader
from itzi.ensemble import (
    DIRECT_INPUT_KEYS,
    GREEN_AMPT_KEYS,
    DocumentFailure,
    DomainConfig,
    EnsembleError,
    ExpandedEnsemble,
    ExpandedSimulation,
    GrassContextConfig,
    NormalizedInfiltration,
    NormalizedTime,
    OutputTemplates,
    SourceDocument,
    check_batch_limits,
    check_unique_ensemble_ids,
    load_yaml_stream,
)


def load_batch(
    config_paths: list[str],
) -> tuple[tuple[ExpandedEnsemble, ...], tuple[DocumentFailure, ...]]:
    """Load an ordered mixed configuration batch without initializing GRASS."""
    ensembles: list[ExpandedEnsemble] = []
    failures: list[DocumentFailure] = []
    for raw_path in config_paths:
        path = Path(raw_path)
        if path.suffix == ".yaml":
            stream = load_yaml_stream(path)
            ensembles.extend(stream.ensembles)
            failures.extend(stream.failures)
        else:
            msgr.warning(f"INI configuration <{path}> is deprecated; use YAML instead.")
            try:
                ensembles.append(_legacy_ensemble(path))
            except Exception as error:  # noqa: BLE001 - preserve document-local continuation.
                failures.append(
                    DocumentFailure(
                        path=path.expanduser().resolve(),
                        document_index=0,
                        line=None,
                        column=None,
                        phase="schema",
                        detail=str(error),
                    )
                )
    check_unique_ensemble_ids(tuple(ensembles))
    check_batch_limits(tuple(ensembles))
    return tuple(ensembles), tuple(failures)


def _legacy_ensemble(path: Path) -> ExpandedEnsemble:
    """Adapt the public one-member INI reader to the scalar batch model."""
    reader = ConfigReader(str(path))
    config = reader.get_sim_params()
    if config.hotstart_config is not None:
        raise EnsembleError("hotstart output requires Stage 2 checkpoint support in mixed batches")
    grass = reader.get_grass_params()
    source_path = path.expanduser().resolve()
    source_bytes = source_path.read_bytes()
    source = SourceDocument(
        path=source_path,
        document_index=0,
        start_line=1,
        file_digest=hashlib.blake2b(source_bytes, digest_size=32).hexdigest(),
        document_digest=hashlib.blake2b(b"legacy-v1\0" + source_bytes, digest_size=32).hexdigest(),
    )
    ensemble_id = f"legacy-{hashlib.blake2b(str(source_path).encode(), digest_size=8).hexdigest()}"
    context = None
    if grass.grassdata is not None:
        context = GrassContextConfig(
            database=grass.grassdata,
            project=grass.location,
            mapset=grass.mapset,
            executable=grass.grass_bin,
        )
    input_names = config.input_map_names
    infiltration_maps = {
        key: input_names[key] for key in ("infiltration", *GREEN_AMPT_KEYS) if key in input_names
    }
    direct_inputs = {key: input_names[key] for key in DIRECT_INPUT_KEYS if key in input_names}
    options = {
        **config.surface_flow_parameters.model_dump(),
        "dtinf": config.dtinf,
    }
    drainage = None
    if config.swmm_inp is not None:
        drainage = {
            "swmm_input": str(config.swmm_inp),
            "orifice_coeff": config.orifice_coeff,
            "free_weir_coeff": config.free_weir_coeff,
            "submerged_weir_coeff": config.submerged_weir_coeff,
        }
    if config.temporal_type == TemporalType.RELATIVE:
        normalized_time = NormalizedTime(
            TemporalType.RELATIVE,
            None,
            None,
            config.end_time - config.start_time,
            config.record_step,
            False,
        )
    else:
        normalized_time = NormalizedTime(
            TemporalType.ABSOLUTE,
            config.start_time,
            config.end_time,
            config.end_time - config.start_time,
            config.record_step,
            False,
        )
    stats_file = reader.get_stats_file()
    if stats_file is not None:
        # The INI deprecation release preserves cwd-relative destinations.
        stats_file = str((Path.cwd() / stats_file).resolve())
    expanded = ExpandedSimulation(
        source=source,
        ensemble_id=ensemble_id,
        ensemble_name=source_path.stem,
        coordinates=(),
        domain=DomainConfig(grass=context, region=grass.region, mask=grass.mask),
        time=normalized_time,
        input_maps=tuple(sorted(direct_inputs.items())),
        infiltration=NormalizedInfiltration(
            config.infiltration_model, tuple(sorted(infiltration_maps.items()))
        ),
        options=tuple(sorted(options.items())),
        drainage=tuple(sorted(drainage.items())) if drainage is not None else None,
        outputs=OutputTemplates(
            raster_prefix=reader.out_prefix,
            raster_variables=tuple(reader.out_values),
            statistics_file=stats_file,
            drainage_dataset=config.drainage_output,
            hotstart_file=None,
            hotstart_interval=None,
        ),
    )
    return ExpandedEnsemble(source, ensemble_id, source_path.stem, None, (expanded,))
