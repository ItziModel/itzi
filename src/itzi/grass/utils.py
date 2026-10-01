"""
Copyright (C) 2026 Laurent Courty

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

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import grass.pygrass.utils as gutils
import grass.script as gscript
import grass.temporal as tgis
import numpy as np
from grass.pygrass.gis import Mapset
from grass.pygrass.gis.region import Region
from itzi_core import INPUT_ARRAY_KEYS, DomainData

import itzi.messenger as msgr
from itzi.ensemble.models import EffectiveMask, EnsembleError
from itzi.grass.session import GrassParams

if TYPE_CHECKING:
    from itzi.grass.interface import GrassInterface

MIN_GRASS_VERSION = (8, 4)


type STDSType = Literal["strds", "stvds"]
type RasterMapType = Literal["CELL", "FCELL", "DCELL"]


class MapData(NamedTuple):
    id: str
    start_time: datetime
    end_time: datetime


def file_exists(name: str) -> bool:
    """Return True if an output exists in the current mapset."""
    if not name:
        return False
    map_id = qualify_output_id(name, get_current_mapset())
    map_name, mapset = split_identifier(map_id)
    return (
        name_is_map(map_id)
        or vector_exists(map_name, mapset)
        or stds_exists(map_id, "strds")
        or stds_exists(map_id, "stvds")
    )


def check_output_files(file_list: list[str]) -> None:
    """Check if the output files exist"""
    if gscript.overwrite() or not file_list:
        return
    for map_name in file_list:
        if file_exists(map_name):
            msgr.fatal(f"File {map_name} exists and will not be overwritten")


def resolve_effective_mask(mask: str | None) -> EffectiveMask:
    """Select the effective mask descriptor without modifying the mapset MASK."""
    current_mapset = gutils.getenv("MAPSET")
    if mask is not None:
        name, requested_mapset = split_identifier(mask)
        mapset = gutils.get_mapset_raster(name, requested_mapset or "")
        if not mapset:
            raise EnsembleError(f"explicit mask {mask!r} was not found")
        return EffectiveMask("explicit", f"{name}@{mapset}")
    if gutils.get_mapset_raster("MASK", current_mapset):
        return EffectiveMask("active", f"MASK@{current_mapset}")
    return EffectiveMask("none", None)


def qualify_output_id(name: str, mapset: str) -> str:
    """Give a new output its destination mapset, never a searched mapset."""
    map_name, requested_mapset = split_identifier(name)
    if requested_mapset and requested_mapset != mapset:
        raise EnsembleError(f"output {name!r} must be in the current mapset {mapset!r}")
    return f"{map_name}@{mapset}"


def output_name(map_id: str) -> str:
    """Return the bare name required by GRASS spatial writers."""
    name, mapset = split_identifier(map_id)
    if mapset != get_current_mapset():
        raise EnsembleError(f"output {map_id!r} must be in the current mapset")
    return name


def get_current_mapset() -> str:
    return gutils.getenv("MAPSET")


def name_is_stds(stds_id: str) -> bool:
    """Return True if the qualified ID identifies a registered STRDS."""
    return bool(tgis.SpaceTimeRasterDataset(stds_id).is_in_db())


def get_crs_wkt() -> str:
    return gscript.read_command("g.proj", flags="fw")


def name_is_map(map_id: str) -> bool:
    """return True if the given name is a map in the grass database
    False if not
    """
    name, mapset = split_identifier(map_id)
    return raster_exists(name, mapset)


def ensure_min_version() -> None:
    full = gscript.parse_command("g.version", flags="g")["version"]
    major, minor = (int(part) for part in full.split(".")[:2])
    if (major, minor) < MIN_GRASS_VERSION:
        required = f"{MIN_GRASS_VERSION[0]}.{MIN_GRASS_VERSION[1]}"
        raise RuntimeError(
            f"itzi requires at least GRASS {required}, found version {major}.{minor}"
        )


def active_grass_params(requested: GrassParams) -> GrassParams:
    """Return the active session context, retaining requested region/mask/bin."""
    return GrassParams(
        grassdata=str(Path(gutils.getenv("GISDBASE")).expanduser().resolve()),
        location=gutils.getenv("LOCATION_NAME"),
        mapset=gutils.getenv("MAPSET"),
        region=requested.region,
        mask=requested.mask,
        grass_bin=requested.grass_bin,
    )


def read_domain(region_id: str | None) -> DomainData:
    """Read the selected computational region without leaving it changed."""
    if gscript.locn_is_latlong():
        raise EnsembleError("latlong locations are not supported")
    using_temp_region = region_id is not None
    if using_temp_region:
        gscript.use_temp_region()
        gscript.run_command("g.region", region=region_id)
    try:
        region = Region()
        if region.cols < 3 or region.rows < 3:
            raise EnsembleError("GRASS Region should be at least 3 cells by 3 cells")
        return DomainData(
            north=region.north,
            south=region.south,
            east=region.east,
            west=region.west,
            rows=region.rows,
            cols=region.cols,
            crs_wkt=get_crs_wkt(),
        )
    finally:
        if using_temp_region:
            gscript.del_temp_region()


def split_identifier(identifier: str) -> tuple[str, str]:
    name, separator, mapset = identifier.partition("@")
    if separator and (not name or not mapset or "@" in mapset):
        raise EnsembleError(f"invalid GRASS identifier {identifier!r}")
    return name, mapset


def resolve_input_identifier(identifier: str) -> tuple[str, Literal["raster", "strds"]]:
    """Resolve both input namespaces and reject a visible ambiguity."""

    name, requested_mapset = split_identifier(identifier)
    raster_mapset = gutils.get_mapset_raster(name, requested_mapset or "")
    raster_id = f"{name}@{raster_mapset}" if raster_mapset else None
    visible_mapsets = [requested_mapset] if requested_mapset else Mapset().visible.read()
    strds_id = None
    for mapset in visible_mapsets:
        candidate = f"{name}@{mapset}"
        try:
            is_strds = tgis.SpaceTimeRasterDataset(candidate).is_in_db()
        except BaseException as error:
            # A visible mapset may have no temporal database or may be
            # read-inaccessible. It cannot contribute a STRDS candidate.
            if isinstance(error, (KeyboardInterrupt, GeneratorExit)):
                raise
            is_strds = False
        if is_strds:
            strds_id = candidate
            break
    if raster_id is not None and strds_id is not None:
        raise EnsembleError(
            f"input {identifier!r} is ambiguous: raster {raster_id} "
            f"and STRDS {strds_id} both exist"
        )
    if raster_id is not None:
        return raster_id, "raster"
    if strds_id is not None:
        return strds_id, "strds"
    raise EnsembleError(f"input {identifier!r} was not found as a raster or STRDS")


def is_clean_name(name: str) -> bool:
    """Return True if name is a valid GRASS map name."""
    return bool(gutils.is_clean_name(name))


def raster_exists(name: str, mapset: str) -> bool:
    return bool(gutils.get_mapset_raster(name, mapset))


def vector_exists(name: str, mapset: str) -> bool:
    """Return True if a vector map exists in the given mapset."""
    return bool(gutils.get_mapset_vector(name, mapset))


def stds_exists(stds_id: str, stds_type: STDSType) -> bool:
    """Return True if a space-time dataset is registered in the database."""
    return bool(tgis.dataset_factory(stds_type, stds_id).is_in_db())


def set_null(map_id: str, threshold: float) -> None:
    """Set null values under a given threshold"""
    gscript.run_command("r.null", flags="f", map=map_id, setnull=f"0.0-{threshold}")


def replace_cell_null_sentinel(raster_type: str, array: np.ndarray) -> np.ndarray:
    """Normalize GRASS CELL nulls to NaN.

    FCELL/DCELL nulls are already exposed as NaN by pygrass. CELL nulls are
    returned as the int32 null sentinel cast to the target dtype.
    """
    if raster_type != "CELL" or not np.issubdtype(array.dtype, np.floating):
        return array

    null_sentinel = array.dtype.type(np.iinfo(np.int32).min)
    array[array == null_sentinel] = np.nan
    return array


def resolve_input_map_lists(
    grass_interface: GrassInterface,
    map_names: Mapping[str, str],
    start_time: datetime,
    end_time: datetime,
    input_kinds: Mapping[str, Literal["raster", "strds"]],
) -> dict[str, list[MapData] | None]:
    """Resolve provider-ready raster lists without creating an output provider."""
    invalid_input_keys = sorted(set(map_names) - INPUT_ARRAY_KEYS)
    if invalid_input_keys:
        raise ValueError(f"Invalid input keys found: {', '.join(invalid_input_keys)}")

    map_lists: dict[str, list[MapData] | None] = {arr_key: None for arr_key in INPUT_ARRAY_KEYS}
    for key, map_name in map_names.items():
        if not map_name:
            continue
        kind = input_kinds[key]
        if kind == "strds":
            if not name_is_stds(map_name):
                msgr.fatal(f"STRDS input <{map_name}> is no longer available")
        elif kind == "raster" and not name_is_map(map_name):
            msgr.fatal(f"raster input <{map_name}> is no longer available")
        if kind == "strds":
            map_list = grass_interface.raster_list_from_strds(map_name)
        else:
            map_list = [MapData(id=map_name, start_time=start_time, end_time=end_time)]
        map_lists[key] = map_list
    return map_lists
