"""Versioned ZIP persistence for core checkpoints."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import IO, Literal

import numpy as np
from itzi_core import HotstartData, HotstartMetadata, RasterHotstartState
from itzi_core.itzi_error import HotstartError
from numpy.lib.npyio import NpzFile
from pydantic import BaseModel, ConfigDict

METADATA_FILENAME = "metadata.json"
RASTER_STATE_FILENAME = "raster_state.npz"
SWMM_HOTSTART_FILENAME = "swmm_hotstart.hsf"
ARCHIVE_VERSION = 3


class HotstartArchiveMetadata(BaseModel):
    """Storage version and integrity information around the core's logical state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    archive_version: Literal[3] = ARCHIVE_VERSION
    metadata: HotstartMetadata
    raster_hash: str
    swmm_hash: str | None


def write_hotstart(hotstart: HotstartData, destination: IO[bytes]) -> None:
    """Write a checkpoint; the caller owns atomic replacement of its destination."""
    hotstart = HotstartData.model_validate(hotstart)
    with io.BytesIO() as raster_buffer:
        np.savez(
            raster_buffer,
            allow_pickle=False,
            mask=hotstart.raster_state.mask,
            **hotstart.raster_state.padded_arrays,
        )
        swmm_bytes = hotstart.swmm_hotstart_bytes
        envelope = HotstartArchiveMetadata(
            metadata=hotstart.metadata,
            raster_hash=hashlib.blake2b(raster_buffer.getbuffer()).hexdigest(),
            swmm_hash=hashlib.blake2b(swmm_bytes).hexdigest() if swmm_bytes is not None else None,
        )
        with zipfile.ZipFile(
            destination, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=8
        ) as archive:
            archive.writestr(RASTER_STATE_FILENAME, raster_buffer.getbuffer())
            if swmm_bytes is not None:
                archive.writestr(SWMM_HOTSTART_FILENAME, swmm_bytes)
            archive.writestr(
                METADATA_FILENAME, envelope.model_dump_json(indent=2, exclude_computed_fields=True)
            )


def load_hotstart(path: str | Path) -> HotstartData:
    """Read and validate a checkpoint's storage before core compatibility checks."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            members = set(names)
            if len(names) != len(members):
                raise HotstartError("Hotstart archive contains duplicate members")
            required = {METADATA_FILENAME, RASTER_STATE_FILENAME}
            missing = required - members
            if missing:
                raise HotstartError(
                    f"Hotstart archive missing required members: {', '.join(sorted(missing))}"
                )
            raw_metadata = json.loads(archive.read(METADATA_FILENAME))
            if not isinstance(raw_metadata, dict):
                raise HotstartError("Hotstart metadata must be a JSON object")
            archive_version = raw_metadata.get(
                "archive_version", raw_metadata.get("hotstart_version")
            )
            if archive_version != ARCHIVE_VERSION:
                raise HotstartError(
                    f"Unsupported hotstart archive version {archive_version!r}; "
                    f"this release reads only version {ARCHIVE_VERSION}. "
                    "Resume older checkpoints with the earlier Itzi release."
                )
            envelope = HotstartArchiveMetadata.model_validate(raw_metadata)
            expected = required | (
                {SWMM_HOTSTART_FILENAME} if envelope.swmm_hash is not None else set()
            )
            if members != expected:
                raise HotstartError("Hotstart archive members do not match its metadata")

            raster_bytes = archive.read(RASTER_STATE_FILENAME)
            if hashlib.blake2b(raster_bytes).hexdigest() != envelope.raster_hash:
                raise HotstartError("Hotstart raster state hash mismatch")
            swmm_bytes = None
            if envelope.swmm_hash is not None:
                swmm_bytes = archive.read(SWMM_HOTSTART_FILENAME)
                if hashlib.blake2b(swmm_bytes).hexdigest() != envelope.swmm_hash:
                    raise HotstartError("Hotstart SWMM state hash mismatch")

        with io.BytesIO(raster_bytes) as raster_buffer:
            npz = np.load(raster_buffer, allow_pickle=False)
            if not isinstance(npz, NpzFile):
                raise HotstartError("Hotstart raster state must be an NPZ archive")
            with npz:
                if len(npz.files) != len(set(npz.files)):
                    raise HotstartError("Hotstart raster state contains duplicate arrays")
                raster_state = RasterHotstartState(
                    mask=npz["mask"],
                    padded_arrays={key: npz[key] for key in npz.files if key != "mask"},
                )
        return HotstartData(
            metadata=envelope.metadata,
            raster_state=raster_state,
            swmm_hotstart_bytes=swmm_bytes,
        )
    except HotstartError:
        raise
    except Exception as error:
        # ZIP codecs expose different exception types across supported Python versions.
        raise HotstartError(f"Failed to read hotstart archive <{path}>: {error}") from error
