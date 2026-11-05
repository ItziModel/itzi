"""Lossless application checkpoint storage and payload integrity."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from itzi_core import (
    DomainData,
    HotstartData,
    SimulationBuilder,
    SimulationConfig,
    SurfaceFlowParameters,
    TemporalType,
)
from itzi_core.itzi_error import HotstartError

from itzi.hotstart import load_hotstart, write_hotstart


@pytest.fixture(params=[False, True], ids=["surface", "swmm"])
def hotstart(request: pytest.FixtureRequest) -> HotstartData:
    config = SimulationConfig(
        start_time=datetime.min,
        end_time=datetime.min + timedelta(seconds=25),
        record_step=timedelta(seconds=10),
        temporal_type=TemporalType.RELATIVE,
        surface_flow_parameters=SurfaceFlowParameters(),
        input_map_names={},
        output_map_names={},
    )
    domain = DomainData(north=20, south=0, east=30, west=0, rows=2, cols=3, crs_wkt="")
    mask = np.zeros(domain.shape, dtype=bool)
    mask[0, 0] = True
    simulation = SimulationBuilder(config, mask).with_domain_data(domain).build().initialize()
    checkpoint = simulation.capture_hotstart()
    simulation.finalize()
    for array in checkpoint.raster_state.padded_arrays.values():
        array[:] = np.arange(array.size).reshape(array.shape)
        if np.issubdtype(array.dtype, np.floating):
            array[0, 0] = -0.0
            array[1, 1] = np.nan
    if request.param:
        checkpoint = checkpoint.model_copy(update={"swmm_hotstart_bytes": b"SWMM\x00\xffpayload"})
    return checkpoint


@pytest.fixture
def checkpoint_path(tmp_path: Path, hotstart: HotstartData) -> Path:
    path = tmp_path / "checkpoint.zip"
    with path.open("wb") as destination:
        write_hotstart(hotstart, destination)
    return path


def read_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def write_members(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def test_checkpoint_roundtrip_is_lossless(checkpoint_path: Path, hotstart: HotstartData) -> None:
    loaded = load_hotstart(checkpoint_path)
    assert loaded.metadata == hotstart.metadata
    assert loaded.swmm_hotstart_bytes == hotstart.swmm_hotstart_bytes
    np.testing.assert_array_equal(loaded.raster_state.mask, hotstart.raster_state.mask)
    assert loaded.raster_state.mask.dtype == hotstart.raster_state.mask.dtype
    assert loaded.raster_state.padded_arrays.keys() == hotstart.raster_state.padded_arrays.keys()
    for key, expected in hotstart.raster_state.padded_arrays.items():
        actual = loaded.raster_state.padded_arrays[key]
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes(), key
        assert not np.shares_memory(actual, expected)


@pytest.mark.parametrize("version", [1, 2, 4])
def test_unsupported_archives_explain_how_to_resume(tmp_path: Path, version: int) -> None:
    path = tmp_path / "old.zip"
    key = "hotstart_version" if version < 3 else "archive_version"
    write_members(
        path, {"metadata.json": json.dumps({key: version}).encode(), "raster_state.npz": b""}
    )
    with pytest.raises(HotstartError, match="Unsupported.*earlier Itzi release"):
        load_hotstart(path)


@pytest.mark.parametrize("member", ["metadata.json", "raster_state.npz", "swmm_hotstart.hsf"])
def test_missing_members_are_rejected(checkpoint_path: Path, member: str) -> None:
    members = read_members(checkpoint_path)
    if member not in members:
        pytest.skip("Surface checkpoint has no SWMM payload")
    del members[member]
    write_members(checkpoint_path, members)
    with pytest.raises(HotstartError, match="missing required members|do not match"):
        load_hotstart(checkpoint_path)


def test_unexpected_members_are_rejected(checkpoint_path: Path) -> None:
    with zipfile.ZipFile(checkpoint_path, "a") as archive:
        archive.writestr("unexpected", b"data")
    with pytest.raises(HotstartError, match="do not match"):
        load_hotstart(checkpoint_path)


def test_duplicate_members_are_rejected(checkpoint_path: Path) -> None:
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(checkpoint_path, "a") as archive:
            archive.writestr("metadata.json", b"{}")
    with pytest.raises(HotstartError, match="duplicate members"):
        load_hotstart(checkpoint_path)


@pytest.mark.parametrize("member", ["raster_state.npz", "swmm_hotstart.hsf"])
def test_payload_hashes_are_checked(checkpoint_path: Path, member: str) -> None:
    members = read_members(checkpoint_path)
    if member not in members:
        pytest.skip("Surface checkpoint has no SWMM payload")
    members[member] += b"corruption"
    write_members(checkpoint_path, members)
    with pytest.raises(HotstartError, match="hash mismatch"):
        load_hotstart(checkpoint_path)


@pytest.mark.parametrize("payload", [b"not a ZIP", b"PK\x03\x04truncated"])
def test_invalid_outer_archive_is_rejected(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "invalid.zip"
    path.write_bytes(payload)
    with pytest.raises(HotstartError, match="Failed to read hotstart archive"):
        load_hotstart(path)


@pytest.mark.parametrize(
    "compression_name, damage_offset", [("ZIP_LZMA", 9), ("ZIP_ZSTANDARD", 0)]
)
def test_corrupt_compressed_member_is_reported_as_hotstart_error(
    checkpoint_path: Path, compression_name: str, damage_offset: int
) -> None:
    compression = getattr(zipfile, compression_name, None)
    if compression is None:
        pytest.skip(f"{compression_name} is unavailable on this Python version")
    members = read_members(checkpoint_path)
    with zipfile.ZipFile(checkpoint_path, "w", compression=compression) as archive:
        archive.writestr("metadata.json", members.pop("metadata.json"))
        for name, payload in members.items():
            archive.writestr(name, payload)
    load_hotstart(checkpoint_path)

    damaged = bytearray(checkpoint_path.read_bytes())
    name_length, extra_length = struct.unpack_from("<HH", damaged, 26)
    # Corrupt compressed metadata after the ZIP local header and codec properties.
    damaged[30 + name_length + extra_length + damage_offset] ^= 0xFF
    checkpoint_path.write_bytes(damaged)
    with pytest.raises(HotstartError, match="Failed to read hotstart archive"):
        load_hotstart(checkpoint_path)


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(HotstartError, match="missing.zip"):
        load_hotstart(tmp_path / "missing.zip")


@pytest.mark.parametrize("metadata", [b"invalid JSON", b"[]", b"\xff"])
def test_invalid_metadata_is_rejected(checkpoint_path: Path, metadata: bytes) -> None:
    members = read_members(checkpoint_path)
    members["metadata.json"] = metadata
    write_members(checkpoint_path, members)
    with pytest.raises(HotstartError, match="Failed to read|JSON object"):
        load_hotstart(checkpoint_path)


@pytest.mark.parametrize("kind", ["invalid", "npy", "pickle", "missing-mask", "duplicate-array"])
def test_invalid_raster_payload_is_rejected(checkpoint_path: Path, kind: str) -> None:
    with io.BytesIO() as buffer:
        if kind == "invalid":
            buffer.write(b"not an NPZ")
        elif kind == "npy":
            np.save(buffer, np.zeros((2, 3)))
        elif kind == "pickle":
            np.savez(buffer, mask=np.array([{"unsafe": True}], dtype=object))
        elif kind == "missing-mask":
            np.savez(buffer, water_depth=np.zeros((4, 5)))
        else:
            np.savez(buffer, mask=np.zeros((2, 3), dtype=bool))
            with pytest.warns(UserWarning, match="Duplicate name"):
                with zipfile.ZipFile(buffer, "a") as npz:
                    npz.writestr("mask.npy", npz.read("mask.npy"))
        payload = buffer.getvalue()
    members = read_members(checkpoint_path)
    members["raster_state.npz"] = payload
    metadata = json.loads(members["metadata.json"])
    metadata["raster_hash"] = hashlib.blake2b(payload).hexdigest()
    members["metadata.json"] = json.dumps(metadata).encode()
    write_members(checkpoint_path, members)
    with pytest.raises(HotstartError, match="Failed to read|NPZ archive|duplicate arrays"):
        load_hotstart(checkpoint_path)
