"""Unit tests for HotstartWriter and HotstartLoader classes.

Tests for the hotstart archive format and validation logic.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from itzi.hotstart import (
    HotstartWriter,
    HotstartLoader,
    HOTSTART_VERSION,
    METADATA_FILENAME,
    RASTER_STATE_FILENAME,
    SWMM_HOTSTART_FILENAME,
)
from itzi.data_containers import (
    HotstartSimulationState,
    SimulationConfig,
    SurfaceFlowParameters,
)
from itzi.providers.domain_data import DomainData
from itzi.itzi_error import HotstartError
from itzi.const import TemporalType


@pytest.fixture
def minimal_domain_data() -> DomainData:
    """Create minimal DomainData for testing."""
    return DomainData(
        north=50.0,
        south=0.0,
        east=50.0,
        west=0.0,
        rows=5,
        cols=5,
        crs_wkt="",
    )


@pytest.fixture
def minimal_simulation_config() -> SimulationConfig:
    """Create minimal SimulationConfig for testing."""
    return SimulationConfig(
        start_time=datetime(2000, 1, 1, 0, 0, 0),
        end_time=datetime(2000, 1, 1, 0, 1, 0),
        record_step=timedelta(seconds=30),
        temporal_type=TemporalType.RELATIVE,
        input_map_names={},
        output_map_names={},
        surface_flow_parameters=SurfaceFlowParameters(),
    )


@pytest.fixture
def minimal_simulation_state() -> HotstartSimulationState:
    """Create minimal HotstartSimulationState for testing."""
    return HotstartSimulationState(
        sim_time="2000-01-01T00:00:30",
        dt=0.5,
        next_ts={},
        time_steps_counters={},
        accum_update_time={},
        old_domain_volume=100.0,
    )


@pytest.fixture
def raster_state_bytes() -> bytes:
    """Create minimal raster state NPZ bytes for testing."""
    buffer = io.BytesIO()
    np.savez(
        buffer,
        water_depth=np.zeros((5, 5), dtype=np.float32),
        qx=np.zeros((5, 5), dtype=np.float32),
        qy=np.zeros((5, 5), dtype=np.float32),
    )
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def swmm_hotstart_bytes() -> bytes:
    """Create fake SWMM hotstart bytes for testing."""
    return b"fake swmm hotstart content"


class TestHotstartWriter:
    """Tests for HotstartWriter.create() method."""

    def test_writer_creates_valid_archive(
        self,
        minimal_domain_data: DomainData,
        minimal_simulation_config: SimulationConfig,
        minimal_simulation_state: HotstartSimulationState,
        raster_state_bytes: bytes,
    ) -> None:
        """Create archive with minimal valid inputs, verify ZIP structure and required members."""
        # Create archive
        archive_buffer = HotstartWriter.create(
            domain_data=minimal_domain_data,
            simulation_config=minimal_simulation_config,
            simulation_state=minimal_simulation_state,
            raster_state_bytes=raster_state_bytes,
        )

        # Verify it's a BytesIO
        assert isinstance(archive_buffer, io.BytesIO)

        # Verify it's a valid ZIP
        archive_buffer.seek(0)
        with zipfile.ZipFile(archive_buffer, mode="r") as zf:
            # Check required members exist
            members = set(zf.namelist())
            assert METADATA_FILENAME in members
            assert RASTER_STATE_FILENAME in members
            assert SWMM_HOTSTART_FILENAME not in members  # Not provided

            # Check metadata is valid JSON
            metadata_bytes = zf.read(METADATA_FILENAME)
            metadata_dict = json.loads(metadata_bytes.decode("utf-8"))
            assert "hotstart_version" in metadata_dict
            assert "creation_date" in metadata_dict
            assert "itzi_version" in metadata_dict
            assert "domain_data" in metadata_dict
            assert "simulation_config" in metadata_dict
            assert "simulation_state" in metadata_dict

    def test_writer_creates_valid_archive_with_swmm(
        self,
        minimal_domain_data: DomainData,
        minimal_simulation_config: SimulationConfig,
        minimal_simulation_state: HotstartSimulationState,
        raster_state_bytes: bytes,
        swmm_hotstart_bytes: bytes,
    ) -> None:
        """Create archive with SWMM hotstart, verify SWMM file is included."""
        archive_buffer = HotstartWriter.create(
            domain_data=minimal_domain_data,
            simulation_config=minimal_simulation_config,
            simulation_state=minimal_simulation_state,
            raster_state_bytes=raster_state_bytes,
            swmm_hotstart_bytes=swmm_hotstart_bytes,
        )

        archive_buffer.seek(0)
        with zipfile.ZipFile(archive_buffer, mode="r") as zf:
            members = set(zf.namelist())
            assert METADATA_FILENAME in members
            assert RASTER_STATE_FILENAME in members
            assert SWMM_HOTSTART_FILENAME in members  # SWMM provided

            # Verify SWMM content
            swmm_content = zf.read(SWMM_HOTSTART_FILENAME)
            assert swmm_content == swmm_hotstart_bytes

    def test_writer_hash_computation(
        self,
        minimal_domain_data: DomainData,
        minimal_simulation_config: SimulationConfig,
        minimal_simulation_state: HotstartSimulationState,
        raster_state_bytes: bytes,
    ) -> None:
        """Verify raster and SWMM hashes are computed correctly from input bytes."""
        # Compute expected hash
        expected_raster_hash = hashlib.blake2b(raster_state_bytes).hexdigest()

        # Create archive
        archive_buffer = HotstartWriter.create(
            domain_data=minimal_domain_data,
            simulation_config=minimal_simulation_config,
            simulation_state=minimal_simulation_state,
            raster_state_bytes=raster_state_bytes,
        )

        # Load and verify hash in metadata
        archive_buffer.seek(0)
        with zipfile.ZipFile(archive_buffer, mode="r") as zf:
            metadata_bytes = zf.read(METADATA_FILENAME)
            metadata_dict = json.loads(metadata_bytes.decode("utf-8"))
            actual_raster_hash = metadata_dict["simulation_state"]["raster_domain_hash"]
            assert actual_raster_hash == expected_raster_hash

    def test_writer_hash_computation_with_swmm(
        self,
        minimal_domain_data: DomainData,
        minimal_simulation_config: SimulationConfig,
        minimal_simulation_state: HotstartSimulationState,
        raster_state_bytes: bytes,
        swmm_hotstart_bytes: bytes,
    ) -> None:
        """Verify SWMM hash is computed correctly when SWMM bytes are provided."""
        # Compute expected hashes
        expected_raster_hash = hashlib.blake2b(raster_state_bytes).hexdigest()
        expected_swmm_hash = hashlib.blake2b(swmm_hotstart_bytes).hexdigest()

        # Create archive
        archive_buffer = HotstartWriter.create(
            domain_data=minimal_domain_data,
            simulation_config=minimal_simulation_config,
            simulation_state=minimal_simulation_state,
            raster_state_bytes=raster_state_bytes,
            swmm_hotstart_bytes=swmm_hotstart_bytes,
        )

        # Load and verify hashes in metadata
        archive_buffer.seek(0)
        with zipfile.ZipFile(archive_buffer, mode="r") as zf:
            metadata_bytes = zf.read(METADATA_FILENAME)
            metadata_dict = json.loads(metadata_bytes.decode("utf-8"))
            actual_raster_hash = metadata_dict["simulation_state"]["raster_domain_hash"]
            actual_swmm_hash = metadata_dict["simulation_state"]["swmm_hotstart_hash"]
            assert actual_raster_hash == expected_raster_hash
            assert actual_swmm_hash == expected_swmm_hash


class TestHotstartLoaderFromFileAndBytes:
    """Tests for HotstartLoader.from_file() and from_bytes() methods."""

    @pytest.fixture
    def valid_archive_buffer(
        self,
        minimal_domain_data: DomainData,
        minimal_simulation_config: SimulationConfig,
        minimal_simulation_state: HotstartSimulationState,
        raster_state_bytes: bytes,
    ) -> io.BytesIO:
        """Create a valid hotstart archive for testing."""
        return HotstartWriter.create(
            domain_data=minimal_domain_data,
            simulation_config=minimal_simulation_config,
            simulation_state=minimal_simulation_state,
            raster_state_bytes=raster_state_bytes,
        )

    def test_loader_from_bytes(
        self,
        valid_archive_buffer: io.BytesIO,
    ) -> None:
        """Load archive from BytesIO."""
        loader = HotstartLoader.from_bytes(valid_archive_buffer)

        assert isinstance(loader, HotstartLoader)
        assert loader._metadata is not None
        assert loader._metadata.hotstart_version == HOTSTART_VERSION

    def test_loader_from_bytes_bytes_input(
        self,
        valid_archive_buffer: io.BytesIO,
    ) -> None:
        """Load archive from raw bytes (not BytesIO)."""
        valid_archive_buffer.seek(0)
        raw_bytes = valid_archive_buffer.read()

        loader = HotstartLoader.from_bytes(raw_bytes)

        assert isinstance(loader, HotstartLoader)
        assert loader._metadata is not None

    def test_loader_from_file(
        self,
        valid_archive_buffer: io.BytesIO,
        tmp_path: Path,
    ) -> None:
        """Load archive from file path."""
        # Write archive to temp file
        hotstart_file = tmp_path / "test_hotstart.zip"
        valid_archive_buffer.seek(0)
        hotstart_file.write_bytes(valid_archive_buffer.read())

        loader = HotstartLoader.from_file(hotstart_file)

        assert isinstance(loader, HotstartLoader)
        assert loader._metadata is not None

    def test_loader_from_file_string_path(
        self,
        valid_archive_buffer: io.BytesIO,
        tmp_path: Path,
    ) -> None:
        """Load archive from string file path (not Path object)."""
        # Write archive to temp file
        hotstart_file = tmp_path / "test_hotstart.zip"
        valid_archive_buffer.seek(0)
        hotstart_file.write_bytes(valid_archive_buffer.read())

        loader = HotstartLoader.from_file(str(hotstart_file))

        assert isinstance(loader, HotstartLoader)


class TestHotstartLoaderRejectsInvalidArchive:
    """Tests for HotstartLoader rejecting invalid archives."""

    def test_loader_rejects_bad_zip(self) -> None:
        """Reject invalid ZIP file."""
        bad_zip = io.BytesIO(b"not a zip file")

        with pytest.raises(HotstartError, match="not a valid ZIP file"):
            HotstartLoader.from_bytes(bad_zip)

    def test_loader_rejects_missing_metadata(self) -> None:
        """Reject archive missing metadata.json."""
        # Create ZIP without metadata
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w") as zf:
            zf.writestr(RASTER_STATE_FILENAME, b"fake raster data")
        buffer.seek(0)

        with pytest.raises(HotstartError, match="missing required members"):
            HotstartLoader.from_bytes(buffer)

    def test_loader_rejects_missing_raster_state(self) -> None:
        """Reject archive missing raster_state.npz."""
        # Create ZIP without raster state
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w") as zf:
            zf.writestr(METADATA_FILENAME, b"{}")
        buffer.seek(0)

        with pytest.raises(HotstartError, match="missing required members"):
            HotstartLoader.from_bytes(buffer)


class TestHotstartLoaderRejectsVersionMismatch:
    """Tests for HotstartLoader rejecting version mismatches."""

    def test_loader_rejects_version_mismatch(
        self,
        minimal_domain_data: DomainData,
        minimal_simulation_config: SimulationConfig,
        minimal_simulation_state: HotstartSimulationState,
        raster_state_bytes: bytes,
    ) -> None:
        """Reject archive with wrong hotstart_version."""
        # Create archive with current version
        archive_buffer = HotstartWriter.create(
            domain_data=minimal_domain_data,
            simulation_config=minimal_simulation_config,
            simulation_state=minimal_simulation_state,
            raster_state_bytes=raster_state_bytes,
        )

        # Modify the version in the archive
        archive_buffer.seek(0)
        with zipfile.ZipFile(archive_buffer, mode="r") as zf_in:
            metadata_bytes = zf_in.read(METADATA_FILENAME)
            metadata_dict = json.loads(metadata_bytes.decode("utf-8"))

        # Change version to invalid value
        metadata_dict["hotstart_version"] = 999

        # Rebuild archive with modified metadata
        archive_buffer.seek(0)
        with zipfile.ZipFile(archive_buffer, mode="r") as zf_in:
            raster_bytes = zf_in.read(RASTER_STATE_FILENAME)

        new_buffer = io.BytesIO()
        with zipfile.ZipFile(new_buffer, mode="w") as zf_out:
            zf_out.writestr(METADATA_FILENAME, json.dumps(metadata_dict).encode("utf-8"))
            zf_out.writestr(RASTER_STATE_FILENAME, raster_bytes)
        new_buffer.seek(0)

        with pytest.raises(HotstartError, match="Unsupported hotstart version"):
            HotstartLoader.from_bytes(new_buffer)


class TestHotstartLoaderRejectsHashMismatch:
    """Tests for HotstartLoader rejecting hash mismatches."""

    def test_loader_rejects_corrupted_raster_bytes(
        self,
        minimal_domain_data: DomainData,
        minimal_simulation_config: SimulationConfig,
        minimal_simulation_state: HotstartSimulationState,
        raster_state_bytes: bytes,
    ) -> None:
        """Reject archive with corrupted raster bytes (hash mismatch)."""
        # Create valid archive
        archive_buffer = HotstartWriter.create(
            domain_data=minimal_domain_data,
            simulation_config=minimal_simulation_config,
            simulation_state=minimal_simulation_state,
            raster_state_bytes=raster_state_bytes,
        )

        # Corrupt the raster state in the archive
        archive_buffer.seek(0)
        with zipfile.ZipFile(archive_buffer, mode="r") as zf_in:
            metadata_bytes = zf_in.read(METADATA_FILENAME)

        corrupted_raster = raster_state_bytes + b"corrupted"
        new_buffer = io.BytesIO()
        with zipfile.ZipFile(new_buffer, mode="w") as zf_out:
            zf_out.writestr(METADATA_FILENAME, metadata_bytes)
            zf_out.writestr(RASTER_STATE_FILENAME, corrupted_raster)
        new_buffer.seek(0)

        with pytest.raises(HotstartError, match="Raster state hash mismatch"):
            HotstartLoader.from_bytes(new_buffer)

    def test_loader_rejects_corrupted_swmm_bytes(
        self,
        minimal_domain_data: DomainData,
        minimal_simulation_config: SimulationConfig,
        minimal_simulation_state: HotstartSimulationState,
        raster_state_bytes: bytes,
        swmm_hotstart_bytes: bytes,
    ) -> None:
        """Reject archive with corrupted SWMM bytes (hash mismatch)."""
        # Create valid archive with SWMM
        archive_buffer = HotstartWriter.create(
            domain_data=minimal_domain_data,
            simulation_config=minimal_simulation_config,
            simulation_state=minimal_simulation_state,
            raster_state_bytes=raster_state_bytes,
            swmm_hotstart_bytes=swmm_hotstart_bytes,
        )

        # Corrupt the SWMM hotstart in the archive
        archive_buffer.seek(0)
        with zipfile.ZipFile(archive_buffer, mode="r") as zf_in:
            metadata_bytes = zf_in.read(METADATA_FILENAME)
            raster_bytes = zf_in.read(RASTER_STATE_FILENAME)

        corrupted_swmm = swmm_hotstart_bytes + b"corrupted"
        new_buffer = io.BytesIO()
        with zipfile.ZipFile(new_buffer, mode="w") as zf_out:
            zf_out.writestr(METADATA_FILENAME, metadata_bytes)
            zf_out.writestr(RASTER_STATE_FILENAME, raster_bytes)
            zf_out.writestr(SWMM_HOTSTART_FILENAME, corrupted_swmm)
        new_buffer.seek(0)

        with pytest.raises(HotstartError, match="SWMM hotstart hash mismatch"):
            HotstartLoader.from_bytes(new_buffer)


class TestHotstartLoaderRejectsSwmmMismatch:
    """Tests for HotstartLoader rejecting SWMM presence mismatches."""

    def test_loader_rejects_swmm_present_but_metadata_says_none(
        self,
        minimal_domain_data: DomainData,
        minimal_simulation_config: SimulationConfig,
        minimal_simulation_state: HotstartSimulationState,
        raster_state_bytes: bytes,
        swmm_hotstart_bytes: bytes,
    ) -> None:
        """Reject when SWMM file present but metadata says none."""
        # Create archive WITHOUT SWMM
        archive_buffer = HotstartWriter.create(
            domain_data=minimal_domain_data,
            simulation_config=minimal_simulation_config,
            simulation_state=minimal_simulation_state,
            raster_state_bytes=raster_state_bytes,
        )

        # Add SWMM file to archive without updating metadata
        archive_buffer.seek(0)
        with zipfile.ZipFile(archive_buffer, mode="r") as zf_in:
            metadata_bytes = zf_in.read(METADATA_FILENAME)
            raster_bytes = zf_in.read(RASTER_STATE_FILENAME)

        new_buffer = io.BytesIO()
        with zipfile.ZipFile(new_buffer, mode="w") as zf_out:
            zf_out.writestr(METADATA_FILENAME, metadata_bytes)
            zf_out.writestr(RASTER_STATE_FILENAME, raster_bytes)
            zf_out.writestr(SWMM_HOTSTART_FILENAME, swmm_hotstart_bytes)
        new_buffer.seek(0)

        with pytest.raises(HotstartError, match="missing swmm_hotstart_hash"):
            HotstartLoader.from_bytes(new_buffer)

    def test_loader_rejects_metadata_says_swmm_but_file_missing(
        self,
        minimal_domain_data: DomainData,
        minimal_simulation_config: SimulationConfig,
        minimal_simulation_state: HotstartSimulationState,
        raster_state_bytes: bytes,
        swmm_hotstart_bytes: bytes,
    ) -> None:
        """Reject when metadata says SWMM present but file is missing."""
        # Create archive WITH SWMM
        archive_buffer = HotstartWriter.create(
            domain_data=minimal_domain_data,
            simulation_config=minimal_simulation_config,
            simulation_state=minimal_simulation_state,
            raster_state_bytes=raster_state_bytes,
            swmm_hotstart_bytes=swmm_hotstart_bytes,
        )

        # Remove SWMM file from archive
        archive_buffer.seek(0)
        with zipfile.ZipFile(archive_buffer, mode="r") as zf_in:
            metadata_bytes = zf_in.read(METADATA_FILENAME)
            raster_bytes = zf_in.read(RASTER_STATE_FILENAME)

        new_buffer = io.BytesIO()
        with zipfile.ZipFile(new_buffer, mode="w") as zf_out:
            zf_out.writestr(METADATA_FILENAME, metadata_bytes)
            zf_out.writestr(RASTER_STATE_FILENAME, raster_bytes)
            # SWMM file intentionally omitted
        new_buffer.seek(0)

        with pytest.raises(
            HotstartError, match="SWMM hotstart should be present but it's missing"
        ):
            HotstartLoader.from_bytes(new_buffer)


class TestHotstartLoaderFileNotFound:
    """Tests for HotstartLoader file not found error."""

    def test_loader_rejects_nonexistent_file(self, tmp_path: Path) -> None:
        """Reject when file does not exist."""
        nonexistent_file = tmp_path / "nonexistent.zip"

        with pytest.raises(HotstartError, match="Hotstart file not found"):
            HotstartLoader.from_file(nonexistent_file)
