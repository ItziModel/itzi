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

import hashlib
import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from importlib.metadata import version

from itzi.itzi_error import HotstartError
from itzi.data_containers import HotstartSimulationState, HotstartMetadata

if TYPE_CHECKING:
    from itzi.providers.domain_data import DomainData
    from itzi.data_containers import SimulationConfig


# Hotstart archive format constants
HOTSTART_VERSION = 1
METADATA_FILENAME = "metadata.json"
RASTER_STATE_FILENAME = "raster_state.npz"
SWMM_HOTSTART_FILENAME = "swmm_hotstart.hsf"


class HotstartWriter:
    """Create hotstart archives from simulation state."""

    @classmethod
    def create(
        cls,
        domain_data: DomainData,
        simulation_config: SimulationConfig,
        simulation_state: HotstartSimulationState,
        raster_state_bytes: bytes,
        swmm_hotstart_bytes: bytes | None = None,
    ) -> io.BytesIO:
        """
        Create a hotstart archive from provided state payloads.

        Hashes for ``raster_state_bytes`` and (optionally) ``swmm_hotstart_bytes``
        are computed here and injected into a copy of ``simulation_state`` before
        the archive is serialised.  Callers do not need to supply hash values.

        Args:
            domain_data: Domain metadata
            simulation_config: Simulation configuration
            simulation_state: Runtime state (sim_time, dt, counters, etc.).
                The ``raster_domain_hash`` and ``swmm_hotstart_hash`` fields are
                ignored; the writer always computes them from the binary payloads.
            raster_state_bytes: Serialized raster domain state (.npz format)
            swmm_hotstart_bytes: Optional SWMM hotstart file bytes

        Returns:
            BytesIO containing the complete hotstart archive
        """
        # Compute hashes from the binary payloads and inject them into state.
        raster_hash = hashlib.blake2b(raster_state_bytes).hexdigest()
        swmm_hash = None
        if swmm_hotstart_bytes is not None:
            swmm_hash = hashlib.blake2b(swmm_hotstart_bytes).hexdigest()
        simulation_state = simulation_state.model_copy(
            update={"raster_domain_hash": raster_hash, "swmm_hotstart_hash": swmm_hash}
        )

        # Build metadata using Pydantic model for validation
        metadata = HotstartMetadata(
            creation_date=datetime.now(timezone.utc),
            itzi_version=version("itzi"),
            hotstart_version=HOTSTART_VERSION,
            domain_data=domain_data,
            simulation_config=simulation_config,
            simulation_state=simulation_state,
        )

        # Create zip archive
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(
            zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=8
        ) as zip_file:
            # Write raster state
            zip_file.writestr(RASTER_STATE_FILENAME, raster_state_bytes)

            # Write SWMM hotstart if present
            if swmm_hotstart_bytes is not None:
                zip_file.writestr(SWMM_HOTSTART_FILENAME, swmm_hotstart_bytes)

            # Write metadata JSON using Pydantic serialization
            json_str = metadata.model_dump_json(indent=2)
            zip_file.writestr(METADATA_FILENAME, json_str)

        # Reset position for reading
        zip_buffer.seek(0)
        return zip_buffer


class HotstartLoader:
    """Load and validate hotstart archives."""

    def __init__(
        self,
        metadata: HotstartMetadata,
        raster_state_bytes: bytes,
        swmm_hotstart_bytes: bytes | None,
    ):
        """
        Initialize a validated hotstart loader.

        Args:
            metadata: Parsed and validated HotstartMetadata instance
            raster_state_bytes: Raster state NPZ bytes
            swmm_hotstart_bytes: Optional SWMM hotstart bytes
        """
        self._metadata = metadata
        self.raster_state_bytes = raster_state_bytes
        self.swmm_hotstart_bytes = swmm_hotstart_bytes

    @classmethod
    def from_file(cls, path: str | Path) -> "HotstartLoader":
        """
        Load a hotstart archive from a file path.

        Args:
            path: Path to the hotstart archive file

        Returns:
            Validated HotstartLoader instance

        Raises:
            HotstartError: If the archive is invalid or corrupted
        """
        path = Path(path)
        if not path.exists():
            raise HotstartError(f"Hotstart file not found: {path}")

        with open(path, "rb") as f:
            data = f.read()

        return cls.from_bytes(data)

    @classmethod
    def from_bytes(cls, data: io.BytesIO | bytes) -> "HotstartLoader":
        """
        Load a hotstart archive from bytes or BytesIO.

        Args:
            data: Hotstart archive as bytes or BytesIO

        Returns:
            Validated HotstartLoader instance

        Raises:
            HotstartError: If the archive is invalid or corrupted
        """
        # Convert to bytes if needed
        if isinstance(data, io.BytesIO):
            data.seek(0)
            archive_bytes = data.read()
        else:
            archive_bytes = data

        # Open as zip archive
        try:
            zip_buffer = io.BytesIO(archive_bytes)
            zip_file = zipfile.ZipFile(zip_buffer, mode="r")
        except zipfile.BadZipFile as e:
            raise HotstartError("Invalid hotstart archive: not a valid ZIP file") from e

        # Validate archive structure
        cls._validate_archive_structure(zip_file)

        # Load and validate metadata using Pydantic
        try:
            metadata_bytes = zip_file.read(METADATA_FILENAME)
            metadata = HotstartMetadata.model_validate_json(metadata_bytes.decode("utf-8"))
        except Exception as e:
            raise HotstartError(f"Failed to validate metadata: {e}") from e

        # Validate hotstart version compatibility
        cls._validate_version(metadata)

        # Load raster state
        try:
            raster_state_bytes = zip_file.read(RASTER_STATE_FILENAME)
        except KeyError as e:
            raise HotstartError("Missing raster state in archive") from e

        # Load optional SWMM hotstart
        swmm_hotstart_bytes = None
        if SWMM_HOTSTART_FILENAME in zip_file.namelist():
            swmm_hotstart_bytes = zip_file.read(SWMM_HOTSTART_FILENAME)

        zip_file.close()

        # Validate hashes using the validated simulation_state
        cls._validate_hashes(metadata.simulation_state, raster_state_bytes, swmm_hotstart_bytes)

        return cls(metadata, raster_state_bytes, swmm_hotstart_bytes)

    @staticmethod
    def _validate_archive_structure(zip_file: zipfile.ZipFile) -> None:
        """
        Validate that required archive members are present.

        Args:
            zip_file: Open ZipFile object

        Raises:
            HotstartError: If required members are missing
        """
        members = set(zip_file.namelist())

        # Check required members
        required = {METADATA_FILENAME, RASTER_STATE_FILENAME}
        missing = required - members

        if missing:
            raise HotstartError(f"Hotstart archive missing required members: {', '.join(missing)}")

        # Validate optional SWMM member rules
        # SWMM hotstart is optional, but if present must be valid
        # (validation happens via hash check later)

    @staticmethod
    def _validate_version(metadata: HotstartMetadata) -> None:
        """
        Validate hotstart version compatibility.

        Args:
            metadata: Parsed metadata dictionary

        Raises:
            HotstartError: If version is unsupported
        """
        # hotstart_version is a required field; if it were absent Pydantic would
        # have already raised a ValidationError during model_validate_json().
        archive_version = metadata.hotstart_version

        if archive_version != HOTSTART_VERSION:
            raise HotstartError(
                f"Unsupported hotstart version {archive_version}. "
                f"Expected version {HOTSTART_VERSION}. "
                f"This hotstart file may have been created by a different version of Itzi."
            )

    @staticmethod
    def _validate_hashes(
        simulation_state: HotstartSimulationState,
        raster_state_bytes: bytes,
        swmm_hotstart_bytes: bytes | None,
    ) -> None:
        """
        Validate stored BLAKE2 hashes against actual data.

        Args:
            simulation_state: Simulation state dictionary from metadata
            raster_state_bytes: Raster state bytes to validate
            swmm_hotstart_bytes: Optional SWMM hotstart bytes to validate

        Raises:
            HotstartError: If hashes don't match or are missing
        """
        # Validate raster state hash
        stored_raster_hash = simulation_state.raster_domain_hash
        computed_raster_hash = hashlib.blake2b(raster_state_bytes).hexdigest()
        if computed_raster_hash != stored_raster_hash:
            raise HotstartError(
                f"Raster state hash mismatch. Archive may be corrupted. "
                f"Expected {stored_raster_hash}, got {computed_raster_hash}"
            )

        # Validate SWMM hotstart hash if present
        stored_swmm_hash = simulation_state.swmm_hotstart_hash

        if swmm_hotstart_bytes is not None:
            if stored_swmm_hash is None:
                raise HotstartError(
                    "Archive contains SWMM hotstart but metadata is missing swmm_hotstart_hash"
                )

            computed_swmm_hash = hashlib.blake2b(swmm_hotstart_bytes).hexdigest()
            if computed_swmm_hash != stored_swmm_hash:
                raise HotstartError(
                    f"SWMM hotstart hash mismatch. Archive may be corrupted. "
                    f"Expected {stored_swmm_hash}, got {computed_swmm_hash}"
                )
        elif stored_swmm_hash is not None:
            raise HotstartError(
                "Metadata indicates SWMM hotstart should be present but it's missing from archive"
            )

    def get_domain_data(self) -> DomainData:
        """
        Get domain metadata from the hotstart.

        Returns:
            DomainData instance
        """
        return self._metadata.domain_data

    def get_simulation_config(self) -> SimulationConfig:
        """
        Get simulation configuration from the hotstart.

        Returns:
            SimulationConfig instance
        """
        return self._metadata.simulation_config

    def get_simulation_state(self) -> HotstartSimulationState:
        """
        Get simulation runtime state from the hotstart.

        Returns:
            HotstartSimulationState instance
        """
        return self._metadata.simulation_state

    def get_raster_state_buffer(self) -> io.BytesIO:
        """Get raster state as a BytesIO buffer."""
        return io.BytesIO(self.raster_state_bytes)

    def get_swmm_hotstart_bytes(self) -> bytes | None:
        """Get SWMM hotstart bytes if present."""
        return self.swmm_hotstart_bytes

    def has_swmm_hotstart(self) -> bool:
        """Check if the hotstart includes SWMM state."""
        return self.swmm_hotstart_bytes is not None
