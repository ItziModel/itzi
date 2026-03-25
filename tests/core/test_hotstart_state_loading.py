"""Tests for RasterDomain.load_state() and SimulationBuilder hotstart integration.

This file implements phases 2 and 3 from the hotstart testing plan:
- Phase 2: RasterDomain.load_state() tests
- Phase 3: SimulationBuilder Hotstart Integration tests
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from itzi.data_containers import HotstartSimulationState, SimulationConfig, SurfaceFlowParameters
from itzi.hotstart import HotstartLoader, HotstartWriter
from itzi.itzi_error import HotstartError
from itzi.providers.domain_data import DomainData
from itzi.providers.memory_output import MemoryRasterOutputProvider, MemoryVectorOutputProvider
from itzi.rasterdomain import RasterDomain
from itzi.simulation_builder import SimulationBuilder
from itzi.const import InfiltrationModelType, TemporalType


class TestRasterDomainLoadState:
    """Tests for RasterDomain.load_state() method."""

    @pytest.fixture
    def raster_domain(self, domain_5by5) -> RasterDomain:
        """Create a RasterDomain for testing."""
        return RasterDomain(
            dtype=np.float32,
            arr_mask=domain_5by5.arr_mask,
            cell_shape=domain_5by5.domain_data.cell_shape,
        )

    @pytest.fixture
    def populated_raster_domain(self, raster_domain, domain_5by5) -> RasterDomain:
        """Create a RasterDomain with all required arrays populated."""
        # Set input arrays
        raster_domain.update_array("water_depth", domain_5by5.arr_start_h.copy())
        raster_domain.update_array("dem", domain_5by5.arr_dem_flat.copy())
        raster_domain.update_array("friction", domain_5by5.arr_n.copy())
        # Set internal arrays (required for save_state)
        # qe and qs are internal state arrays (flux at eastern/southern edges)
        qe = np.full(raster_domain.shape, 1.1, dtype=np.float32)
        qs = np.full(raster_domain.shape, 1.2, dtype=np.float32)
        raster_domain.update_array("qe", qe)
        raster_domain.update_array("qs", qs)
        return raster_domain

    def test_load_state_round_trip(self, populated_raster_domain: RasterDomain) -> None:
        """save_state → load_state cycle should preserve all arrays."""
        saved_state = populated_raster_domain.save_state()

        new_domain = RasterDomain(
            dtype=np.float32,
            arr_mask=populated_raster_domain.mask.copy(),
            cell_shape=(populated_raster_domain.dx, populated_raster_domain.dy),
        )
        new_domain.load_state(saved_state)

        for key in populated_raster_domain.k_all:
            original = populated_raster_domain.get_array(key)
            restored = new_domain.get_array(key)
            np.testing.assert_allclose(restored, original, err_msg=f"Mismatch in {key}")

    def test_load_state_rejects_shape_mismatch(
        self, populated_raster_domain: RasterDomain
    ) -> None:
        """load_state should reject state with wrong shape."""
        # Create state with all required keys but wrong shape
        wrong_shape = (3, 3)
        buffer = io.BytesIO()
        arrays = {"mask": np.full(wrong_shape, False, dtype=np.bool_)}
        for key in populated_raster_domain.k_all:
            arrays[key] = np.zeros(wrong_shape, dtype=np.float32)
        np.savez(buffer, allow_pickle=False, **arrays)
        buffer.seek(0)
        with pytest.raises(HotstartError, match="Mask shape mismatch"):
            populated_raster_domain.load_state(buffer)

    def test_load_state_rejects_mask_mismatch(self, populated_raster_domain: RasterDomain) -> None:
        """load_state should reject state with different mask content."""
        saved_state = populated_raster_domain.save_state()
        saved_state.seek(0)
        npz = np.load(saved_state, allow_pickle=False)
        arrays = {k: npz[k] for k in npz.files}
        arrays["mask"] = ~arrays["mask"]  # Flip mask
        buffer = io.BytesIO()
        np.savez(buffer, allow_pickle=False, **arrays)
        buffer.seek(0)
        with pytest.raises(HotstartError, match="Mask content mismatch"):
            populated_raster_domain.load_state(buffer)

    def test_load_state_rejects_missing_keys(self, populated_raster_domain: RasterDomain) -> None:
        """load_state should reject state missing required keys."""
        buffer = io.BytesIO()
        np.savez(
            buffer,
            allow_pickle=False,
            mask=populated_raster_domain.mask.copy(),
            water_depth=np.zeros(populated_raster_domain.shape, dtype=np.float32),
            # Missing qe and qs (internal arrays)
        )
        buffer.seek(0)
        with pytest.raises(HotstartError, match="missing required arrays"):
            populated_raster_domain.load_state(buffer)

    def test_load_state_rejects_dtype_mismatch(
        self, populated_raster_domain: RasterDomain
    ) -> None:
        """load_state should reject state with incompatible dtype."""
        saved_state = populated_raster_domain.save_state()
        saved_state.seek(0)
        npz = np.load(saved_state, allow_pickle=False)
        arrays = {k: npz[k] for k in npz.files}
        arrays["water_depth"] = arrays["water_depth"].astype(np.complex128)
        buffer = io.BytesIO()
        np.savez(buffer, allow_pickle=False, **arrays)
        buffer.seek(0)
        with pytest.raises(HotstartError, match="dtype mismatch"):
            populated_raster_domain.load_state(buffer)


class TestSimulationBuilderHotstart:
    """Tests for SimulationBuilder hotstart integration."""

    @staticmethod
    def _create_hotstart_bytes(domain_5by5, sim_config: SimulationConfig) -> io.BytesIO:
        raster_output = MemoryRasterOutputProvider({"out_map_names": sim_config.output_map_names})
        simulation = (
            SimulationBuilder(sim_config, domain_5by5.arr_mask, np.float32)
            .with_domain_data(domain_5by5.domain_data)
            .with_raster_output_provider(raster_output)
            .with_vector_output_provider(MemoryVectorOutputProvider({}))
            .build()
        )
        simulation.set_array("dem", domain_5by5.arr_dem_flat.copy())
        simulation.set_array("friction", domain_5by5.arr_n.copy())
        simulation.set_array("water_depth", domain_5by5.arr_start_h.copy())
        simulation.initialize()
        return simulation.create_hotstart()

    @staticmethod
    def _build_with_hotstart(
        domain_5by5,
        sim_config: SimulationConfig,
        hotstart_bytes: io.BytesIO,
        domain_data: DomainData | None = None,
    ):
        raster_output = MemoryRasterOutputProvider({"out_map_names": sim_config.output_map_names})
        builder = (
            SimulationBuilder(sim_config, domain_5by5.arr_mask, np.float32)
            .with_domain_data(domain_data or domain_5by5.domain_data)
            .with_raster_output_provider(raster_output)
            .with_vector_output_provider(MemoryVectorOutputProvider({}))
            .with_hotstart(hotstart_bytes)
        )
        return builder.build()

    @pytest.fixture
    def sim_config(self, helpers) -> SimulationConfig:
        """Create a basic SimulationConfig for testing."""
        return SimulationConfig(
            start_time=datetime(2000, 1, 1, 0, 0, 0),
            end_time=datetime(2000, 1, 1, 0, 1, 0),
            record_step=timedelta(seconds=30),
            temporal_type=TemporalType.RELATIVE,
            input_map_names=helpers.make_input_map_names(
                dem="z",
                friction="n",
                water_depth="start_h",
            ),
            output_map_names=helpers.make_output_map_names(
                "out_test",
                ["water_depth", "qx", "qy"],
            ),
            surface_flow_parameters=SurfaceFlowParameters(hmin=0.0001, dtmax=0.3, cfl=0.2),
            infiltration_model=InfiltrationModelType.NULL,
        )

    @pytest.fixture
    def valid_hotstart_bytes(
        self,
        domain_5by5,
        sim_config: SimulationConfig,
    ) -> io.BytesIO:
        """Create a valid hotstart archive for testing."""
        return self._create_hotstart_bytes(domain_5by5, sim_config)

    def test_with_hotstart_from_bytes(
        self,
        domain_5by5,
        sim_config: SimulationConfig,
        valid_hotstart_bytes: io.BytesIO,
    ) -> None:
        """Builder should store HotstartLoader from BytesIO."""
        raster_output = MemoryRasterOutputProvider({"out_map_names": sim_config.output_map_names})
        builder = (
            SimulationBuilder(sim_config, domain_5by5.arr_mask, np.float32)
            .with_domain_data(domain_5by5.domain_data)
            .with_raster_output_provider(raster_output)
            .with_vector_output_provider(MemoryVectorOutputProvider({}))
            .with_hotstart(valid_hotstart_bytes)
        )

        assert builder.hotstart_loader is not None
        assert isinstance(builder.hotstart_loader, HotstartLoader)

    def test_with_hotstart_from_file(
        self,
        domain_5by5,
        sim_config: SimulationConfig,
        valid_hotstart_bytes: io.BytesIO,
        tmp_path: Path,
    ) -> None:
        """Builder should store HotstartLoader from file path."""
        hotstart_file = tmp_path / "test_hotstart.zip"
        valid_hotstart_bytes.seek(0)
        hotstart_file.write_bytes(valid_hotstart_bytes.read())

        raster_output = MemoryRasterOutputProvider({"out_map_names": sim_config.output_map_names})
        builder = (
            SimulationBuilder(sim_config, domain_5by5.arr_mask, np.float32)
            .with_domain_data(domain_5by5.domain_data)
            .with_raster_output_provider(raster_output)
            .with_vector_output_provider(MemoryVectorOutputProvider({}))
            .with_hotstart(hotstart_file)
        )

        assert builder.hotstart_loader is not None
        assert isinstance(builder.hotstart_loader, HotstartLoader)

    @pytest.mark.parametrize(
        "field,value,expected_error",
        [
            ("north", 100.0, "Domain north mismatch"),
            ("south", 25.0, "Domain south mismatch"),
            ("east", 100.0, "Domain east mismatch"),
            ("west", 25.0, "Domain west mismatch"),
            ("rows", 999, "Domain rows mismatch"),
            ("cols", 999, "Domain cols mismatch"),
        ],
    )
    def test_build_rejects_domain_mismatch(
        self,
        domain_5by5,
        sim_config: SimulationConfig,
        valid_hotstart_bytes: io.BytesIO,
        field: str,
        value: float | int,
        expected_error: str,
    ) -> None:
        """build() should reject domain metadata mismatches."""
        # Create modified domain data with valid values (east > west, north > south)
        dd = domain_5by5.domain_data
        kwargs = {
            "north": dd.north,
            "south": dd.south,
            "east": dd.east,
            "west": dd.west,
            "rows": dd.rows,
            "cols": dd.cols,
            "crs_wkt": dd.crs_wkt,
        }
        kwargs[field] = value
        mismatched_domain = DomainData(**kwargs)

        with pytest.raises(HotstartError, match=expected_error):
            self._build_with_hotstart(
                domain_5by5,
                sim_config,
                valid_hotstart_bytes,
                domain_data=mismatched_domain,
            )

    def test_build_rejects_crs_mismatch(
        self,
        domain_5by5,
        sim_config: SimulationConfig,
        valid_hotstart_bytes: io.BytesIO,
    ) -> None:
        """build() should reject domain CRS mismatches."""
        dd = domain_5by5.domain_data
        mismatched_domain = DomainData(
            north=dd.north,
            south=dd.south,
            east=dd.east,
            west=dd.west,
            rows=dd.rows,
            cols=dd.cols,
            crs_wkt="EPSG:4326",
        )

        with pytest.raises(HotstartError, match="Domain CRS mismatch"):
            self._build_with_hotstart(
                domain_5by5,
                sim_config,
                valid_hotstart_bytes,
                domain_data=mismatched_domain,
            )

    @pytest.mark.parametrize(
        ("archived_model", "resumed_model"),
        [
            (InfiltrationModelType.GREEN_AMPT, InfiltrationModelType.NULL),
            (InfiltrationModelType.NULL, InfiltrationModelType.GREEN_AMPT),
        ],
    )
    def test_build_rejects_infiltration_model_mismatch(
        self,
        domain_5by5,
        sim_config: SimulationConfig,
        archived_model: InfiltrationModelType,
        resumed_model: InfiltrationModelType,
    ) -> None:
        """build() should reject hotstarts with incompatible infiltration models."""
        archived_config = sim_config.model_copy(update={"infiltration_model": archived_model})
        resumed_config = sim_config.model_copy(update={"infiltration_model": resumed_model})
        hotstart_bytes = self._create_hotstart_bytes(domain_5by5, archived_config)

        with pytest.raises(HotstartError, match="infiltration"):
            self._build_with_hotstart(domain_5by5, resumed_config, hotstart_bytes)

    @pytest.mark.parametrize(
        ("parameter", "value"),
        [
            ("hmin", 0.0002),
        ],
    )
    def test_build_rejects_surface_flow_parameter_mismatch(
        self,
        domain_5by5,
        sim_config: SimulationConfig,
        valid_hotstart_bytes: io.BytesIO,
        parameter: str,
        value: float,
    ) -> None:
        """build() should reject solver-affecting surface-flow parameter changes."""
        resumed_config = sim_config.model_copy(
            update={
                "surface_flow_parameters": sim_config.surface_flow_parameters.model_copy(
                    update={parameter: value}
                )
            }
        )

        with pytest.raises(HotstartError, match=f"{parameter}|surface"):
            self._build_with_hotstart(domain_5by5, resumed_config, valid_hotstart_bytes)

    def test_build_rejects_drainage_mismatch_hotstart_has_drainage(
        self,
        domain_5by5,
        helpers,
    ) -> None:
        """build() should reject when hotstart has drainage but config does not."""
        config_with_drainage = SimulationConfig(
            start_time=datetime(2000, 1, 1, 0, 0, 0),
            end_time=datetime(2000, 1, 1, 0, 1, 0),
            record_step=timedelta(seconds=30),
            temporal_type=TemporalType.RELATIVE,
            input_map_names=helpers.make_input_map_names(),
            output_map_names=helpers.make_output_map_names("out", ["water_depth"]),
            surface_flow_parameters=SurfaceFlowParameters(),
            swmm_inp="fake.inp",  # Has drainage
        )

        raster_domain = RasterDomain(
            dtype=np.float32,
            arr_mask=domain_5by5.arr_mask,
            cell_shape=domain_5by5.domain_data.cell_shape,
        )
        raster_state = raster_domain.save_state()

        simulation_state = HotstartSimulationState(
            sim_time="2000-01-01T00:00:30",
            dt=0.5,
            next_ts={},
            time_steps_counters={},
            accum_update_time={},
            old_domain_volume=100.0,
        )

        hotstart_bytes = HotstartWriter.create(
            domain_data=domain_5by5.domain_data,
            simulation_config=config_with_drainage,
            simulation_state=simulation_state,
            raster_state_bytes=raster_state.getvalue(),
            swmm_hotstart_bytes=b"fake swmm data",
        )

        config_no_drainage = SimulationConfig(
            start_time=datetime(2000, 1, 1, 0, 0, 0),
            end_time=datetime(2000, 1, 1, 0, 1, 0),
            record_step=timedelta(seconds=30),
            temporal_type=TemporalType.RELATIVE,
            input_map_names=helpers.make_input_map_names(),
            output_map_names=helpers.make_output_map_names("out", ["water_depth"]),
            surface_flow_parameters=SurfaceFlowParameters(),
            swmm_inp=None,  # No drainage
        )

        raster_output = MemoryRasterOutputProvider(
            {"out_map_names": config_no_drainage.output_map_names}
        )

        with pytest.raises(
            HotstartError, match="Hotstart contains drainage state but current configuration"
        ):
            (
                SimulationBuilder(config_no_drainage, domain_5by5.arr_mask, np.float32)
                .with_domain_data(domain_5by5.domain_data)
                .with_raster_output_provider(raster_output)
                .with_vector_output_provider(MemoryVectorOutputProvider({}))
                .with_hotstart(hotstart_bytes)
                .build()
            )

    def test_build_rejects_drainage_mismatch_config_has_drainage(
        self,
        domain_5by5,
        helpers,
    ) -> None:
        """build() should reject when config has drainage but hotstart does not."""
        config_no_drainage = SimulationConfig(
            start_time=datetime(2000, 1, 1, 0, 0, 0),
            end_time=datetime(2000, 1, 1, 0, 1, 0),
            record_step=timedelta(seconds=30),
            temporal_type=TemporalType.RELATIVE,
            input_map_names=helpers.make_input_map_names(),
            output_map_names=helpers.make_output_map_names("out", ["water_depth"]),
            surface_flow_parameters=SurfaceFlowParameters(),
            swmm_inp=None,  # No drainage
        )

        raster_domain = RasterDomain(
            dtype=np.float32,
            arr_mask=domain_5by5.arr_mask,
            cell_shape=domain_5by5.domain_data.cell_shape,
        )
        raster_state = raster_domain.save_state()

        simulation_state = HotstartSimulationState(
            sim_time="2000-01-01T00:00:30",
            dt=0.5,
            next_ts={},
            time_steps_counters={},
            accum_update_time={},
            old_domain_volume=100.0,
        )

        hotstart_bytes = HotstartWriter.create(
            domain_data=domain_5by5.domain_data,
            simulation_config=config_no_drainage,
            simulation_state=simulation_state,
            raster_state_bytes=raster_state.getvalue(),
        )

        config_with_drainage = SimulationConfig(
            start_time=datetime(2000, 1, 1, 0, 0, 0),
            end_time=datetime(2000, 1, 1, 0, 1, 0),
            record_step=timedelta(seconds=30),
            temporal_type=TemporalType.RELATIVE,
            input_map_names=helpers.make_input_map_names(),
            output_map_names=helpers.make_output_map_names("out", ["water_depth"]),
            surface_flow_parameters=SurfaceFlowParameters(),
            swmm_inp="fake.inp",  # Has drainage
        )

        raster_output = MemoryRasterOutputProvider(
            {"out_map_names": config_with_drainage.output_map_names}
        )

        with pytest.raises(
            HotstartError, match="Hotstart has no drainage state but current configuration"
        ):
            (
                SimulationBuilder(config_with_drainage, domain_5by5.arr_mask, np.float32)
                .with_domain_data(domain_5by5.domain_data)
                .with_raster_output_provider(raster_output)
                .with_vector_output_provider(MemoryVectorOutputProvider({}))
                .with_hotstart(hotstart_bytes)
                .build()
            )
