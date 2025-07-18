import pytest
import tempfile
from datetime import datetime

from itzi.massbalance import MassBalanceLogger
from itzi.data_containers import MassBalanceData


@pytest.fixture
def logger_fixture():
    start_time = datetime.now()
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    file_name = temp_file.name
    yield {
        "start_time": start_time,
        "file_name": file_name,
    }
    temp_file.close()


def test_init_with_custom_filename(logger_fixture):
    logger = MassBalanceLogger(
        file_name=logger_fixture["file_name"],
        start_time=logger_fixture["start_time"],
        temporal_type="absolute",
    )
    assert logger.file_name == logger_fixture["file_name"]


def test_init_with_default_filename():
    start_time = datetime.now()
    logger = MassBalanceLogger(
        file_name="",
        start_time=start_time,
        temporal_type="absolute",
    )
    assert logger.file_name.endswith("_stats.csv")


def test_init_invalid_temporal_type(logger_fixture):
    with pytest.raises(ValueError):
        MassBalanceLogger(
            file_name=logger_fixture["file_name"],
            start_time=logger_fixture["start_time"],
            temporal_type="invalid",
        )


def test_log_absolute_time(logger_fixture):
    logger = MassBalanceLogger(
        file_name=logger_fixture["file_name"],
        start_time=logger_fixture["start_time"],
        temporal_type="absolute",
    )
    test_time = datetime.now()
    test_data = MassBalanceData(
        simulation_time=test_time,
        average_timestep=12.42345,
        timesteps=34,
        boundary_volume=123.456789,
        rainfall_volume=12.34567,
        infiltration_volume=-12.434567,
        inflow_volume=12.34567,
        losses_volume=-12.34567,
        drainage_network_volume=12.34567,
        domain_volume=12.34567,
        volume_change=12.34567,
        volume_error=12.34567,
        percent_error=0.123456,
    )

    logger.log(test_data)
    with open(logger_fixture["file_name"], "r") as f:
        lines = f.readlines()
        assert str(test_time) in lines[1]  # datetime formatting
        assert "123.457" in lines[1]  # float formatting
        assert "12.346" in lines[1]  # float formatting
        assert "34" in lines[1]  # int formatting
        assert "12.35%" in lines[1]  # percentage formatting


def test_log_relative_time(logger_fixture):
    logger = MassBalanceLogger(
        file_name=logger_fixture["file_name"],
        start_time=logger_fixture["start_time"],
        temporal_type="relative",
    )
    test_time = datetime.now()
    test_data = MassBalanceData(
        simulation_time=test_time,
        average_timestep=12.42345,
        timesteps=34,
        boundary_volume=123.456789,
        rainfall_volume=12.34567,
        infiltration_volume=-12.434567,
        inflow_volume=12.34567,
        losses_volume=-12.34567,
        drainage_network_volume=12.34567,
        domain_volume=12.34567,
        volume_change=12.34567,
        volume_error=12.34567,
        percent_error=0.123456,
    )
    logger.log(test_data)
    with open(logger_fixture["file_name"], "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        # Verify relative time calculation
        expected_time = str((test_time - logger_fixture["start_time"]))
        assert expected_time in lines[1]
