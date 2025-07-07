import pytest
import tempfile
from datetime import datetime
from itzi.massbalance import MassBalanceLogger


@pytest.fixture
def logger_fixture():
    fields = ["sim_time", "volume", "%error"]
    start_time = datetime.now()
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    file_name = temp_file.name
    yield {
        "fields": fields,
        "start_time": start_time,
        "file_name": file_name,
        "temp_file": temp_file,
    }
    temp_file.close()


def test_init_with_custom_filename(logger_fixture):
    logger = MassBalanceLogger(
        file_name=logger_fixture["file_name"],
        start_time=logger_fixture["start_time"],
        temporal_type="absolute",
        fields=logger_fixture["fields"],
    )
    assert logger.file_name == logger_fixture["file_name"]


def test_init_with_default_filename():
    fields = ["sim_time", "volume", "%error"]
    start_time = datetime.now()
    logger = MassBalanceLogger(
        file_name="", start_time=start_time, temporal_type="absolute", fields=fields
    )
    assert logger.file_name.endswith("_stats.csv")


def test_init_invalid_temporal_type(logger_fixture):
    with pytest.raises(ValueError):
        MassBalanceLogger(
            file_name=logger_fixture["file_name"],
            start_time=logger_fixture["start_time"],
            temporal_type="invalid",
            fields=logger_fixture["fields"],
        )


def test_log_absolute_time(logger_fixture):
    logger = MassBalanceLogger(
        file_name=logger_fixture["file_name"],
        start_time=logger_fixture["start_time"],
        temporal_type="absolute",
        fields=logger_fixture["fields"],
    )
    test_data = {"sim_time": datetime.now(), "volume": 123.456789, "%error": 0.123456}
    logger.log(test_data)
    with open(logger_fixture["file_name"], "r") as f:
        lines = f.readlines()
        assert len(lines) == 2  # header + 1 data row
        assert "123.457" in lines[1]  # float formatting
        assert "12.35%" in lines[1]  # percentage formatting


def test_log_relative_time(logger_fixture):
    logger = MassBalanceLogger(
        file_name=logger_fixture["file_name"],
        start_time=logger_fixture["start_time"],
        temporal_type="relative",
        fields=logger_fixture["fields"],
    )
    test_time = datetime.now()
    test_data = {"sim_time": test_time, "volume": 123.456789, "%error": 0.123456}
    logger.log(test_data)
    with open(logger_fixture["file_name"], "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        # Verify relative time calculation
        expected_time = str((test_time - logger_fixture["start_time"]))
        assert expected_time in lines[1]
