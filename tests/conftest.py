import os

import pytest


class Helpers:
    @staticmethod
    def roughness(timeseries):
        """Sum of the squared difference of
        the normalized differences.
        """
        f = timeseries.diff()
        normed_f = (f - f.mean()) / f.std()
        return (normed_f.diff() ** 2).sum()


@pytest.fixture(scope="session")
def helpers():
    return Helpers


@pytest.fixture(scope="session")
def test_data_path():
    """Path to the permanent test data directory."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, "test_data")


@pytest.fixture(scope="session")
def test_data_temp_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Worker-local directory where generated test data resides."""
    return str(tmp_path_factory.mktemp("test_data"))
