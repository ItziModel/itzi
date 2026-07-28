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
def test_data_temp_path():
    """Directory where generated test data resides."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    temp_path = os.path.join(dir_path, "test_data_temp")
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    return temp_path
