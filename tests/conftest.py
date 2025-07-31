import hashlib
import os

import pytest
import numpy as np


class Helpers:
    @staticmethod
    def get_rmse(model, ref):
        """return root mean square error"""
        return np.sqrt(np.mean((model - ref) ** 2))

    @staticmethod
    def get_nse(model, ref):
        """Nash-Sutcliffe Efficiency"""
        noise = np.mean((ref - model) ** 2)
        information = np.mean((ref - np.mean(ref)) ** 2)
        return 1 - (noise / information)

    @staticmethod
    def get_rsr(model, ref):
        """RMSE/StdDev ratio"""
        rmse = Helpers.get_rmse(model, ref)
        return rmse / np.std(ref)

    @staticmethod
    def roughness(timeseries):
        """Sum of the squared difference of
        the normalized differences.
        """
        f = timeseries.diff()
        normed_f = (f - f.mean()) / f.std()
        return (normed_f.diff() ** 2).sum()

    @staticmethod
    def sha256(file_path):
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    @staticmethod
    def md5(file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


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
