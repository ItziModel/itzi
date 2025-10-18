"""Benchmark for the tutorial simulation."""

import os

import pytest

from itzi import SimulationRunner
from itzi.profiler import profile_context

from tests.grass_tests.test_tutorial import (  # noqa: F401
    itzi_tutorial,
    grass_tutorial_session,
    tutorial_test_file,
)


def run_simulation(config_file, test_data_temp_path):
    """The function to benchmark.
    This function runs the tutorial simulation.
    """
    profile_path = os.path.join(test_data_temp_path, "tutorial_profile.txt")
    with profile_context(profile_path):
        sim_runner = SimulationRunner()
        sim_runner.initialize(config_file)
        sim_runner.run().finalize()


@pytest.mark.usefixtures("itzi_tutorial")
def test_benchmark_tutorial(benchmark, test_data_path, test_data_temp_path):
    """Run the benchmark for the tutorial"""
    os.environ["GRASS_OVERWRITE"] = "1"
    config_file = os.path.join(test_data_path, "tutorial_files", "tutorial.ini")
    # benchmark the run_simulation function
    benchmark(run_simulation, config_file, test_data_temp_path)
