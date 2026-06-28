from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from itzi.providers.domain_data import DomainData
from itzi.providers.memory_input import MemoryRasterInputProvider, TimedRasterSlice


@pytest.fixture
def domain_data() -> DomainData:
    return DomainData(
        north=30.0,
        south=0.0,
        east=40.0,
        west=0.0,
        rows=3,
        cols=4,
        crs_wkt="",
    )


@pytest.fixture
def simulation_times() -> dict[str, datetime]:
    start_time = datetime(2000, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(hours=6)
    return {"start_time": start_time, "end_time": end_time}


def make_config(
    domain_data: DomainData,
    simulation_times: dict[str, datetime],
    **overrides,
) -> dict:
    config = {
        "domain_data": domain_data,
        "simulation_start_time": simulation_times["start_time"],
        "simulation_end_time": simulation_times["end_time"],
    }
    config.update(overrides)
    return config


def test_provider_creation_with_empty_arrays(
    domain_data: DomainData, simulation_times: dict[str, datetime]
) -> None:
    provider = MemoryRasterInputProvider(make_config(domain_data, simulation_times))

    assert provider.get_domain_data() == domain_data

    array, start_time, end_time = provider.get_array("dem", simulation_times["start_time"])
    assert array is None
    assert start_time == simulation_times["start_time"]
    assert end_time == simulation_times["end_time"]


def test_get_array_static_returns_copy_and_simulation_bounds(
    domain_data: DomainData, simulation_times: dict[str, datetime]
) -> None:
    dem = np.full(domain_data.shape, 2.5, dtype=np.float32)
    provider = MemoryRasterInputProvider(
        make_config(domain_data, simulation_times, static_arrays={"dem": dem})
    )

    array, start_time, end_time = provider.get_array(
        "dem", simulation_times["start_time"] + timedelta(hours=2)
    )

    assert array is not None
    np.testing.assert_allclose(array, dem)
    assert start_time == simulation_times["start_time"]
    assert end_time == simulation_times["end_time"]

    array[0, 0] = 99.0
    array_again, _, _ = provider.get_array("dem", simulation_times["start_time"])
    assert array_again is not None
    assert array_again[0, 0] == pytest.approx(2.5)


def test_provider_copies_static_arrays_at_construction(
    domain_data: DomainData, simulation_times: dict[str, datetime]
) -> None:
    dem = np.full(domain_data.shape, 1.0, dtype=np.float32)
    provider = MemoryRasterInputProvider(
        make_config(domain_data, simulation_times, static_arrays={"dem": dem})
    )

    dem[0, 0] = 77.0

    array, _, _ = provider.get_array("dem", simulation_times["start_time"])
    assert array is not None
    assert array[0, 0] == pytest.approx(1.0)


def test_get_array_timed_returns_matching_slice_and_copy(
    domain_data: DomainData, simulation_times: dict[str, datetime]
) -> None:
    start_time = simulation_times["start_time"]
    first_rain = np.full(domain_data.shape, 10.0, dtype=np.float32)
    second_rain = np.full(domain_data.shape, 20.0, dtype=np.float32)

    provider = MemoryRasterInputProvider(
        make_config(
            domain_data,
            simulation_times,
            timed_arrays={
                "rain": [
                    TimedRasterSlice(
                        start_time=start_time,
                        end_time=start_time + timedelta(hours=1),
                        array=first_rain,
                    ),
                    TimedRasterSlice(
                        start_time=start_time + timedelta(hours=1),
                        end_time=start_time + timedelta(hours=3),
                        array=second_rain,
                    ),
                ]
            },
        )
    )

    first_array, first_start, first_end = provider.get_array(
        "rain", start_time + timedelta(minutes=30)
    )
    second_array, second_start, second_end = provider.get_array(
        "rain", start_time + timedelta(hours=1)
    )

    assert first_array is not None
    assert second_array is not None
    np.testing.assert_allclose(first_array, first_rain)
    np.testing.assert_allclose(second_array, second_rain)
    assert first_start == start_time
    assert first_end == start_time + timedelta(hours=1)
    assert second_start == start_time + timedelta(hours=1)
    assert second_end == start_time + timedelta(hours=3)

    second_array[0, 0] = 123.0
    second_array_again, _, _ = provider.get_array("rain", start_time + timedelta(hours=1))
    assert second_array_again is not None
    assert second_array_again[0, 0] == pytest.approx(20.0)


def test_provider_copies_timed_arrays_at_construction(
    domain_data: DomainData, simulation_times: dict[str, datetime]
) -> None:
    start_time = simulation_times["start_time"]
    rain = np.full(domain_data.shape, 3.0, dtype=np.float32)

    provider = MemoryRasterInputProvider(
        make_config(
            domain_data,
            simulation_times,
            timed_arrays={
                "rain": [
                    TimedRasterSlice(
                        start_time=start_time,
                        end_time=start_time + timedelta(hours=1),
                        array=rain,
                    )
                ]
            },
        )
    )

    rain[0, 0] = 42.0

    array, _, _ = provider.get_array("rain", start_time + timedelta(minutes=30))
    assert array is not None
    assert array[0, 0] == pytest.approx(3.0)


def test_get_array_returns_none_when_time_is_not_covered(
    domain_data: DomainData, simulation_times: dict[str, datetime]
) -> None:
    start_time = simulation_times["start_time"]
    provider = MemoryRasterInputProvider(
        make_config(
            domain_data,
            simulation_times,
            timed_arrays={
                "rain": [
                    TimedRasterSlice(
                        start_time=start_time,
                        end_time=start_time + timedelta(hours=1),
                        array=np.full(domain_data.shape, 1.0, dtype=np.float32),
                    )
                ]
            },
        )
    )

    array, slice_start, slice_end = provider.get_array("rain", start_time + timedelta(hours=4))

    assert array is None
    assert slice_start == start_time + timedelta(hours=1)
    assert slice_end == simulation_times["end_time"]


def test_rejects_invalid_simulation_time_bounds(
    domain_data: DomainData, simulation_times: dict[str, datetime]
) -> None:
    with pytest.raises(ValueError, match="simulation_start_time"):
        MemoryRasterInputProvider(
            {
                "domain_data": domain_data,
                "simulation_start_time": simulation_times["end_time"],
                "simulation_end_time": simulation_times["start_time"],
            }
        )


@pytest.mark.parametrize(
    ("config_key", "input_key"),
    [("static_arrays", "not_an_input"), ("timed_arrays", "also_not_an_input")],
)
def test_rejects_invalid_input_keys(
    domain_data: DomainData,
    simulation_times: dict[str, datetime],
    config_key: str,
    input_key: str,
) -> None:
    invalid_value = (
        {input_key: np.zeros(domain_data.shape, dtype=np.float32)}
        if config_key == "static_arrays"
        else {
            input_key: [
                TimedRasterSlice(
                    start_time=simulation_times["start_time"],
                    end_time=simulation_times["end_time"],
                    array=np.zeros(domain_data.shape, dtype=np.float32),
                )
            ]
        }
    )

    with pytest.raises(ValueError, match="invalid array keys"):
        MemoryRasterInputProvider(
            make_config(domain_data, simulation_times, **{config_key: invalid_value})
        )


def test_rejects_duplicate_keys_across_static_and_timed_arrays(
    domain_data: DomainData, simulation_times: dict[str, datetime]
) -> None:
    with pytest.raises(ValueError, match="duplicate keys"):
        MemoryRasterInputProvider(
            make_config(
                domain_data,
                simulation_times,
                static_arrays={"dem": np.zeros(domain_data.shape, dtype=np.float32)},
                timed_arrays={
                    "dem": [
                        TimedRasterSlice(
                            start_time=simulation_times["start_time"],
                            end_time=simulation_times["end_time"],
                            array=np.ones(domain_data.shape, dtype=np.float32),
                        )
                    ]
                },
            )
        )


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((3, 3), dtype=np.float32),
        np.zeros((3, 4, 1), dtype=np.float32),
    ],
)
def test_rejects_invalid_static_array_shape_or_dimensionality(
    domain_data: DomainData,
    simulation_times: dict[str, datetime],
    array: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="input array 'dem'"):
        MemoryRasterInputProvider(
            make_config(domain_data, simulation_times, static_arrays={"dem": array})
        )


def test_rejects_invalid_timed_array_shape(
    domain_data: DomainData, simulation_times: dict[str, datetime]
) -> None:
    with pytest.raises(ValueError, match="input array 'rain'"):
        MemoryRasterInputProvider(
            make_config(
                domain_data,
                simulation_times,
                timed_arrays={
                    "rain": [
                        TimedRasterSlice(
                            start_time=simulation_times["start_time"],
                            end_time=simulation_times["end_time"],
                            array=np.zeros((2, 4), dtype=np.float32),
                        )
                    ]
                },
            )
        )


def test_rejects_unsorted_timed_slices(
    domain_data: DomainData, simulation_times: dict[str, datetime]
) -> None:
    start_time = simulation_times["start_time"]

    with pytest.raises(ValueError, match="must be sorted"):
        MemoryRasterInputProvider(
            make_config(
                domain_data,
                simulation_times,
                timed_arrays={
                    "rain": [
                        TimedRasterSlice(
                            start_time=start_time + timedelta(hours=2),
                            end_time=start_time + timedelta(hours=3),
                            array=np.full(domain_data.shape, 2.0, dtype=np.float32),
                        ),
                        TimedRasterSlice(
                            start_time=start_time,
                            end_time=start_time + timedelta(hours=1),
                            array=np.full(domain_data.shape, 1.0, dtype=np.float32),
                        ),
                    ]
                },
            )
        )


def test_rejects_overlapping_timed_slices(
    domain_data: DomainData, simulation_times: dict[str, datetime]
) -> None:
    start_time = simulation_times["start_time"]

    with pytest.raises(ValueError, match="must not overlap"):
        MemoryRasterInputProvider(
            make_config(
                domain_data,
                simulation_times,
                timed_arrays={
                    "rain": [
                        TimedRasterSlice(
                            start_time=start_time,
                            end_time=start_time + timedelta(hours=2),
                            array=np.full(domain_data.shape, 1.0, dtype=np.float32),
                        ),
                        TimedRasterSlice(
                            start_time=start_time + timedelta(hours=1),
                            end_time=start_time + timedelta(hours=3),
                            array=np.full(domain_data.shape, 2.0, dtype=np.float32),
                        ),
                    ]
                },
            )
        )


def test_rejects_timed_slice_with_reversed_bounds(
    domain_data: DomainData, simulation_times: dict[str, datetime]
) -> None:
    with pytest.raises(ValueError, match="must have start_time before end_time"):
        MemoryRasterInputProvider(
            make_config(
                domain_data,
                simulation_times,
                timed_arrays={
                    "rain": [
                        TimedRasterSlice(
                            start_time=simulation_times["end_time"],
                            end_time=simulation_times["start_time"],
                            array=np.full(domain_data.shape, 1.0, dtype=np.float32),
                        )
                    ]
                },
            )
        )


def test_rejects_zero_length_timed_slice(
    domain_data: DomainData, simulation_times: dict[str, datetime]
) -> None:
    start_time = simulation_times["start_time"]

    with pytest.raises(ValueError, match="must have start_time before end_time"):
        MemoryRasterInputProvider(
            make_config(
                domain_data,
                simulation_times,
                timed_arrays={
                    "rain": [
                        TimedRasterSlice(
                            start_time=start_time,
                            end_time=start_time,
                            array=np.full(domain_data.shape, 1.0, dtype=np.float32),
                        )
                    ]
                },
            )
        )
