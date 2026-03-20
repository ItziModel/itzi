from collections import namedtuple

import pytest
import numpy as np


Domain5by5Data = namedtuple(
    "Domain5by5Data",
    [
        "domain_data",
        "arr_dem_flat",
        "arr_dem_high",
        "arr_n",
        "arr_start_h",
        "arr_start_wse",
        "arr_mask",
        "arr_bctype",
        "arr_rain",
        "arr_inf",
        "arr_loss",
        "arr_inflow",
    ],
)


@pytest.fixture(scope="module")
def domain_5by5() -> Domain5by5Data:
    """Create a 5x5 domain with all base arrays."""
    rows, cols = 5, 5
    north, south, east, west = 50.0, 0.0, 50.0, 0.0

    from itzi.providers.domain_data import DomainData

    domain_data = DomainData(
        north=north, south=south, east=east, west=west, rows=rows, cols=cols, crs_wkt=""
    )

    arr_dem_flat = np.zeros(domain_data.shape, dtype=np.float32)
    arr_dem_high = np.full(domain_data.shape, 132.0, dtype=np.float32)
    arr_n = np.full(domain_data.shape, 0.05, dtype=np.float32)

    arr_start_h = np.zeros(domain_data.shape, dtype=np.float32)
    arr_start_h[2, 2] = 0.2

    arr_start_wse = np.zeros(domain_data.shape, dtype=np.float32)
    arr_start_wse[2, 2] = 132.2

    arr_mask = np.full(domain_data.shape, False, dtype=np.bool_)

    arr_bctype = np.zeros(domain_data.shape, dtype=np.float32)
    arr_bctype[0, :] = 2
    arr_bctype[4, :] = 2
    arr_bctype[:, 0] = 2
    arr_bctype[:, 4] = 2

    arr_rain = np.full(domain_data.shape, 10.0 / (1000 * 3600), dtype=np.float32)
    arr_inf = np.full(domain_data.shape, 2.0 / (1000 * 3600), dtype=np.float32)
    arr_loss = np.full(domain_data.shape, 1.5 / (1000 * 3600), dtype=np.float32)
    arr_inflow = np.full(domain_data.shape, 0.1, dtype=np.float32)

    return Domain5by5Data(
        domain_data=domain_data,
        arr_dem_flat=arr_dem_flat,
        arr_dem_high=arr_dem_high,
        arr_n=arr_n,
        arr_start_h=arr_start_h,
        arr_start_wse=arr_start_wse,
        arr_mask=arr_mask,
        arr_bctype=arr_bctype,
        arr_rain=arr_rain,
        arr_inf=arr_inf,
        arr_loss=arr_loss,
        arr_inflow=arr_inflow,
    )
