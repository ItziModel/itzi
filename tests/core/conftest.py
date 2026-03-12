from collections import namedtuple


import pytest
import numpy as np

from itzi.providers.domain_data import DomainData


Domain5by5Data = namedtuple(
    "Domain5by5Data",
    [
        "domain_data",  # DomainData instance
        "arr_dem_flat",  # DEM with z=0
        "arr_dem_high",  # DEM with z=132
        "arr_n",  # Manning's n = 0.05
        "arr_start_h",  # Initial depth: 0.2 at center [2,2]
        "arr_start_wse",  # Initial WSE: 132.2 at center [2,2]
        "arr_mask",  # All False (no mask)
        "arr_bctype",  # Boundary condition type for open boundaries
        "arr_rain",  # Rainfall in m/s
        "arr_inf",  # Infiltration in m/s
        "arr_loss",  # Losses in m/s
        "arr_inflow",  # Inflow in m/s
    ],
)


@pytest.fixture(scope="module")
def domain_5by5() -> Domain5by5Data:
    """Create a 5x5 domain with all base arrays.

    This fixture provides the foundational data for all 5x5 tests:
    - 5x5 grid at 10m resolution
    - Domain extends: north=50, south=0, east=50, west=0
    - Total area: 2500 m²
    """
    # Domain dimensions
    rows, cols = 5, 5
    north, south, east, west = 50.0, 0.0, 50.0, 0.0

    # Create DomainData
    domain_data = DomainData(
        north=north, south=south, east=east, west=west, rows=rows, cols=cols, crs_wkt=""
    )

    # DEM arrays
    arr_dem_flat = np.zeros(domain_data.shape, dtype=np.float32)
    arr_dem_high = np.full(domain_data.shape, 132.0, dtype=np.float32)

    # Manning's n
    arr_n = np.full(domain_data.shape, 0.05, dtype=np.float32)

    # Initial water depth: 0.2m at center cell [2, 2], 0 elsewhere
    arr_start_h = np.zeros(domain_data.shape, dtype=np.float32)
    arr_start_h[2, 2] = 0.2

    # Initial water surface elevation: 132.2m at center cell [2, 2]
    # (high DEM + 0.2m depth)
    arr_start_wse = np.zeros(domain_data.shape, dtype=np.float32)
    arr_start_wse[2, 2] = 132.2

    # No mask - whole domain active
    arr_mask = np.full(domain_data.shape, False, dtype=np.bool_)

    # Boundary condition type: 2 (open) at all 16 edge cells
    arr_bctype = np.zeros(domain_data.shape, dtype=np.float32)
    # Top and bottom rows
    arr_bctype[0, :] = 2
    arr_bctype[4, :] = 2
    # Left and right columns (excluding corners already set)
    arr_bctype[:, 0] = 2
    arr_bctype[:, 4] = 2

    # Rate arrays in m/s
    # Rainfall: 10 mm/h = 10/(1000*3600) m/s
    arr_rain = np.full(domain_data.shape, 10.0 / (1000 * 3600), dtype=np.float32)
    # Infiltration: 2 mm/h = 2/(1000*3600) m/s
    arr_inf = np.full(domain_data.shape, 2.0 / (1000 * 3600), dtype=np.float32)
    # Losses: 1.5 mm/h = 1.5/(1000*3600) m/s
    arr_loss = np.full(domain_data.shape, 1.5 / (1000 * 3600), dtype=np.float32)
    # Inflow: 0.1 m/s (already in m/s)
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
