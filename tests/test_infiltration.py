#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test the infiltration module.
"""

import math
from collections import namedtuple
import pytest
import numpy as np
from scipy.special import lambertw

from itzi import InfGreenAmpt
from itzi import RasterDomain


def ga_serrano2001(inf_params):
    """Lambert W solution presented in:
    Serrano, S. E. (2001). Explicit solution to Green and Ampt infiltration
    equation. Journal of Hydrologic Engineering, 6(4), 336–340.
    https://doi.org/10.1061/(ASCE)1084-0699(2001)6:4(336)"""
    inf_init = 0
    available_porosity = max(0, inf_params.eff_porosity - inf_params.init_wat_content)
    total_head = inf_params.cap_pressure + inf_params.pond_depth
    a = total_head * available_porosity
    b = inf_params.hydr_cond * inf_params.time + inf_init - a * math.log(inf_init + a)
    term1 = -(b + a) / a
    total_inf = -a - a * lambertw(-(math.pow(math.e, term1) / a), -1)
    if total_inf.imag == 0:
        return total_inf.real
    else:
        assert False


def ga_barry2009(inf_params):
    """Estimate total Green-Ampt infiltration using formula in:
    Barry, D. A., Parlange, J. Y., & Bakhtyar, R. (2010).
    Discussion of “application of a nonstandard explicit integration to
    solve Green and Ampt infiltration equation”
    by D.R. Mailapalli, W.W. Wallender, R. Singh, and N.S. Raghuwanshi.
    Journal of Hydrologic Engineering, 15(7), 595–596.
    https://doi.org/10.1061/(ASCE)HE.1943-5584.0000164"""
    available_porosity = max(0, inf_params.eff_porosity - inf_params.init_wat_content)
    total_head = inf_params.cap_pressure + inf_params.pond_depth
    sorptivity = math.sqrt(2 * inf_params.hydr_cond * total_head * available_porosity)
    total_inf = (
        sorptivity * math.sqrt(inf_params.time) + 2 * inf_params.hydr_cond * inf_params.time / 3
    )
    return total_inf


@pytest.fixture(scope="session")
def infiltration_parameters():
    param_names = [
        "pond_depth",
        "time",
        "eff_porosity",
        "init_wat_content",
        "cap_pressure",
        "hydr_cond",
    ]
    InfParameters = namedtuple("InfParameters", param_names)
    # Silt loam from Rawls, Brakensiek and Miller (1983)
    # total_porosity = 0.501
    inf_params = InfParameters(
        pond_depth=0.4,
        time=24 * 3600,
        eff_porosity=0.486,
        init_wat_content=0.3,
        cap_pressure=16.68 / 100,  # cm to m
        hydr_cond=0.65 / (100 * 3600),  # cm/h to m/s
    )
    return inf_params


@pytest.fixture(scope="session")
def reference_infiltration(infiltration_parameters):
    # total_inf = ga_barry2009(infiltration_parameters)
    total_inf = ga_serrano2001(infiltration_parameters)
    return total_inf


@pytest.fixture(scope="session")
def infiltration_sim(infiltration_parameters):
    array_shape = (3, 3)
    cell_shape = (5, 5)
    dt = 10  # in seconds
    dtype = np.float32
    inf_params = infiltration_parameters
    mask = np.full(shape=array_shape, fill_value=False, dtype=np.bool_)
    arr_depth = np.full(shape=array_shape, fill_value=inf_params.pond_depth)
    arr_por = np.full(shape=array_shape, fill_value=inf_params.eff_porosity)
    arr_cond = np.full(shape=array_shape, fill_value=inf_params.hydr_cond)
    arr_cap_pressure = np.full(shape=array_shape, fill_value=inf_params.cap_pressure)
    arr_water_content = np.full(shape=array_shape, fill_value=inf_params.init_wat_content)
    raster_domain = RasterDomain(dtype=dtype, cell_shape=cell_shape, arr_mask=mask)
    raster_domain.update_array("water_depth", arr_depth)
    raster_domain.update_array("effective_porosity", arr_por)
    raster_domain.update_array("capillary_pressure", arr_cap_pressure)
    raster_domain.update_array("hydraulic_conductivity", arr_cond)
    raster_domain.update_array("soil_water_content", arr_water_content)
    inf_sim = InfGreenAmpt(raster_domain=raster_domain, dt_inf=dt)
    elapsed_time = 0
    while elapsed_time < inf_params.time:
        inf_sim.step()
        elapsed_time += inf_sim._dt
    return inf_sim.infiltration_amount.max()


def test_infiltration(reference_infiltration, infiltration_sim):
    inf_err = abs(reference_infiltration - infiltration_sim)
    percent_error = inf_err / reference_infiltration
    # Accept less than 1% error
    assert percent_error < 0.01
