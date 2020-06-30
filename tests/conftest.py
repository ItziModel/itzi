#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

import os
import math
import zipfile
import hashlib
from collections import namedtuple
from configparser import ConfigParser

import pytest
import pandas as pd
import numpy as np
import requests
from grass_session import Session as GrassSession
import grass.script as gscript

from itzi import BmiItzi
from itzi import SimulationRunner
from itzi import InfGreenAmpt
from itzi import RasterDomain


# URL of zip file with test data
EA_TESTS_URL = 'https://web.archive.org/web/20200527005028/http://evidence.environment-agency.gov.uk/FCERM/Libraries/FCERM_Project_Documents/Benchmarking_Model_Data.sflb.ashx'
DATA_SHA256 = 'dd91fda6f049df34428b0cacdb21badcd9f0d5e92613e3541e8540a6b92cfda7'

MapInfo = namedtuple('MapInfo', ['name', 'start', 'end', 'value'])


def sha256(file_path):
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


@pytest.fixture(scope="session")
def test_data_path():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, 'test_data')


@pytest.fixture(scope="session")
def ea_test_files(test_data_path):
    """Download and extract the EA tests main file.
    """
    file_name = 'benchmarking_model_data.zip'
    file_path = os.path.join(test_data_path, file_name)
    # Check if the file exists and has the right hash
    try:
        assert sha256(file_path) == DATA_SHA256
    except:
        # Download the file
        with open(file_path, "wb") as data_file:
            response = requests.get(EA_TESTS_URL)
            # write to file
            data_file.write(response.content)

    # Unzip the main file
    unzip_path = os.path.join(test_data_path, 'ea_test_files')
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_path)
    return unzip_path


@pytest.fixture(scope="session")
def grass_xy_session(tmpdir_factory):
    """Create a GRASS session in a new XY location and PERMANENT mapset
    """
    tmpdir = str(tmpdir_factory.mktemp("grassdata"))
    print(tmpdir)
    grass_session = GrassSession()
    grass_session.open(gisdb=tmpdir,
                       location='xy',
                       mapset=None,  # PERMANENT
                       create_opts='XY',
                       loadlibs=True)
    os.environ['GRASS_VERBOSE'] = '1'
    # os.environ['ITZI_VERBOSE'] = '4'
    yield grass_session
    grass_session.close()


@pytest.fixture(scope="session")
def ea_test8a(grass_xy_session, ea_test_files, test_data_path):
    """Create the GRASS env for ea test 8a.
    """
    # Unzip the file
    file_path = os.path.join(ea_test_files, 'Test8A_dataset_2010.zip')
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(ea_test_files)
    unzip_path = os.path.join(ea_test_files, 'Test8A dataset 2010')
    # Create new mapset
    mapset_name = 'ea8a'
    gscript.run_command('g.mapset', mapset=mapset_name, flags='c')
    # Define the region
    region = gscript.parse_command('g.region', res=2,
                                   s=664408, w=263976,
                                   e=264940, n=664808, flags='g')
    assert int(region['rows']) == 200
    assert int(region['cols']) == 482
    # DEM
    dem_path = os.path.join(unzip_path, 'Test8DEM.asc')
    gscript.run_command('r.in.gdal', input=dem_path, output='dem50cm')
    gscript.run_command('r.resamp.stats', input='dem50cm', output='dem2m')
    univar_dem = gscript.parse_command('r.univar', map='dem2m', flags='g')
    assert int(univar_dem['null_cells']) == 0
    # Manning
    road_path = os.path.join(unzip_path, 'Test8RoadPavement.asc')
    gscript.run_command('r.in.gdal', input=road_path, output='road50cm')
    gscript.mapcalc('n=if(isnull(road50cm), 0.05, 0.02)')
    univar_n = gscript.parse_command('r.univar', map='n', flags='g')
    assert float(univar_n['min']) == 0.02
    assert float(univar_n['max']) == 0.05
    assert int(univar_n['null_cells']) == 0
    # Rainfall
    rain_maps = [MapInfo('rain0', 0, 60, 0),
                 MapInfo('rain1', 60, 240, 400),
                 MapInfo('rain2', 240, 18000, 0)]
    for rain_map in rain_maps:
        gscript.mapcalc(f'{rain_map.name}={rain_map.value}')
        gscript.run_command('t.register', type='raster',  maps=rain_map.name,
                            start=rain_map.start, end=rain_map.end, unit='seconds')
    gscript.run_command('t.create', type='strds', temporaltype='relative',
                            output='rainfall', title='rainfall', descr='rainfall')
    gscript.run_command('t.register', type='raster', input='rainfall',
                        maps=[i.name for i in rain_maps])
    # Output points #
    stages_path = os.path.join(unzip_path, 'Test8Output.csv')
    gscript.run_command('v.in.ascii', input=stages_path, output='output_points',
                        format='point', sep='comma', skip=1, cat=1, x=2, y=3)
    # Point inflow #
    # Read the inflow point location
    point_path = os.path.join(unzip_path, 'Test8A-inflow-location.csv')
    gscript.run_command('v.in.ascii', input=point_path, output='inflow_point',
                        skip=1, format='point', sep='comma')
    # Transform to raster using the CSV flux values, and register map in STRDS
    inflow_path = os.path.join(test_data_path, 'EA_test_8', 'a', 'point-inflow.csv')
    df_flow = pd.read_csv(inflow_path)
    for idx, row in df_flow.iterrows():
        gscript.run_command('v.to.rast', input='inflow_point',
                            output=f'inflow{idx}', type='point',
                            use='val', value=row.flux)
        gscript.run_command('r.null', map=f'inflow{idx}', null=0)
        gscript.run_command('t.register', type='raster',  maps=f'inflow{idx}',
                            start=int(row.start), end=int(row.end), unit='seconds')
    gscript.run_command('t.create', type='strds', temporaltype='relative',
                        output='inflow', title='inflow', descr='inflow')
    gscript.run_command('t.register', type='raster', input='inflow',
                        maps=[f'inflow{i}' for i, r in df_flow.iterrows()])
    # Linear interpolation of the STRDS
    gscript.run_command('t.rast.gapfill', input='inflow', basename='inflow', quiet=True)
    return None


@pytest.fixture(scope="session")
def ea_test8a_reference(test_data_path):
    """Take the results from LISFLOOD-FP as reference.
    """
    col_names = ['Time (min)'] + list(range(1,10))
    file_path = os.path.join(test_data_path, 'EA_test_8', 'a', 'ea2dt8a.stage')
    df_ref = pd.read_csv(file_path, sep='    ',
                         header=15, index_col=0, engine='python',
                         names=col_names)
    # Convert to minutes
    df_ref.index /= 60.
    # round entries
    df_ref.index = np.round(df_ref.index, 1)
    return df_ref


@pytest.fixture(scope="session")
def ea_test8a_sim(ea_test8a, test_data_path):
    """
    """
    current_mapset = gscript.read_command('g.mapset', flags='p').rstrip()
    assert current_mapset == 'ea8a'
    config_file = os.path.join(test_data_path, 'EA_test_8', 'a', 'ea2dt8a.ini')
    sim_runner = SimulationRunner()
    sim_runner.initialize(config_file)
    sim_runner.run().finalize()
    return sim_runner


@pytest.fixture(scope="session")
def ea_test8b(grass_xy_session, ea_test_files):
    """Create the GRASS env for ea test 8a.
    """
    # Unzip the file
    file_path = os.path.join(ea_test_files, 'Test8B_dataset_2010.zip')
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(ea_test_files)
    unzip_path = os.path.join(ea_test_files, 'Test8B dataset 2010')
    # Create new mapset
    mapset_name = 'ea8b'
    gscript.run_command('g.mapset', mapset=mapset_name, flags='c')
    # Define the region
    region = gscript.parse_command('g.region', res=2,
                                   s=664408, w=263976,
                                   e=264940, n=664808, flags='g')
    assert int(region['rows']) == 200
    assert int(region['cols']) == 482
    # DEM
    dem_path = os.path.join(unzip_path, 'Test8DEM.asc')
    gscript.run_command('r.in.gdal', input=dem_path, output='dem50cm')
    gscript.run_command('r.resamp.stats', input='dem50cm', output='dem2m')
    univar_dem = gscript.parse_command('r.univar', map='dem2m', flags='g')
    assert int(univar_dem['null_cells']) == 0
    # Buildings
    buildings_path = os.path.join(unzip_path, 'Test8Buildings.asc')
    gscript.run_command('r.in.gdal', input=buildings_path, output='buildings')
    gscript.mapcalc('dem2m_buildings=if(isnull(buildings), dem2m, dem2m+5)')
    # Manning
    road_path = os.path.join(unzip_path, 'Test8RoadPavement.asc')
    gscript.run_command('r.in.gdal', input=road_path, output='road50cm')
    gscript.mapcalc('n=if(isnull(road50cm), 0.05, 0.02)')
    univar_n = gscript.parse_command('r.univar', map='n', flags='g')
    assert float(univar_n['min']) == 0.02
    assert float(univar_n['max']) == 0.05
    assert int(univar_n['null_cells']) == 0
    # Output points #
    stages_path = os.path.join(unzip_path, 'Test8Output.csv')
    gscript.run_command('v.in.ascii', input=stages_path, output='output_points',
                        format='point', sep='comma', skip=1, cat=1, x=2, y=3)
    # Manhole location
    gscript.write_command('v.in.ascii', input='-',
                          stdin='264895|664747',
                          output='manhole_location')
    return None


@pytest.fixture(scope="session")
def ea_test8b_reference(test_data_path):
    """Take the results from xpstorm as reference.
    """
    file_path = os.path.join(test_data_path, 'EA_test_8', 'b', 'xpstorm.csv')
    series = pd.read_csv(file_path, index_col=0, squeeze=True, names=['time', 'value'])
    series.name = 'xpstorm'
    # transform index to timedelta and round to 1 sec
    series.index = pd.to_timedelta(series.index, unit='m').round(freq='s')
    # remove duplicate indices
    series.loc[~series.index.duplicated(keep='first')]
    # resample to 1 sec with linear interpolation
    series_hr = series.resample(rule='s').interpolate(method='time')
    # resample to 30 sec
    series_30s = series_hr.resample('30s').asfreq()
    series_30s.name = 'reference'
    return series_30s


@pytest.fixture(scope="session")
def ea_test8b_sim(ea_test8b, test_data_path):
    """
    """
    current_mapset = gscript.read_command('g.mapset', flags='p').rstrip()
    assert current_mapset == 'ea8b'
    inp_file = os.path.join(test_data_path, 'EA_test_8', 'b',
                            'test8b_drainage_ponding.inp')
    config_dict = {'time': {'duration': '03:20:00', 'record_step': '00:00:30'},
                   'input': {'dem': 'dem2m_buildings', 'friction': 'n'},
                   'output': {'prefix': 'out', 'values': 'h, drainage_stats'},
                   'options': {'theta': 0.7, 'cfl': 0.5},
                   'drainage': {'swmm_inp': inp_file, 'orifice_coeff': 1}}
    parser = ConfigParser()
    parser.read_dict(config_dict)
    conf_file = os.path.join(test_data_path, 'EA_test_8', 'b',
                            'ea2dt8b.ini')
    with open(conf_file, 'w') as f:
        parser.write(f)
    sim_runner = SimulationRunner()
    sim_runner.initialize(conf_file)
    sim_runner.run().finalize()
    return sim_runner


@pytest.fixture(scope="session")
def grass_5by5(grass_xy_session, test_data_path):
    """Create a square, 5 by 5 domain.
    """
    # Create new mapset
    gscript.run_command('g.mapset', mapset='5by5', flags='c')
    # Create 3by5 named region
    gscript.run_command('g.region', res=10, s=10, n=40, w=0, e=50, save='3by5')
    region = gscript.parse_command('g.region', flags='pg')
    assert int(region["cells"]) == 15
    # Create raster for mask (do not apply mask)
    gscript.run_command('g.region', res=10, s=0, n=50, w=10, e=40)
    region = gscript.parse_command('g.region', flags='pg')
    assert int(region["cells"]) == 15
    gscript.mapcalc('5by3=1')
    # Set a 5x5 region
    gscript.run_command('g.region', res=10, s=0, w=0, e=50, n=50)
    region = gscript.parse_command('g.region', flags='pg')
    assert int(region["cells"]) == 25
    # DEM
    gscript.mapcalc('z=0')
    univar_z = gscript.parse_command('r.univar', map='z', flags='g')
    assert int(univar_z['min']) == 0
    assert int(univar_z['max']) == 0
    # Manning
    gscript.mapcalc('n=0.05')
    univar_n = gscript.parse_command('r.univar', map='n', flags='g')
    assert float(univar_n['min']) == 0.05
    assert float(univar_n['max']) == 0.05
    # Start depth
    gscript.write_command('v.in.ascii', input='-',
                          stdin='25|25',
                          output='start_h')
    gscript.run_command('v.to.rast', input='start_h',
                        output='start_h', type='point',
                        use='val', value=0.2)
    gscript.run_command('r.null', map='start_h', null=0)
    univar_start_h = gscript.parse_command('r.univar', map='start_h', flags='g')
    assert float(univar_start_h['min']) == 0
    assert float(univar_start_h['max']) == 0.2
    # Symmetry control points
    control_points = os.path.join(test_data_path, '5by5', 'control_points.csv')
    gscript.run_command('v.in.ascii', input=control_points,
                        output='control_points',
                        separator='comma')
    return None


@pytest.fixture(scope="session")
def grass_5by5_sim(grass_5by5, test_data_path):
    """
    """
    current_mapset = gscript.read_command('g.mapset', flags='p').rstrip()
    assert current_mapset == '5by5'
    config_file = os.path.join(test_data_path, '5by5', '5by5.ini')
    sim_runner = SimulationRunner()
    assert isinstance(sim_runner, SimulationRunner)
    sim_runner.initialize(config_file)
    sim_runner.run().finalize()
    return sim_runner


@pytest.fixture(scope="session")
def mcdo_norain_reference(test_data_path):
    """
    """
    file_path = os.path.join(test_data_path, 'McDonald_long_channel_wo_rain', 'mcdo_norain.csv')
    return pd.read_csv(file_path)


@pytest.fixture(scope="session")
def grass_mcdo_norain(grass_xy_session, test_data_path):
    """Create a domain for MacDonald 1D solution long channel without rain.
    Delestre, O., Lucas, C., Ksinant, P.-A., Darboux, F., Laguerre, C., Vo, T.-N.-T., … Cordier, S. (2013).
    SWASHES: a compilation of shallow water analytic solutions for hydraulic and environmental studies.
    International Journal for Numerical Methods in Fluids, 72(3), 269–300. https://doi.org/10.1002/fld.3741
    """
    data_dir = os.path.join(test_data_path, 'McDonald_long_channel_wo_rain')
    dem_path = os.path.join(data_dir, 'dem.asc')
    bctype_path = os.path.join(data_dir, 'bctype.asc')
    inflow_path = os.path.join(data_dir, 'q.asc')
    points_path = os.path.join(data_dir, 'axis_points.json')
    # Create new mapset
    gscript.run_command('g.mapset', mapset='mcdo_norain', flags='c')
    # Load raster data
    gscript.run_command('r.in.gdal', input=dem_path, output='dem')
    gscript.run_command('r.in.gdal', input=bctype_path, output='bctype')
    gscript.run_command('r.in.gdal', input=inflow_path, output='inflow')
    # Generate Manning map
    gscript.run_command('g.region', raster='dem')
    region = gscript.parse_command('g.region', flags='pg')
    assert int(region["cells"]) == 600
    gscript.mapcalc('n=0.033')
    # Load axis points vector
    gscript.run_command('v.in.ogr', input=points_path, output='axis_points', flags='o')
    return None


@pytest.fixture(scope="session")
def grass_mcdo_norain_sim(grass_mcdo_norain, test_data_path):
    """
    """
    current_mapset = gscript.read_command('g.mapset', flags='p').rstrip()
    assert current_mapset == 'mcdo_norain'
    # accessible_mapsets = gscript.read_command('g.mapsets', flags='p').rstrip()
    config_file = os.path.join(test_data_path, 'McDonald_long_channel_wo_rain', 'mcdo_norain.ini')
    sim_runner = SimulationRunner()
    assert isinstance(sim_runner, SimulationRunner)
    sim_runner.initialize(config_file)
    sim_runner.run().finalize()
    return sim_runner


@pytest.fixture(scope="session")
def mcdo_rain_reference(test_data_path):
    """
    """
    file_path = os.path.join(test_data_path, 'McDonald_long_channel_rain', 'mcdo_rain.csv')
    return pd.read_csv(file_path)


@pytest.fixture(scope="session")
def grass_mcdo_rain(grass_xy_session, test_data_path):
    """Create a domain for MacDonald 1D solution long channel with rain.
    Delestre, O., Lucas, C., Ksinant, P.-A., Darboux, F., Laguerre, C., Vo, T.-N.-T., … Cordier, S. (2013).
    SWASHES: a compilation of shallow water analytic solutions for hydraulic and environmental studies.
    International Journal for Numerical Methods in Fluids, 72(3), 269–300. https://doi.org/10.1002/fld.3741
    """
    data_dir = os.path.join(test_data_path, 'McDonald_long_channel_rain')
    dem_path = os.path.join(data_dir, 'dem.asc')
    bctype_path = os.path.join(data_dir, 'bctype.asc')
    inflow_path = os.path.join(data_dir, 'q.asc')
    points_path = os.path.join(data_dir, 'axis_points.json')
    # Create new mapset
    gscript.run_command('g.mapset', mapset='mcdo_rain', flags='c')
    # Load raster data
    gscript.run_command('r.in.gdal', input=dem_path, output='dem')
    gscript.run_command('r.in.gdal', input=bctype_path, output='bctype')
    gscript.run_command('r.in.gdal', input=inflow_path, output='inflow')
    # Create Manning map
    gscript.run_command('g.region', raster='dem')
    region = gscript.parse_command('g.region', flags='pg')
    assert int(region["cells"]) == 600
    gscript.mapcalc('n=0.033')
    # Create rain map
    gscript.mapcalc('rain=3600')
    # Load axis points vector
    gscript.run_command('v.in.ogr', input=points_path, output='axis_points', flags='o')
    return None


@pytest.fixture(scope="session")
def grass_mcdo_rain_sim(grass_mcdo_rain, test_data_path):
    """
    """
    current_mapset = gscript.read_command('g.mapset', flags='p').rstrip()
    assert current_mapset == 'mcdo_rain'
    # accessible_mapsets = gscript.read_command('g.mapsets', flags='p').rstrip()
    # print(accessible_mapsets)
    config_file = os.path.join(test_data_path, 'McDonald_long_channel_rain', 'mcdo_rain.ini')
    sim_runner = SimulationRunner()
    assert isinstance(sim_runner, SimulationRunner)
    sim_runner.initialize(config_file)
    sim_runner.run().finalize()
    return sim_runner


@pytest.fixture(scope="session")
def bmi_object(grass_5by5, test_data_path):
    itzi_bmi = BmiItzi()
    conf_file = os.path.join(test_data_path, '5by5', '5by5.ini')
    itzi_bmi.initialize(conf_file)
    return itzi_bmi


@pytest.fixture(scope="session")
def total_infiltration(infiltration_parameters):
    # Estimate total Green-Ampt infiltration using formula in:
    # Barry, D. A., Parlange, J. Y., & Bakhtyar, R. (2010).
    # Discussion of “application of a nonstandard explicit integration to
    # solve green and ampt infiltration equation”
    # by D.R. Mailapalli, W.W. Wallender, R. Singh, and N.S. Raghuwanshi.
    # Journal of Hydrologic Engineering, 15(7), 595–596.
    # https://doi.org/10.1061/(ASCE)HE.1943-5584.0000164
    time = 24 * 3600
    porosity, cap_pressure, conductivity = infiltration_parameters
    sorptivity = math.sqrt(2 * conductivity * cap_pressure * porosity)
    total_inf = sorptivity * math.sqrt(time) + 2 * conductivity * time / 3
    return total_inf


@pytest.fixture(scope="session")
def infiltration_parameters():
    # Silt loam from Rawls, Brakensiek and Miller (1983)
    total_porosity = 0.501
    effective_porosity = 0.486
    capillary_pressure = 16.68 / 100  # cm to m
    hydraulic_conductivity = 0.65 / (100 * 3600)  # cm/h to m/s
    water_depth = 0.3
    soil_moisture = 0.3
    return effective_porosity, capillary_pressure, hydraulic_conductivity

@pytest.fixture(scope="session")
def infiltration_sim(infiltration_parameters):
    array_shape = (3, 3)
    cell_shape = (5,5)
    dt = 1  # in seconds
    sim_duration = 24 * 3600
    porosity, cap_pressure, conductivity = infiltration_parameters
    mask = np.full(shape=array_shape, fill_value=False, dtype=np.bool_)
    arr_depth = np.full(shape=array_shape, fill_value=0.3)
    arr_por = np.full(shape=array_shape, fill_value=porosity)
    arr_cond = np.full(shape=array_shape, fill_value=conductivity)
    arr_cap_pressure = np.full(shape=array_shape, fill_value=cap_pressure)
    raster_domain = RasterDomain(dtype=np.float32, cell_shape=cell_shape,
                                 arr_mask=mask)
    raster_domain.update_array('h', arr_depth)
    raster_domain.update_array('effective_porosity', arr_por)
    raster_domain.update_array('capillary_pressure', arr_cap_pressure)
    raster_domain.update_array('hydraulic_conductivity', arr_cond)
    inf_sim = InfGreenAmpt(raster_domain=raster_domain, dt_inf=dt)
    elapsed_time = 0
    while elapsed_time < sim_duration:
        inf_sim.step()
        elapsed_time += inf_sim._dt
    # print(inf_sim.infiltration_amount)
    # print(arr_depth)
    theoretical_depth = arr_depth - inf_sim.infiltration_amount
    print(theoretical_depth)
    print(raster_domain.get_array('h'))
    return inf_sim.infiltration_amount.max()
