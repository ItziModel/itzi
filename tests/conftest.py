#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

import os
import pytest
import pandas as pd
from grass_session import Session as GrassSession
import grass.script as gscript

from itzi import SimulationRunner


@pytest.fixture(scope="session")
def test_data_path():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, 'test_data')


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
def grass_5by5(grass_xy_session):
    """Create a square, 5 by 5 domain.
    """
    # Create new mapset
    gscript.run_command('g.mapset', mapset='5by5', flags='c')

    # Set a 5x5 region
    gscript.run_command('g.region', res=10, s=0, w=0, e=50, n=50)
    region = gscript.parse_command('g.region', flags='pg')
    assert region["cells"] == '25'
    # DEM
    gscript.mapcalc('z=0')
    univar_z = gscript.parse_command('r.univar', map='z', flags='g')
    assert int(univar_z['min']) == 0
    assert int(univar_z['max']) == 0
    # Manning
    gscript.mapcalc('n=0.005')
    univar_n = gscript.parse_command('r.univar', map='n', flags='g')
    assert float(univar_n['min']) == 0.005
    assert float(univar_n['max']) == 0.005
    # Start depth 10cm
    gscript.write_command('v.in.ascii', input='-',
                          stdin='2.5|2.5',
                          output='start_h')
    gscript.run_command('v.to.rast', input='start_h',
                        output='start_h', type='point',
                        use='val', value=0.2)
    gscript.run_command('r.null', map='start_h', null=0)
    univar_start_h = gscript.parse_command('r.univar', map='start_h', flags='g')
    assert float(univar_start_h['min']) == 0
    assert float(univar_start_h['max']) == 0.2
    return None


@pytest.fixture(scope="session")
def grass_5by5_sim(grass_5by5, test_data_path):
    """
    """
    current_mapset = gscript.read_command('g.mapset', flags='p').rstrip()
    assert current_mapset == '5by5'
    config_file = os.path.join(test_data_path, '5by5.ini')
    sim_runner = SimulationRunner(need_grass_session=False)
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
    sim_runner = SimulationRunner(need_grass_session=False)
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
    sim_runner = SimulationRunner(need_grass_session=False)
    assert isinstance(sim_runner, SimulationRunner)
    sim_runner.initialize(config_file)
    sim_runner.run().finalize()
    return sim_runner
    