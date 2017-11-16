# coding=utf8
"""
Copyright (C) 2016-2017 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

from __future__ import division
from __future__ import absolute_import
from datetime import datetime
import csv
import copy


class MassBal(object):
    """Follow-up the mass balance during the simulation run
    Intended use:
    at each record time, using write_values():
    averaged or cumulated values for the considered time difference are
    written to a CSV file
    """
    def __init__(self, file_name, rast_dom, start_time, temporal_type):
        self.dom = rast_dom
        self.start_time = start_time
        if temporal_type not in ['absolute', 'relative']:
            assert False, u"unknown temporal type <{}>".format(temporal_type)
        self.temporal_type = temporal_type
        # values to be written on each record time
        self.fields = ['sim_time',  # either timedelta or datetime
                       'avg_timestep', '#timesteps',
                       'boundary_vol', 'rain_vol', 'inf_vol',
                       'inflow_vol', 'losses_vol', 'drain_net_vol',
                       'domain_vol', 'created_vol',
                       '%error']
        # data written to file as one line
        self.line = dict.fromkeys(self.fields)
        # data collected during simulation
        self.sim_data = {'tstep': []}
        # set file name and create file
        self.file_name = self.set_file_name(file_name)
        self.create_file()
        # water volume in the domain
        self.old_dom_vol = None
        self.new_dom_vol = 0.

    def set_file_name(self, file_name):
        '''Generate output file name
        '''
        if not file_name:
            file_name = "{}_stats.csv".format(
                str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')))
        return file_name

    def create_file(self):
        '''Create a csv file and write headers
        '''
        with open(self.file_name, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()
        return self

    def read_dom_vol(self):
        """read new domain volume
        store the current volume as old domain volume
        """
        if self.old_dom_vol is None:
            self.old_dom_vol = self.dom.start_volume
        else:
            self.old_dom_vol = copy.copy(self.new_dom_vol)
        self.new_dom_vol = self.dom.water_volume()
        return self

    def add_value(self, key, value):
        '''add a value to sim_data
        '''
        assert key in self.sim_data, "unknown key!"
        self.sim_data[key].append(value)
        return self

    def write_values(self, sim_time):
        """Calculate statistics and write them to the file
        """
        if self.temporal_type == 'relative':
            self.line['sim_time'] = sim_time - self.start_time
        else:
            self.line['sim_time'] = sim_time

        # number of time-step during the interval is the number of records
        self.line['#timesteps'] = len(self.sim_data['tstep'])
        # average time-step calculation
        elapsed_time = sum(self.sim_data['tstep'])
        try:
            avg_timestep = elapsed_time / self.line['#timesteps']
            self.line['avg_timestep'] = '{:.3f}'.format(avg_timestep)
        except ZeroDivisionError:
            self.line['avg_timestep'] = '-'

        # domain volume
        self.read_dom_vol()
        self.line['domain_vol'] = '{:.3f}'.format(self.new_dom_vol)

        # sum of inflow (positive) / outflow (negative) volumes
        boundary_vol = self.dom.boundary_vol()
        self.line['boundary_vol'] = '{:.3f}'.format(boundary_vol)
        rain_vol = self.dom.rain_vol(sim_time)
        self.line['rain_vol'] = '{:.3f}'.format(rain_vol)
        inf_vol = - self.dom.inf_vol(sim_time)
        self.line['inf_vol'] = '{:.3f}'.format(inf_vol)
        inflow_vol = self.dom.inflow_vol(sim_time)
        self.line['inflow_vol'] = '{:.3f}'.format(inflow_vol)
        losses_vol = - self.dom.losses_vol(sim_time)
        self.line['losses_vol'] = '{:.3f}'.format(losses_vol)
        drain_net_vol = self.dom.ndrain_vol(sim_time)
        self.line['drain_net_vol'] = '{:.3f}'.format(drain_net_vol)

        # Computation error from array
        vol_error = self.dom.err_vol()
        self.line['created_vol'] = '{:.3f}'.format(vol_error)
        # Continuity error: part of volume change due to error
        dom_vol_diff = self.new_dom_vol - self.old_dom_vol
        if dom_vol_diff <= 0:
            self.line['%error'] = '-'
        else:
            self.line['%error'] = '{:.2%}'.format(vol_error/dom_vol_diff)

        # Add line to file
        with open(self.file_name, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(self.line)

        # empty dictionaries
        self.sim_data = {k: [] for k in self.sim_data.keys()}
        self.line = dict.fromkeys(self.line.keys())
        return self
