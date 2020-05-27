# coding=utf8
"""
Copyright (C) 2015-2020 Laurent Courty

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
from configparser import ConfigParser
from datetime import datetime, timedelta

import itzi.messenger as msgr
from itzi.const import DefaultValues


class ConfigReader():
    """Parse the configuration file and check validity of given options
    """
    def __init__(self, filename):
        self.config_file = filename
        # default values to be passed to simulation
        self.__set_defaults()
        # read entry values
        self.set_entry_values()

    def __set_defaults(self):
        """Set dictionaries of default values to be passed to simulation"""
        k_raw_input_times = ['start_time', 'end_time',
                             'duration', 'record_step']
        self.ga_list = ['effective_porosity', 'capillary_pressure',
                        'hydraulic_conductivity']
        k_input_map_names = ['dem', 'friction', 'start_h', 'start_y',
                             'rain', 'inflow', 'bcval', 'bctype',
                             'infiltration', 'losses'] + self.ga_list
        k_output_map_names = ['h', 'wse', 'v', 'vdir', 'qx', 'qy', 'fr',
                              'boundaries', 'infiltration', 'rainfall',
                              'inflow', 'losses', 'drainage_stats',
                              'verror']
        self.drainage_params = {'swmm_inp': None, 'output': None,
                                'orifice_coeff': DefaultValues.ORIFICE_COEFF,
                                'free_weir_coeff': DefaultValues.FREE_WEIR_COEFF,
                                'submerged_weir_coeff': DefaultValues.SUBMERGED_WEIR_COEFF}
        self.sim_param = {'hmin': DefaultValues.HFMIN, 'cfl': DefaultValues.CFL,
                          'theta': DefaultValues.THETA, 'g': DefaultValues.G,
                          'vrouting': DefaultValues.VROUTING, 'dtmax': DefaultValues.DTMAX,
                          'slmax': DefaultValues.SLMAX, 'dtinf': DefaultValues.DTINF,
                          'inf_model': None}
        self.grass_mandatory = ['grassdata', 'location', 'mapset']
        k_grass_params = self.grass_mandatory + ['region', 'mask', 'grass_bin']
        self.raw_input_times = dict.fromkeys(k_raw_input_times)
        self.output_map_names = dict.fromkeys(k_output_map_names)
        self.input_map_names = dict.fromkeys(k_input_map_names)
        self.grass_params = dict.fromkeys(k_grass_params)
        self.out_prefix = f"itzi_results_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        self.stats_file = None
        return self

    def set_entry_values(self):
        """Read and check entry values
        """
        # read file and populate dictionaries
        self.read_param_file()
        # process inputs times
        self.sim_times = SimulationTimes(self.raw_input_times)
        # check if mandatory parameters are present
        self.check_mandatory()
        # check coherence of infiltrations entries
        self.check_inf_maps()
        # check the sanity of simulation parameters
        self.check_sim_params()
        # check the sanity of GRASS parameters
        self.check_grass_params()
        return self

    def read_param_file(self):
        """Read the parameter file and populate the relevant dictionaries
        """
        self.out_values = []
        # read the file
        params = ConfigParser(allow_no_value=True)
        f = params.read(self.config_file)
        if not f:
            msgr.fatal(u"File <{}> not found".format(self.config_file))
        # populate dictionaries using loops instead of using update() method
        # in order to not add invalid key
        for k in self.raw_input_times:
            if params.has_option('time', k):
                self.raw_input_times[k] = params.get('time', k)
        for k in self.sim_param:
            if params.has_option('options', k):
                self.sim_param[k] = params.getfloat('options', k)
        for k in self.grass_params:
            if params.has_option('grass', k):
                self.grass_params[k] = params.get('grass', k)
        # check for deprecated input names
        if params.has_option('input', "drainage_capacity"):
            msgr.warning(u"'drainage_capacity' is deprecated. "
                         u"Use 'losses' instead.")
            self.input_map_names['losses'] = params.get('input', "drainage_capacity")
        if params.has_option('input', "effective_pororosity"):
            msgr.warning(u"'effective_pororosity' is deprecated. "
                         u"Use 'effective_porosity' instead.")
            self.input_map_names['effective_porosity'] = params.get('input', "effective_pororosity")
        # search for valid inputs
        for k in self.input_map_names:
            if params.has_option('input', k):
                self.input_map_names[k] = params.get('input', k)

        # drainage parameters
        for k in self.drainage_params:
            if params.has_option('drainage', k):
                if k in ['swmm_inp', 'output']:
                    self.drainage_params[k] = params.get('drainage', k)
                else:
                    self.drainage_params[k] = params.getfloat('drainage', k)
        # statistic file
        if params.has_option('statistics', 'stats_file'):
            self.stats_file = params.get('statistics', 'stats_file')
        else:
            self.stats_file = None
        # output maps
        if params.has_option('output', 'prefix'):
            self.out_prefix = params.get('output', 'prefix')
        if params.has_option('output', 'values'):
            out_values = params.get('output', 'values').split(',')
            self.out_values = [e.strip() for e in out_values]
            # check for deprecated values
            if 'drainage_cap' in self.out_values and 'losses' not in self.out_values:
                msgr.warning(u"'drainage_cap' is deprecated. "
                             u"Use 'losses' instead.")
                self.out_values.append('losses')
        self.generate_output_name()
        return self

    def generate_output_name(self):
        """Generate the name of the strds
        """
        for v in self.out_values:
            if v in self.output_map_names:
                self.output_map_names[v] = '{}_{}'.format(self.out_prefix, v)
        return self

    def check_sim_params(self):
        """Check if the simulations parameters are positives and valid
        """
        for k, v in self.sim_param.items():
            if k == 'theta':
                if not 0 <= v <= 1:
                    msgr.fatal(u"{} value must be between 0 and 1".format(k))
            elif k == 'inf_model':
                continue
            else:
                if not v > 0:
                    msgr.fatal(u"{} value must be positive".format(k))

    def check_grass_params(self):
        """Check if all grass params are presents if one is given
        """
        grass_any = any(self.grass_params[i] for i in self.grass_mandatory)
        grass_all = all(self.grass_params[i] for i in self.grass_mandatory)
        if grass_any and not grass_all:
            msgr.fatal(u"{} are mutualy inclusive".format(self.grass_mandatory))
        return self

    def check_inf_maps(self):
        """check coherence of input infiltration maps
        set infiltration model type
        """
        inf_k = 'infiltration'
        # if at least one Green-Ampt parameters is present
        ga_any = any(self.input_map_names[i] for i in self.ga_list)
        # if all Green-Ampt parameters are present
        ga_all = all(self.input_map_names[i] for i in self.ga_list)
        # verify parameters
        if not self.input_map_names[inf_k] and not ga_any:
            self.sim_param['inf_model'] = 'null'
        elif self.input_map_names[inf_k] and not ga_any:
            self.sim_param['inf_model'] = 'constant'
        elif self.input_map_names[inf_k] and ga_any:
            msgr.fatal(u"Infiltration model incompatible with user-defined rate")
        # check if all maps for Green-Ampt are presents
        elif ga_any and not ga_all:
            msgr.fatal(u"{} are mutualy inclusive".format(self.ga_list))
        elif ga_all and not self.input_map_names[inf_k]:
            self.sim_param['inf_model'] = 'green-ampt'
        return self

    def check_mandatory(self):
        """check if mandatory parameters are present
        """
        if not all([self.input_map_names['dem'],
                    self.input_map_names['friction'],
                    self.sim_times.record_step]):
            msgr.fatal(u"inputs <dem>, <friction> and "
                       u"<record_step> are mandatory")

    def display_sim_param(self):
        """Display simulation parameters if verbose
        """
        inter_txt = u"#"*50
        txt_template = u"{:<24s} {}"
        msgr.verbose(u"{}".format(inter_txt))
        msgr.verbose(u"Input maps:")
        for k, v in self.input_map_names.items():
            msgr.verbose(txt_template.format(k, v))
        msgr.verbose(u"{}".format(inter_txt))
        msgr.verbose(u"Output maps:")
        for k, v in self.output_map_names.items():
            msgr.verbose(txt_template.format(k, v))
        msgr.verbose(u"{}".format(inter_txt))
        msgr.verbose(u"Simulation parameters:")
        for k, v in self.sim_param.items():
            msgr.verbose(txt_template.format(k, v))
        # simulation times
        msgr.verbose(u"{}".format(inter_txt))
        msgr.verbose(u"Simulation times:")
        txt_start_time = self.sim_times.start.isoformat(" ").split(".")[0]
        txt_end_time = self.sim_times.end.isoformat(" ").split(".")[0]
        msgr.verbose(txt_template.format('start', txt_start_time))
        msgr.verbose(txt_template.format('end', txt_end_time))
        msgr.verbose(txt_template.format('duration', self.sim_times.duration))
        msgr.verbose(txt_template.format('record_step', self.sim_times.record_step))
        msgr.verbose(u"{}".format(inter_txt))


class SimulationTimes():
    """Store the information about simulation start & end time and duration
    """
    def __init__(self, raw_input_times):
        self.read_simulation_times(raw_input_times)

    def read_simulation_times(self, raw_input_times):
        """Read a given dictionary of input times.
        Check the coherence of the input and store it in the object
        """
        self.raw_duration = raw_input_times['duration']
        self.raw_start = raw_input_times['start_time']
        self.raw_end = raw_input_times['end_time']
        self.raw_record_step = raw_input_times['record_step']

        self.date_format = '%Y-%m-%d %H:%M'
        self.rel_err_msg = u"{}: invalid format (should be HH:MM:SS)"
        self.abs_err_msg = u"{}: invalid format (should be yyyy-mm-dd HH:MM)"

        # check if the given times are coherent
        self.check_combination()

        # transform duration and record_step to timedelta object
        self.duration = self.read_timedelta(self.raw_duration)
        self.record_step = self.read_timedelta(self.raw_record_step)

        # transform start and end to datetime object
        self.start = self.read_datetime(self.raw_start)
        self.end = self.read_datetime(self.raw_end)

        # check coherence of the properties
        self.check_coherence()

        # make sure everything went fine
        assert isinstance(self.end, datetime)
        assert isinstance(self.start, datetime)
        assert isinstance(self.duration, timedelta)
        assert self.end >= self.start
        assert self.duration == (self.end - self.start)

        return self

    def str_to_timedelta(self, inp_str):
        """Takes a string in the form HH:MM:SS
        and return a timedelta object
        """
        data = inp_str.split(":")
        hours = int(data[0])
        minutes = int(data[1])
        seconds = int(data[2])
        if hours < 0:
            raise ValueError
        if not 0 <= minutes <= 59 or not 0 <= seconds <= 59:
            raise ValueError
        obj_dt = timedelta(hours=hours,
                           minutes=minutes,
                           seconds=seconds)
        return obj_dt

    def check_combination(self):
        """Verifies if the given input times combination is valid.
        Sets temporal type.
        """
        comb_err_msg = (u"accepted combinations: "
                        u"{d} alone, {s} and {d}, "
                        u"{s} and {e}").format(d='duration',
                                               s='start_time',
                                               e='end_time')
        b_dur = (self.raw_duration and
                 not self.raw_start and
                 not self.raw_end)
        b_start_dur = (self.raw_start and
                       self.raw_duration and
                       not self.raw_end)
        b_start_end = (self.raw_start and self.raw_end and
                       not self.raw_duration)
        if not (b_dur or b_start_dur or b_start_end):
            msgr.fatal(comb_err_msg)
        # if only duration is given, temporal type is relative
        if b_dur:
            self.temporal_type = 'relative'
        else:
            self.temporal_type = 'absolute'
        return self

    def read_timedelta(self, string):
        """Try to transform string in timedelta object.
        If it fail, return an error message
        If string is None, return None
        """
        if string:
            try:
                return self.str_to_timedelta(string)
            except ValueError:
                msgr.fatal(self.rel_err_msg.format(string))
        else:
            return None

    def read_datetime(self, string):
        """Try to transform string in datetime object.
        If it fail, return an error message
        If string is None, return None
        """
        if string:
            try:
                return datetime.strptime(string, self.date_format)
            except ValueError:
                msgr.fatal(self.abs_err_msg.format(string))
        else:
            return None

    def check_coherence(self):
        """Sets end or duration if not given
        Verifies if end is superior to starts
        """
        if self.start is None:
            self.start = datetime.min
        if self.end is None:
            self.end = self.start + self.duration
        if self.start >= self.end:
            msgr.fatal("Simulation duration must be positive")
        if self.duration is None:
            self.duration = self.end - self.start
