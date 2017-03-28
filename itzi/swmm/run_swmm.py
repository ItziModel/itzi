#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run a swmm simulation
"""

from swmm import Swmm5

input_file = '/home/laurent/Datos_geo/kolkata/drainage/100_FE.INP'
report_file = '/home/laurent/Datos_geo/kolkata/drainage/100_FE.report'
output_file = '/home/laurent/Datos_geo/kolkata/drainage/100_FE.out'

swmm5 = Swmm5()

swmm5.c_run(input_file=input_file,
            report_file=report_file,
            output_file=output_file)
