# coding=utf8
"""
Copyright (C) 2017 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

# Default values #

# Threshold for determining the flow equation (m)
HFMIN = 0.005
# Coefficient applied to time-step calculation
CFL = 0.7
# Damping weighting coefficient
THETA = 0.9
# Gravity constant (m/s²)
G = 9.80665
# Routing velocity (m/s)
VROUTING = 0.1
# Maximum time-step duration (s)
DTMAX = 5.
# Maximum slope (m/m). Not used
SLMAX = 0.1
# Hydrology time step (s)
DTINF = 60.
# maximum Froude number. Not used.
FRMAX = 1.


# Verbosity levels #
SUPER_QUIET = 0
QUIET = 1
MESSAGE = 2
VERBOSE = 3
DEBUG = 4
