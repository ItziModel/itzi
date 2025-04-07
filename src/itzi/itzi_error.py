# coding=utf8
"""
Copyright (C) 2015-2017  Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""


class ItziError(Exception):
    """General error class"""
    pass


class NullError(ItziError):
    """Raised when null values is detected in simulation"""
    pass


class DtError(ItziError):
    """Error related to time-step calculation
    """
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


class ItziFatal(ItziError):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)
