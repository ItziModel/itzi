#! /usr/bin/python
# coding=utf8

# COPYRIGHT:    (C) 2015 by Laurent Courty
#
#               This program is free software under the GNU General Public
#               License (v3). Read the file LICENCE for details.

import numpy as np

def pad_array(arr):
    """pad aarray with one cell
    """
    arr_p = np.pad(arr, 1, 'constant', constant_values = 0)
    
    arr = arr_p[1:-1,1:-1]
    
    return arr, arr_p
