#! /usr/bin/python
# coding=utf8

# COPYRIGHT:    (C) 2015 by Laurent Courty
#
#               This program is free software under the GNU General Public
#               License (v3). Read the LICENCE file for details.

import math
import warnings
import numpy as np
import numexpr as ne
from grass.pygrass import raster, utils
import grass.script as grass


def solve_q(g, theta, q_n_im12, q_n_im32, q_n_ip12, hflow, Dt, 
            Dx, Dy, y_n_i, y_n_im1, nf):
    """
    calculate the flow at a considered cell face

    inputs:
    g: gravity constant
    theta: weighting factor
    q_n_im12: flow at the current face
    q_n_im32: fow at the precedent face
    q_n_ip12: flow at the following face
    hflow: flow depth
    Dt: time step
    Dx: length of cell in that direction
    y_n_i: water surface elevevation, center of current cell
    y_n_im1: water surface elevation, center of prev. cell
    
    output: new flow in m3/s at current face
    """

    # flow formula (formula #41 in almeida et al 2012)
    term_1 = theta * q_n_im12 + ((1 - theta) / 2) * (q_n_im32 + q_n_ip12)
    term_2 = g * hflow * (Dt / Dx) * (y_n_i - y_n_im1)
    term_3 = 1 + g * Dt * math.pow(nf,2) * math.fabs(q_n_im12) / math.pow(hflow, 7/3) 
    q_np1_im12 = (term_1 - term_2) / term_3
    
    return q_np1_im12 * Dy  # output in m3/s


def get_flow(z_grid_padded, n_grid_padded, depth_grid, hf_grid,
            flow_grid_padded, h_grid_np1_padded, flow_grid_np1,
            hf_min, Dt, Dx, Dy, g, theta):
    """
    Calculate the flow at the next time-step in all the numpy array 
    using a loop
    
    return a numpy array
    """

    for c, i in np.ndenumerate(depth_grid):
        x = c[1] # x coordinate of the cell
        y = c[0] # y coordinate of the cell
        xp = x + 1 # x coordinate of cell in padded grid
        yp = y + 1 # y coordinate of cell in padded grid

        # retrieve manning's n at current cell
        nf = n_grid_padded[yp, xp]
        
        # flow at prev. W face
        q_n_im32 = flow_grid_padded[yp, xp - 1]['W'] / Dy
        # flow at W face
        q_n_im12 = flow_grid_padded[yp, xp]['W'] / Dy
        # flow at E face, i.e at next cell W face
        q_n_ip12 = flow_grid_padded[yp, xp + 1]['W'] / Dy

        # flow at prev. S face
        q_n_jm32 = flow_grid_padded[yp + 1, xp]['S'] / Dx
        # flow at S face
        q_n_jm12 = flow_grid_padded[yp, xp]['S'] / Dx
        # flow at N face i.e at next cell S face
        q_n_jp12 = flow_grid_padded[yp - 1, xp]['S'] / Dx

        # retrieve water depth
        h_i = h_grid_np1_padded[yp, xp]
        h_im1 = h_grid_np1_padded[yp, xp - 1]
        h_jm1 = h_grid_np1_padded[yp + 1, xp]
        
        # retrieve altitude 
        z_i = z_grid_padded[yp, xp]
        z_im1 = z_grid_padded[yp, xp - 1]
        z_jm1 = z_grid_padded[yp + 1, xp]

        # calculate WSE
        y_i = z_i + h_i
        y_im1 = z_im1 + h_im1
        y_jm1 = z_jm1 + h_jm1

        # retrieve the flow height
        hflow_W = hf_grid[y, x]['W']
        hflow_S = hf_grid[y, x]['S']

        # W flow
        # calculate if not on first col (controled by bound. cond.)
        if not x == 0:
            # q set to zero if flow elevation is lower than threshold
            if hflow_W <= hf_min:
                Qnp1_i = 0
            else:
                Qnp1_i = solve_q(
                    g, theta, q_n_im12, q_n_im32, q_n_ip12,
                    hflow_W, Dt, Dx, Dy, y_i, y_im1, nf)
    
            # write flow results to result grid
            flow_grid_np1[y, x]['W'] = Qnp1_i

        # S flow
        # calculate if not on last row (controled by bound. cond.)
        if not y == depth_grid.shape[0]-1:
            # q set to zero if flow elevation is lower than threshold
            if hflow_S <= hf_min:
                Qnp1_j = 0
            else:
                Qnp1_j = solve_q(
                    g, theta, q_n_jm12, q_n_jm32, q_n_jp12,
                    hflow_S, Dt, Dy, Dx, y_i, y_jm1, nf)

            # write flow results to result grid
            flow_grid_np1[y, x]['S'] = Qnp1_j

    return flow_grid_np1


def flow_dir(arr_z_p):
    '''
    Return a "slope" array re presenting the flow direction for simple routing
    input: arr_z: terrain elevation, numpy array
    output: 
    '''

    Z = arr_z_p[1:-1, 1:-1]
    N = arr_z_p[0:-2, 1:-1]
    S = arr_z_p[2:, 1:-1]
    E = arr_z_p[1:-1, 2:]
    W = arr_z_p[1:-1, 0:-2]
    dN = Z - N
    dE = Z - E
    dS = Z - S
    dW = Z - W

    # return minimum neighbour
    arr_min = np.maximum(np.maximum(dN, dS), np.maximum(dE, dW))

    # assign values
    arr_slope = np.zeros(shape = Z.shape, dtype = np.string_)

    for c, cell in np.ndenumerate(arr_min):

        if cell <= 0:
           arr_slope[c] = '0'
        elif cell == dN[c]:
            arr_slope[c] = 'N'
        elif cell == dS[c]:
            arr_slope[c] = 'S'
        elif cell == dE[c]:
            arr_slope[c] = 'E'
        elif cell == dW[c]:
            arr_slope[c] = 'W'

    return arr_slope


def routing(velocity, width, depth):
    '''calculate a flow with a given velocity'''
    return width * depth * velocity

