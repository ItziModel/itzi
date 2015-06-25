#! /usr/bin/python
# coding=utf8

"""
COPYRIGHT:    (C) 2015 by Laurent Courty

               This program is free software under the GNU General Public
               License (v3). Read the LICENCE file for details.
"""

import math
import numpy as np


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
    
    arr_q_n_i = flow_grid_padded['W']
    arr_q_n_j = flow_grid_padded['S']

    arr_hflow_W = hf_grid['W']
    arr_hflow_S = hf_grid['S']

    ymax = depth_grid.shape[0]
    xmax = depth_grid.shape[1]
    for y in range(ymax):
        for x in range(xmax):
            xp = x + 1 # x coordinate of cell in padded grid
            yp = y + 1 # y coordinate of cell in padded grid

            # retrieve manning's n at current cell
            nf = n_grid_padded[yp, xp]

            # flow at prev. W face
            q_n_im32 = arr_q_n_i[yp, xp - 1] / Dy
            # flow at W face
            q_n_im12 = arr_q_n_i[yp, xp] / Dy
            # flow at E face, i.e at next cell W face
            q_n_ip12 = arr_q_n_i[yp, xp + 1] / Dy

            # flow at prev. S face
            q_n_jm32 = arr_q_n_j[yp + 1, xp] / Dx
            # flow at S face
            q_n_jm12 = arr_q_n_j[yp, xp] / Dx
            # flow at N face i.e at next cell S face
            q_n_jp12 = arr_q_n_j[yp - 1, xp] / Dx

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
            hflow_W = arr_hflow_W[y, x]
            hflow_S = arr_hflow_S[y, x]

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
