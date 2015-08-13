#! /usr/bin/python
# coding=utf8

"""
COPYRIGHT:    (C) 2015 by Laurent Courty

              This program is free software under the GNU General Public
              License (v3). Read the file Read the LICENCE file for details.
"""

import sys
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

    if hflow == 0:
        return 0
    else:
        # flow formula (formula #41 in almeida et al 2012)
        term_1 = theta * q_n_im12 + ((1 - theta) / 2) * (q_n_im32 + q_n_ip12)
        term_2 = g * hflow * (Dt / Dx) * (y_n_i - y_n_im1)
        term_3 = 1 + g * Dt * (nf*nf) * abs(q_n_im12) / pow(hflow, 7/3)
        q_np1_im12 = (term_1 - term_2) / term_3

        return q_np1_im12 * Dy  # output in m3/s


def apply_bc(BCi, z_grid_padded, depth_grid_padded,\
            flow_grid_padded, hf_grid_padded, n_grid_padded, \
            Dx, Dy, Dt, g, theta, hf_min):
    """Set the boundary conditions
    BCi: dictionnary of 1D arrays containing the boundary type of each side :
        1: closed: z=huge, depth=0, q = 0
        2: open: z=neighbour, depth=neighbour, v=neighbour
        3: fixed_h: z=neighbour,  wse=user defined, q=variable
        4: note defined yet. could be:
            slope: z=exension of 2 neighb. cells slope, depth=neighbour, q=variable
    
    input:
        z_grid_padded: 2D array of terrain elevation
        depth_grid_padded: 2D array of water depth
        flow_grid_padded: 2D array of flows at W and S
        hf_grid_padded: 2D array of flow height at W and S
    """

    # define domain from the padded array (removing padding)
    flow_grid = flow_grid_padded[1:-1, 1:-1]
    z_grid = z_grid_padded[1:-1, 1:-1]
    depth_grid = depth_grid_padded[1:-1, 1:-1]
    hf_grid = hf_grid_padded[1:-1, 1:-1]
    n_grid = n_grid_padded[1:-1, 1:-1]

    # define 1D boudary arrays outside of domain,
    # i.e where should be applied boundary conditions
    W_BC_z = z_grid_padded[1:-1, 0]
    E_BC_z = z_grid_padded[1:-1, -1]
    N_BC_z = z_grid_padded[0, 1:-1]
    S_BC_z = z_grid_padded[-1, 1:-1]
    
    W_BC_h = depth_grid_padded[1:-1, 0]
    E_BC_h = depth_grid_padded[1:-1, -1]
    N_BC_h = depth_grid_padded[0, 1:-1]
    S_BC_h = depth_grid_padded[-1, 1:-1]

    W_BC_n = n_grid_padded[1:-1, 0]
    E_BC_n = n_grid_padded[1:-1, -1]
    N_BC_n = n_grid_padded[0, 1:-1]
    S_BC_n = n_grid_padded[-1, 1:-1]

    W_BC_q = flow_grid[:, 0]['W']
    E_BC_q = flow_grid_padded[1:-1, -1]['W']
    N_BC_q = flow_grid_padded[0, 1:-1]['S']
    S_BC_q = flow_grid[-1, :]['S']

    W_BC_hf = hf_grid[:, 0]['W']
    E_BC_hf = hf_grid_padded[1:-1, -1]['W']
    N_BC_hf = hf_grid_padded[0, 1:-1]['S']
    S_BC_hf = hf_grid[-1, :]['S']

    ##############
    # W Boundary #
    ##############
    BCw = BCi['W']

    # type 1
    W_BC_coord_t1 = np.where(BCw['t'] == 1)
    W_BC_q[W_BC_coord_t1] = 0

    # type 2
    W_BC_coord_t2 = np.where(BCw['t'] == 2)
    nW_flow = flow_grid[W_BC_coord_t2, 1]['W'] / hf_grid[W_BC_coord_t2, 1]['W'] * W_BC_hf[W_BC_coord_t2]
    W_BC_q[W_BC_coord_t2] = np.where(np.logical_or(hf_grid[W_BC_coord_t2, 1]['W'] < hf_min, depth_grid[W_BC_coord_t2, 0] < hf_min), 0, nW_flow)

    # type 3
    W_BC_coord_t3 = np.where(BCw['t'] == 3)
    # set terrain elevation
    W_BC_z[W_BC_coord_t3] = z_grid[W_BC_coord_t3, 0]
    # water depth
    W_BC_h[W_BC_coord_t3] = BCw[W_BC_coord_t3]['v'] - W_BC_z[W_BC_coord_t3]
    # Friction
    W_BC_n[W_BC_coord_t3] = n_grid_padded[W_BC_coord_t3, 1]
    # flow at the boundary
    q_n_im12 = W_BC_q[W_BC_coord_t3]
    # flow inside the domain
    q_n_ip12 = flow_grid[W_BC_coord_t3, 1]['W']
    # flow outside the domain
    q_n_im32 = 0
    # wse in the domain
    y_n_i = depth_grid[W_BC_coord_t3, 0] + z_grid[W_BC_coord_t3, 0]
    # wse outside the domain = user-defined wse
    y_n_im1 = BCw[W_BC_coord_t3]['v']
    # solve flow
    solve_q_np = np.vectorize(solve_q)
    if W_BC_n[W_BC_coord_t3].size < 1:
        W_BC_q[W_BC_coord_t3] = 0
    else:
        W_BC_q[W_BC_coord_t3] = solve_q_np(g, theta, q_n_im12, q_n_im32,\
            q_n_ip12, W_BC_hf[W_BC_coord_t3], \
            Dt, Dx, Dy, y_n_i, y_n_im1,
            W_BC_n[W_BC_coord_t3])


    ##############
    # E Boundary #
    ##############
    BCe = BCi['E']

    # type 1
    E_BC_coord_t1 = np.where(BCe['t'] == 1)
    E_BC_q[E_BC_coord_t1] = 0

    # type 2
    E_BC_coord_t2 = np.where(BCe['t'] == 2)
    nE_flow = flow_grid[E_BC_coord_t2, -1]['W'] / hf_grid[E_BC_coord_t2, -1]['W'] * E_BC_hf[E_BC_coord_t2]
    E_BC_q[E_BC_coord_t2] = np.where(
                                hf_grid[E_BC_coord_t2, -1]['W'] == 0,
                                0, nE_flow)

    # type 3
    E_BC_coord_t3 = np.where(BCe['t'] == 3)
    #~ # for testing purpose, to be corrected
    #~ # the intended behaviour is:
    #~ # 1 - assign the value to the outside cells
    #~ # 2 - calculate the flow through the boudary
    depth_grid[E_BC_coord_t3, -1] = BCe[E_BC_coord_t3]['v'] - z_grid[E_BC_coord_t3, -1]

    # set terrain elevation
    #~ z_E = z_grid[E_BC_coord_t3, -1]
    #~ # water depth = user-defined wse - dem calculated at the previous step
    #~ h_E = BCe[E_BC_coord_t3]['v'] - z_E
    #~ # Friction =  neighbour
    #~ n_E = n_grid[E_BC_coord_t3, -1]
    #~ print 'n_E', n_E
    #~ # flow at the boundary
    #~ q_n_im12 = E_BC_q[E_BC_coord_t3]
    #~ print 'q_n_im12', q_n_im12
    #~ # flow inside the domain
    #~ q_n_im32 = flow_grid[E_BC_coord_t3, -1]['W']
    #~ print 'q_n_im32', q_n_im32
    #~ # flow outside the domain
    #~ q_n_ip12 = 0
    #~ # wse in the domain
    #~ y_n_im1 = depth_grid[E_BC_coord_t3, -1] + z_grid[E_BC_coord_t3, -1]
    #~ # wse outside the domain = user-defined wse
    #~ y_n_i = BCe[E_BC_coord_t3]['v']
    #~ # hflow. z is equal on both sides
    #~ E_BC_hf[E_BC_coord_t3] = np.maximum(y_n_im1, y_n_i) - z_E
    #~ print 'hf', E_BC_hf[E_BC_coord_t3]
    #~ # solve flow
    #~ solve_q_np = np.vectorize(solve_q)
    #~ if E_BC_n[E_BC_coord_t3].size < 1:
        #~ E_BC_q[E_BC_coord_t3] = 0
    #~ else:
        #~ E_BC_q[E_BC_coord_t3] = solve_q_np(g, theta, q_n_im12, q_n_im32,\
            #~ q_n_ip12, E_BC_hf[E_BC_coord_t3], \
            #~ Dt, Dx, Dy, y_n_i, y_n_im1, n_E)


    ##############
    # N boundary #
    ##############
    BCn = BCi['N']

    # type 1
    N_BC_coord_t1 = np.where(BCn['t'] == 1)
    N_BC_q[N_BC_coord_t1] = 0

    # type 2
    N_BC_coord_t2 = np.where(BCn['t'] == 2)
    #~ N_BC_z[N_BC_coord_t2] = z_grid[0, N_BC_coord_t2]
    #~ N_BC_h[N_BC_coord_t2] = depth_grid[0, N_BC_coord_t2]
    nN_flow = flow_grid[0, N_BC_coord_t2]['S'] / hf_grid[0, N_BC_coord_t2]['S'] * N_BC_hf[N_BC_coord_t2]
    N_BC_q[N_BC_coord_t2] = np.where(
                                hf_grid[0, N_BC_coord_t2]['S'] == 0,
                                0, nN_flow)

    # type 3
    N_BC_coord_t3 = np.where(BCn['t'] == 3)
    # for testing purpose, to be corrected
    # the intended behaviour is:
    # 1 - assign the value to the outside cells
    # 2 - calculate the flow through the boudary
    depth_grid[0, N_BC_coord_t3] = BCn[N_BC_coord_t3]['v'] - z_grid[0, N_BC_coord_t3]


    ##############
    # S Boundary #
    ##############
    BCs = BCi['S']

    # type 1
    S_BC_coord_t1 = np.where(BCs['t'] == 1)
    S_BC_q[S_BC_coord_t1] = 0

    # type 2
    S_BC_coord_t2 = np.where(BCs['t'] == 2)
    nS_flow = flow_grid[-2, S_BC_coord_t2]['S'] / hf_grid[-2, S_BC_coord_t2]['S'] * S_BC_hf[S_BC_coord_t2]
    S_BC_q[S_BC_coord_t2] = np.where(hf_grid[-2, S_BC_coord_t2]['S'] == 0, 0, nS_flow)

    # type 3
    S_BC_coord_t3 = np.where(BCs['t'] == 3)
    # set terrain elevation
    S_BC_z[S_BC_coord_t3] = z_grid[-1, S_BC_coord_t3]
    # water depth
    S_BC_h[S_BC_coord_t3] = BCs[S_BC_coord_t3]['v'] - S_BC_z[S_BC_coord_t3]
    # Friction
    nf = n_grid_padded[-1, S_BC_coord_t3]
    # flow at the boundary
    q_n_jm12 = S_BC_q[S_BC_coord_t3]
    # flow inside the domain
    q_n_jp12 = flow_grid[-1, S_BC_coord_t3]['S']
    # flow outside the domain
    q_n_jm32 = 0
    # wse in the domain
    y_n_j = depth_grid[-1, S_BC_coord_t3] + z_grid[-1, S_BC_coord_t3]
    # wse outside the domain
    y_n_jm1 = BCs[S_BC_coord_t3]['v']
    # solve flow
    solve_q_np = np.vectorize(solve_q)
    if S_BC_hf[S_BC_coord_t3].size == 0:
        S_BC_q[S_BC_coord_t3] = 0
    else:
        S_BC_q[S_BC_coord_t3] = solve_q_np(g, theta, q_n_jm12, q_n_jm32,\
                    q_n_jp12, S_BC_hf[S_BC_coord_t3], \
                    Dt, Dx, Dy, y_n_j, y_n_jm1, nf)


    ##########################
    # record boundary volume #
    ##########################
    bound_vol = Dt * math.fsum([np.sum(W_BC_q), np.sum(E_BC_q),
                             np.sum(N_BC_q), np.sum(S_BC_q)])
    

    return bound_vol


