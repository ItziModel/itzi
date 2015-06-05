#! /usr/bin/python
# coding=utf8

############################################################################
#
# COPYRIGHT:    (C) 2015 by Laurent Courty
#
#               This program is free software under the GNU General Public
#               License (v3). Read the file Read the file LICENCE for details.
#
#############################################################################

import sys
import math
import numpy as np

import hydro

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
    for coord, bcell in np.ndenumerate(BCi['W']):
        if bcell['t'] == 1:  # Boundary type 1
            #~ W_BC_z[coord] = np.finfo(np.float32)
            #~ W_BC_h[coord] = 0
            W_BC_q = 0
        elif bcell['t'] == 2:  # Boundary type 2
            nW_flow = flow_grid[coord, 1]['W'] / hf_grid[coord, 1]['W'] * W_BC_hf[coord]
            if hf_grid[coord, 1]['W'] < hf_min or depth_grid[coord, 0] < hf_min:
                W_BC_q[coord] = 0
            else:
                W_BC_q[coord] = nW_flow

        elif bcell['t'] == 3:  # Boundary type 3
            W_BC_z[coord] = z_grid[coord, 0]  # terrain elevation
            # depth = user_wse - z
            W_BC_h[coord] = bcell['v'] - W_BC_z[coord]

            nf = n_grid_padded[coord, 1]

            # solve flow
            if W_BC_hf[coord] <= hf_min:
                W_BC_q[coord] = 0
            else:
                # flow at the boundary
                q_n_im12 = W_BC_q[coord]
                # flow inside the domain
                q_n_ip12 = flow_grid[coord, 1]['W']
                # flow outside the domain
                q_n_im32 = 0
                 # wse in the domain
                y_n_i = depth_grid[coord, 0] + z_grid[coord, 0]
                # wse outside the domain = user-defined wse
                y_n_im1 = bcell['v']
                # solve flow
                W_BC_q[coord] = hydro.solve_q(g, theta, q_n_im12, q_n_im32,\
                    q_n_ip12, W_BC_hf[coord], \
                    Dt, Dx, Dy, y_n_i, y_n_im1, nf)

        elif bcell['t'] == 4:  # Boundary type 4
            pass
        else:
            print 'warning: unknown boundary type'
        
    
    ##############
    # E Boundary #
    ##############
    for coord, bcell in np.ndenumerate(BCi['E']):
        if bcell['t'] == 1:  # Boundary type 1
            #~ E_BC_z[coord] = np.finfo(np.float32)
            #~ E_BC_h[coord] = 0
            E_BC_q[coord] = 0
        elif bcell['t'] == 2:  # Boundary type 2
            E_BC_z[coord] = z_grid[coord, -1]
            E_BC_h[coord] = depth_grid[coord, -1]
            if hf_grid[coord, -1]['W'] == 0:
                E_BC_q[coord] = 0
            else:
                E_BC_q[coord] = flow_grid[coord, -1]['W'] / hf_grid[coord, -1]['W'] * E_BC_hf[coord]
        elif bcell['t'] == 3:  # Boundary type 3
            #~ pass

            # for testing purpose, to be corrected
            # the intended behaviour is:
            # 1 - assign the value to the outside cells
            # 2 - calculate the flow through the boudary
            depth_grid[coord, -1] = bcell['v'] - z_grid[coord, -1]

            #~ # terrain elevation
            #~ E_BC_z[coord] = z_grid[coord, -1]  
            #~ # depth = user_wse - z
            #~ E_BC_h[coord] = bcell['v'] - E_BC_z[coord]
#~ 
            #~ nf = n_grid_padded[coord, -1] # Manning's n
#~ 
            #~ # solve flow
            #~ if E_BC_hf[coord] <= hf_min:
                #~ E_BC_q[coord] = 0
            #~ else:
                #~ # flow at the boundary
                #~ q_n_im12 = E_BC_q[coord]
                #~ # flow inside the domain
                #~ q_n_im32 = flow_grid[coord, -1]['W']
                #~ # flow outside the domain
                #~ q_n_ip12 = 0
                #~ # wse inside the domain
                #~ y_n_im1 = depth_grid[coord, -1] + z_grid[coord, -1]
                #~ # wse outside the domain
                #~ y_n_i = bcell['v']
                #~ # solve flow
                #~ E_BC_q[coord] = solve_q(g, theta, q_n_im12, q_n_im32,\
                    #~ q_n_ip12, E_BC_hf[coord], \
                    #~ Dt, Dx, Dy, y_n_i, y_n_im1, nf)
                #~ print  'q_n_im32', q_n_im32, 'q_n_im12', q_n_im12, 'q_n_ip12', q_n_ip12, 'y_n_im1', y_n_im1, 'y_n_i', y_n_i, 'q', E_BC_q[coord], 'hf', E_BC_hf[coord]

        elif bcell == 4:  # Boundary type 4
            pass
        else:
            print 'warning: unknown boundary type'            
    
    
    
    ##############
    # N boundary #
    ##############
    for coord, bcell in np.ndenumerate(BCi['N']):
        if bcell['t'] == 1:  # Boundary type 1
            #~ N_BC_z[coord] = np.finfo(np.float32)
            #~ N_BC_h[coord] = 0
            N_BC_q[coord] = 0
        elif bcell['t'] == 2:  # Boundary type 2
            N_BC_z[coord] = z_grid[0, coord]
            N_BC_h[coord] = depth_grid[0, coord]
            if hf_grid[0, coord]['S'] == 0:
                N_BC_q[coord] = 0
            else:
                N_BC_q[coord] = flow_grid[0, coord]['S'] / hf_grid[0, coord]['S'] * N_BC_hf[coord]
        elif bcell['t'] == 3:  # Boundary type 3
            # for testing purpose, to be corrected
            # the intended behaviour is:
            # 1 - assign the value to the outside cells
            # 2 - calculate the flow through the boudary
            depth_grid[0, coord] = bcell['v'] - z_grid[0, coord]
            
        elif bcell['t'] == 4:  # Boundary type 4
            pass
        else:
            print 'warning: unknown boundary type'            


    ##############
    # S Boundary #
    ##############
    for coord, bcell in np.ndenumerate(BCi['S']):
        if bcell['t'] == 1:  # Boundary type 1
            #~ S_BC_z[coord] = np.finfo(np.float32)
            #~ S_BC_h[coord] = 0
            S_BC_q[coord] = 0
        elif bcell['t'] == 2:  # Boundary type 2
            S_BC_z[coord] = z_grid[-1, coord]
            S_BC_h[coord] = depth_grid[-1, coord]
            if hf_grid[-2, coord]['S'] == 0:
                S_BC_q[coord] = 0
            else:
                S_BC_q[coord] = flow_grid[-2, coord]['S'] / hf_grid[-2, coord]['S'] * S_BC_hf[coord]
        elif bcell['t'] == 3:  # Boundary type 3
            S_BC_z[coord] = z_grid[-1, coord]  # terrain elevation
            S_BC_h[coord] = bcell['v'] - S_BC_z[coord] # depth = user_wse - z

            nf = n_grid_padded[-1, coord]

            # solve flow
            if S_BC_hf[coord] <= hf_min:
                S_BC_q[coord] = 0
            else:
                # flow at the boundary
                q_n_jm12 = S_BC_q[coord]
                # flow inside the domain
                q_n_jp12 = flow_grid[-1, coord]['S']
                # flow outside the domain
                q_n_jm32 = 0
                # wse in the domain
                y_n_j = depth_grid[-1, coord] + z_grid[-1, coord]
                # wse outside the domain
                y_n_jm1 = bcell['v']
                # solve flow
                S_BC_q[coord] = hydro.solve_q(g, theta, q_n_jm12, q_n_jm32,\
                    q_n_jp12, S_BC_hf[coord], \
                    Dt, Dx, Dy, y_n_j, y_n_jm1, nf)
            
        elif bcell['t'] == 4:  # Boundary type 4
            pass
        else:
            print 'warning: unknown boundary type'            

    ##########################
    # record boundary volume #
    ##########################
    bound_vol = Dt * math.fsum([np.sum(W_BC_q), np.sum(E_BC_q),
                             np.sum(N_BC_q), np.sum(S_BC_q)])
    

    return bound_vol


