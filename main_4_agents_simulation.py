
# -*- coding: utf-8 -*-
"""
@author: Amir
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import norm


"""
Constants of the simulations
"""
w_x, w_y, w_z = 0.3, 0.3, 0.3

num_agents = 4
dimension = 2
alpha = 1
gamma = 2
k_p = 5
k_v = 10
delta_t = 0.01

w_matrix_3d = np.array([[0, -w_z, w_y], 
                        [w_z, 0, -w_x], 
                        [-w_y, w_x, 0]])

w_matrix_2d = np.array([[0, -w_x],
                        [w_y, 0]])

###----------------------------------------------------------------------------

position = 10 * np.random.rand(num_agents, dimension, 1) 
velocity = np.random.rand(num_agents, dimension, 1)    


position_list = []
for i in range(num_agents):
    position_list.append([])
    
orientation_error_list = []
for i in range(num_agents):
    orientation_error_list.append([])


#### --------------------------------------------------------------------------
def frobeniusNorm(mat):
    sumSq = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            sumSq += pow(mat[i][j], 2)

    res = math.sqrt(sumSq)
    return round(res, 5)


#### --------------------------------------------------------------------------
def cross(a, b):
    c = np.array([[a[1][0]*b[2][0] - a[2][0]*b[1][0]],
                  [a[2][0]*b[0][0] - a[0][0]*b[2][0]],
                  [a[0][0]*b[1][0] - a[1][0]*b[0][0]]])
    return c


#### --------------------------------------------------------------------------
def gram_sch(X):
    Q, _ = np.linalg.qr(X)
    return Q 


#### --------------------------------------------------------------------------
def initialize(num_agents, dimension):
    auxiliary_matrix = np.zeros(shape = (num_agents, dimension, dimension-1), dtype=float)
    orientation_matrix = np.zeros(shape = (num_agents, dimension, dimension), dtype=float)
    rot_90 = np.array([[0, -1], [1, 0]])
    for agent in range(num_agents):
        auxiliary_matrix[agent] = np.random.rand(dimension, dimension-1)
        gs_auxiliary_matrix = gram_sch(auxiliary_matrix[agent])
        t_orientation_matrix = orientation_matrix[agent].transpose()
        if dimension == 2:
            t_orientation_matrix[:, 0] = gs_auxiliary_matrix.transpose()
            t_orientation_matrix[:, 1] = np.dot(rot_90, t_orientation_matrix[:, 0])
        elif dimension == 3:
            t_orientation_matrix[:, 0:2] = gs_auxiliary_matrix
            t_orientation_matrix[:, 2] = np.cross(t_orientation_matrix[:, 0], t_orientation_matrix[:, 1])
        orientation_matrix[agent] = t_orientation_matrix.transpose() 
    return orientation_matrix, auxiliary_matrix


#### --------------------------------------------------------------------------
def find_neighbor(adjacency_matrix, agent_num):
    neighbors = []
    agents = adjacency_matrix[agent_num]
    for idx in range(agents.shape[0]):
        if agents[idx] == 1:
            neighbors.append(idx)
    return neighbors


#### --------------------------------------------------------------------------
def update_orientation(agent_num, auxiliary_matrix, orientation_matrix):
    t_orientation_matrix = orientation_matrix[agent_num].transpose()
    gs_auxiliary_matrix = gram_sch(auxiliary_matrix[agent_num])
    rot_90 = np.array([[0, -1], [1, 0]])
    
    if dimension == 2:
        t_orientation_matrix[:, 0] = gs_auxiliary_matrix.transpose()
        t_orientation_matrix[:, 1] = np.dot(rot_90, t_orientation_matrix[:, 0])
    elif dimension == 3:
        t_orientation_matrix[:, 0:2] = gs_auxiliary_matrix
        t_orientation_matrix[:, 2]   = np.cross(t_orientation_matrix[:, 0], t_orientation_matrix[:, 1])
    orientation_matrix[agent_num]    = t_orientation_matrix.transpose()
    return orientation_matrix


#### --------------------------------------------------------------------------
def real_orientation_dynamic_holo(real_orientation_matrix, num_agent):
    T = delta_t
    if dimension == 2:
        R = np.dot(real_orientation_matrix[num_agent], w_matrix_2d) * T + real_orientation_matrix[num_agent]
    elif dimension == 3:
        R = np.dot(real_orientation_matrix[num_agent], w_matrix_3d) * T + real_orientation_matrix[num_agent]
    return R


#### --------------------------------------------------------------------------
"""
P is the auxilliary matrix for orientation estimation
"""
def p_dynamic(real_orientation_matrix, auxiliary_matrix, adjacency_matrix, agent_num):
    T = delta_t
    neighbors = find_neighbor(adjacency_matrix, agent_num)
    summation = np.zeros(shape = (dimension, dimension-1), dtype=float)
    r_i = real_orientation_matrix[agent_num]
    for neighbor in neighbors:
        r_j  = real_orientation_matrix[neighbor]
        r_ij = np.dot(r_i.transpose(), r_j)
        summation = summation + np.dot(r_ij, auxiliary_matrix[neighbor]) - auxiliary_matrix[agent_num]
    if dimension == 2:
        p = (np.dot(-w_matrix_2d, auxiliary_matrix[agent_num]) + summation) * T + auxiliary_matrix[agent_num]
    elif dimension == 3:
        p = (np.dot(-w_matrix_3d, auxiliary_matrix[agent_num]) + summation) * T + auxiliary_matrix[agent_num]
    return p


#### --------------------------------------------------------------------------
def dynamic_holo(real_orientation_matrix, position, velocity, num_agent, u, flag_maneuver):
    if flag_maneuver and num_agent == 0:
        delta_x = 0.01
        p = np.array([[position[num_agent][0][0] + delta_x], [10]])
        v = 5
    else:
        T = delta_t
        v = T * np.dot(real_orientation_matrix[num_agent], u) + velocity[num_agent]
        p = T * v + position[num_agent]
    return [p, v]
    

#### --------------------------------------------------------------------------
def dynamic_nonholo(position, real_orientation_matrix, flag_maneuver, num_agent, w, v):
    w_ = np.array([[0,       -w[2][0],  w[1][0]], 
                  [w[2][0],   0,       -w[0][0]], 
                  [-w[1][0],  w[0][0],  0]])
    if flag_maneuver and num_agent == 0:
        pass
    else:
        T = delta_t
        R = np.dot(real_orientation_matrix[num_agent], w_) * T + real_orientation_matrix[num_agent]
        #p = np.dot(R, np.array([[v], [0], [0]])) * T + position[num_agent]
        p = ((real_orientation_matrix[num_agent][:, 0] * v) * T).reshape((3,1)) + position[num_agent]
    return p, R


#### --------------------------------------------------------------------------
def dynamic_unicycle(position, real_orientation_matrix, flag_maneuver, num_agent, w, v):
    w_ = np.array([[0, -w],
                   [w,  0]])
    if flag_maneuver and num_agent == 0:
        pass
    else:
        T = delta_t
        R = np.dot(real_orientation_matrix[num_agent], w_) * T + real_orientation_matrix[num_agent]
        #p = np.dot(R, np.array([[v], [0]])) * T + position[num_agent]
        p = ((real_orientation_matrix[num_agent][:, 0] * v) * T).reshape((2,1)) + position[num_agent]
    return p, R
    

#### --------------------------------------------------------------------------
"""
Formation Stabilization of holonomic agents in 3-D
"""
def formation_stabilization_holo(position, velocity, orientation_matrix, agent_num):
    
    adjacency_matrix = np.array([[ 0,  1,  0,   0],
                                 [-1,  0,  1,  -1],
                                 [ 0, -1,  0,   1],
                                 [ 0,  1, -1,   0]])

    position_desired_3d = {0:{1:np.array([[4],  [0],  [0]])},
                           1:{2:np.array([[-4], [0],  [4]])},
                           2:{3:np.array([[0],  [4],  [-4]])},
                           3:{1:np.array([[4],  [-4], [0]])}}

    
    neighbors = find_neighbor(adjacency_matrix, agent_num)
    summation = np.zeros(shape = (dimension, 1), dtype=float)
    for neighbor in neighbors:
        v_ij_i = np.dot(orientation_matrix[agent_num].transpose(), velocity[neighbor] - velocity[agent_num])
        p_ij = position[neighbor] - position[agent_num]
        p_desired = position_desired_3d[agent_num][neighbor]
        summation += np.dot(orientation_matrix[agent_num].transpose(), p_ij) - np.dot(orientation_matrix[agent_num].transpose(), p_desired) + gamma * v_ij_i
    u_i = summation - alpha * np.dot(orientation_matrix[agent_num].transpose(), velocity[agent_num])
    return u_i

    
#### --------------------------------------------------------------------------
"""
Moving Formation and Maneuver of holonomic agents in 2-D
"""
def formation_maneuver_holo(position, velocity, orientation_matrix, agent_num, step_time):
    T = delta_t
    adjacency_matrix = np.array([[ 0,  1,  0,  -1],
                                 [-1,  0,  1,  -1],
                                 [ 0, -1,  0,   1],
                                 [ 1,  1, -1,   0]])
    if step_time * T < 18:
        position_desired_2d = {0:{1:np.array([[2],  [3]])},
                               1:{2:np.array([[0],  [5]])},
                               2:{3:np.array([[-2], [3]])},
                               3:{1:np.array([[2],  [-8]]), 
                                  0:np.array([[0],  [-11]])}}
    else:
        position_desired_2d = {0:{1:np.array([[2],  [3]])},
                               1:{2:np.array([[-4], [2]])},
                               2:{3:np.array([[-1], [-1]])},
                               3:{1:np.array([[5],  [-1]]), 
                                  0:np.array([[3],  [-4]])}}
    
    neighbors = find_neighbor(adjacency_matrix, agent_num)
    summation = np.zeros(shape = (dimension, 1), dtype=float)
    
    for neighbor in neighbors:
        v_ij_i = np.dot(orientation_matrix[agent_num].transpose(), velocity[neighbor] - velocity[agent_num])
        p_ij = position[neighbor] - position[agent_num]
        p_desired = position_desired_2d[agent_num][neighbor]
        p_ = np.dot(orientation_matrix[agent_num].transpose(), p_ij) - np.dot(orientation_matrix[agent_num].transpose(), p_desired)
        summation +=  k_p * p_ + k_v * v_ij_i

    u_i = summation
    return u_i


#### --------------------------------------------------------------------------
"""
Formation Stabilization of Nonholonomic agents in 3-D
"""
def formation_stabilization_nonholo(position, orientation_matrix, agent_num):
    
    adjacency_matrix = np.array([[ 0,  1,  0,   0],
                                 [-1,  0,  1,  -1],
                                 [ 0, -1,  0,   1],
                                 [ 0,  1, -1,   0]])

    position_desired_3d = {0:{1:np.array([[7],  [0],  [0]])},
                           1:{2:np.array([[0], [0],  [7]])},
                           2:{3:np.array([[-7],  [0],  [0]])},
                           3:{1:np.array([[7],  [0], [-7]])}}
    
    neighbors = find_neighbor(adjacency_matrix, agent_num)
    summation = np.zeros(shape = (dimension, 1), dtype=float)
    h_i = np.array([[1], 
                    [0], 
                    [0]])
    for neighbor in neighbors:
        r_ij = np.dot(orientation_matrix[agent_num].transpose(), orientation_matrix[neighbor])
        p_j_j = np.dot(orientation_matrix[neighbor].transpose(), position[neighbor])
        p_ij_i = np.dot(r_ij, p_j_j) - np.dot(orientation_matrix[agent_num].transpose(), position[agent_num])
        p_desired = position_desired_3d[agent_num][neighbor]
        summation += (p_ij_i - np.dot(orientation_matrix[agent_num].transpose(), p_desired))
    v_i = np.dot(h_i.transpose(), summation)
    w_i = cross(h_i, summation)
    return v_i[0][0], w_i


#### --------------------------------------------------------------------------
"""
Formation Stabilization of Unicycle agents in 2-D
"""
def formation_stabilization_unicycle(position, real_orientation_matrix, orientation_matrix, agent_num):

    adjacency_matrix = np.array([[0, 1, 1, 0],
                                 [1, 0, 0, 1],
                                 [1, 0, 0, 1],
                                 [0, 1, 1, 0]])
    
    position_desired_2d = {0:{1:np.array([[2],  [2]]),
                              2:np.array([[-2],  [2]])},
                           1:{0:np.array([[-2], [-2]]),
                              3:np.array([[-2],  [2]])},
                           2:{0:np.array([[2], [-2]]),
                              3:np.array([[2], [2]])},
                           3:{1:np.array([[2], [-2]]),
                              2:np.array([[-2], [-2]])}}
    
    neighbors = find_neighbor(adjacency_matrix, agent_num)
    summation = np.zeros(shape = (dimension, 1), dtype=float)
    
    for neighbor in neighbors:
        p_ij = position[neighbor] - position[agent_num]
        p_desired = position_desired_2d[agent_num][neighbor]
        summation += np.dot(orientation_matrix[agent_num].transpose(), p_ij) - np.dot(orientation_matrix[agent_num].transpose(), p_desired)

    v_i = summation[0][0]
    w_i = summation[1][0]
    return v_i, w_i
    

#### ----------------------- ### MAIN ### ---------------------------------------------------
orientation_matrix, auxiliary_matrix = initialize(num_agents, dimension)
real_orientation_matrix =   orientation_matrix 
flag_maneuver = False

#"""
### For Holonomic
adjacency_matrix = np.array([[ 0,  1,  0,  0],
                             [-1,  0,  1, -1],
                             [ 0, -1,  0,  1],
                             [ 0,  1, -1,  0]])
#"""
"""
### For Unicycle
adjacency_matrix = np.array([[0, 1, 1, 0],
                             [1, 0, 0, 1],
                             [1, 0, 0, 1],
                             [0, 1, 1, 0]])

"""

for step_time in range(4000):
    for agent_num in range(num_agents):
        
        P = p_dynamic(real_orientation_matrix, auxiliary_matrix, adjacency_matrix, agent_num)
        auxiliary_matrix[agent_num] = P
        
        orientation_matrix = update_orientation(agent_num, auxiliary_matrix, orientation_matrix)
        
        R_real_holo = real_orientation_dynamic_holo(real_orientation_matrix, agent_num)
        
        ### Control Laws ###
        u_holo = formation_maneuver_holo(position, velocity, orientation_matrix, agent_num, step_time)
        
        #u_holo = formation_stabilization_holo(position, velocity, orientation_matrix, agent_num)
        
        #v_nonholo, w_nonholo = formation_stabilization_nonholo(position, orientation_matrix, agent_num)
        
        #v_unicycle, w_unicycle = formation_stabilization_unicycle(position, real_orientation_matrix, orientation_matrix, agent_num)
        
        [p_holo, v_holo] = dynamic_holo(real_orientation_matrix, position, velocity, agent_num, u_holo, flag_maneuver=True)
        
        #p_unicycle, R_real_unicycle = dynamic_unicycle(position, real_orientation_matrix, flag_maneuver, agent_num, w_unicycle, v_unicycle)
        
        #p_nonholo, R_real_nonholo = dynamic_nonholo(position, real_orientation_matrix, flag_maneuver, agent_num, w_nonholo, v_nonholo)

        ### UPDATE ###
        real_orientation_matrix[agent_num] = R_real_holo
        velocity[agent_num] = v_holo 
        position[agent_num] = p_holo
        
        # For plot        
        position_list[agent_num].append(p_holo)



### --------------------------------------------------------------------------   
x_1, x_2, x_3, x_4 = [], [], [], []
y_1, y_2, y_3, y_4 = [], [], [], []
z_1, z_2, z_3, z_4 = [], [], [], [] 
idx = 1      

for i in range(len(position_list[idx])):
    x_1.append(position_list[0][i][0][0])  
    y_1.append(position_list[0][i][1][0])
    
    x_2.append(position_list[1][i][0][0])  
    y_2.append(position_list[1][i][1][0])
    
    x_3.append(position_list[2][i][0][0])  
    y_3.append(position_list[2][i][1][0])
    
    x_4.append(position_list[3][i][0][0])  
    y_4.append(position_list[3][i][1][0])
    
    if dimension == 3:
        z_1.append(position_list[0][i][2][0])
        z_2.append(position_list[1][i][2][0])
        z_3.append(position_list[2][i][2][0])
        z_4.append(position_list[3][i][2][0])


#"""
def plot_2d():
    plt.scatter(x_1[0], y_1[0])
    plt.scatter(x_2[0], y_2[0])
    plt.scatter(x_3[0], y_3[0])
    plt.scatter(x_4[0], y_4[0])
    plt.plot(x_1, y_1, label = "Agent 1")
    plt.plot(x_2, y_2, label = "Agent 2")
    plt.plot(x_3, y_3, label = "Agent 3")
    plt.plot(x_4, y_4, label = "Agent 4")

    plt.plot([x_1[-1], x_2[-1]], [y_1[-1], y_2[-1]],'k-')
    plt.plot([x_2[-1], x_4[-1]], [y_2[-1], y_4[-1]],'k-')
    plt.plot([x_3[-1], x_4[-1]], [y_3[-1], y_4[-1]],'k-')
    plt.plot([x_3[-1], x_1[-1]], [y_3[-1], y_1[-1]],'k-')
    #plt.plot([x_3[-1], x_1[-1]], [y_3[-1], y_1[-1]],'k-')
    
    plt.legend() 
    plt.show()
    
def plot_2d_unicycle():
    plt.scatter(x_1[0], y_1[0])
    plt.scatter(x_2[0], y_2[0])
    plt.scatter(x_3[0], y_3[0])
    plt.scatter(x_4[0], y_4[0])
    plt.plot(x_1, y_1, label = "Agent 1")
    plt.plot(x_2, y_2, label = "Agent 2")
    plt.plot(x_3, y_3, label = "Agent 3")
    plt.plot(x_4, y_4, label = "Agent 4")

    plt.plot([x_1[-1], x_2[-1]], [y_1[-1], y_2[-1]],'k-')
    plt.plot([x_2[-1], x_4[-1]], [y_2[-1], y_4[-1]],'k-')
    plt.plot([x_3[-1], x_4[-1]], [y_3[-1], y_4[-1]],'k-')
    plt.plot([x_3[-1], x_1[-1]], [y_3[-1], y_1[-1]],'k-')
    
    plt.legend() 
    #plt.savefig('unicycle.png', dpi=1200)
    plt.show()
    
    
def plot_2d_maneuver():

    plt.plot(x_1, y_1, label = "Agent 1")
    plt.plot(x_2, y_2, label = "Agent 2")
    plt.plot(x_3, y_3, label = "Agent 3")
    plt.plot(x_4, y_4, label = "Agent 4")

    plt.plot([x_1[-1], x_2[-1]], [y_1[-1], y_2[-1]],'k-')
    plt.plot([x_2[-1], x_3[-1]], [y_2[-1], y_3[-1]],'k-')
    plt.plot([x_3[-1], x_4[-1]], [y_3[-1], y_4[-1]],'k-')
    plt.plot([x_4[-1], x_1[-1]], [y_4[-1], y_1[-1]],'k-')
    
    plt.plot([x_1[0], x_2[0]], [y_1[0], y_2[0]],'k-')
    plt.plot([x_2[0], x_4[0]], [y_2[0], y_4[0]],'k-')
    plt.plot([x_3[0], x_4[0]], [y_3[0], y_4[0]],'k-')
    plt.plot([x_3[0], x_1[0]], [y_3[0], y_1[0]],'k-')
    
    plt.plot([x_1[1200], x_2[1200]], [y_1[1200], y_2[1200]],'k-')
    plt.plot([x_2[1200], x_3[1200]], [y_2[1200], y_3[1200]],'k-')
    plt.plot([x_3[1200], x_4[1200]], [y_3[1200], y_4[1200]],'k-')
    plt.plot([x_4[1200], x_1[1200]], [y_4[1200], y_1[1200]],'k-')
    
    plt.plot([x_1[1800], x_2[1800]], [y_1[1800], y_2[1800]],'k-')
    plt.plot([x_2[1800], x_3[1800]], [y_2[1800], y_3[1800]],'k-')
    plt.plot([x_3[1800], x_4[1800]], [y_3[1800], y_4[1800]],'k-')
    plt.plot([x_4[1800], x_1[1800]], [y_4[1800], y_1[1800]],'k-')
    
    plt.legend() 
    #plt.savefig('maneuver.png', dpi=1200)
    plt.show()
    

def plot_3d():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x_1, y_1, z_1, label='Agent 1')
    ax.plot(x_2, y_2, z_2, label='Agent 2')
    ax.plot(x_3, y_3, z_3, label='Agent 3')
    ax.plot(x_4, y_4, z_4, label='Agent 4')
    
    ax.plot([x_1[-1], x_2[-1]], [y_1[-1], y_2[-1]], [z_1[-1], z_2[-1]], 'k-')
    ax.plot([x_2[-1], x_3[-1]], [y_2[-1], y_3[-1]], [z_2[-1], z_3[-1]], 'k-')
    ax.plot([x_2[-1], x_4[-1]], [y_2[-1], y_4[-1]], [z_2[-1], z_4[-1]], 'k-')
    ax.plot([x_3[-1], x_4[-1]], [y_3[-1], y_4[-1]], [z_3[-1], z_4[-1]], 'k-')
    ax.plot([x_4[-1], x_1[-1]], [y_4[-1], y_1[-1]], [z_4[-1], z_1[-1]], 'k-')
    ax.plot([x_3[-1], x_1[-1]], [y_3[-1], y_1[-1]], [z_3[-1], z_1[-1]], 'k-')
    
    ax.legend()
    #plt.savefig('holo_stabilization.png', dpi=1200)
    
    plt.show()
    
def plot_estimation_error():
    agent1 = []
    agent2 = []
    agent3 = []
    agent4 = []
    counter = 0
    time = []
    for i in range(len(orientation_error_list[0])):

        agent1.append(orientation_error_list[0][i])
        agent2.append(orientation_error_list[1][i])
        agent3.append(orientation_error_list[2][i])
        agent4.append(orientation_error_list[3][i])
        time.append(counter)
        counter += 1
    plt.plot(time, agent1, label = "Agent 1")
    plt.plot(time, agent2, label = "Agent 2")
    plt.plot(time, agent3, label = "Agent 3")
    plt.plot(time, agent4, label = "Agent 4")
    plt.legend() 
    plt.show()
    





#plot_3d()
plot_2d_maneuver()
#plot_2d_unicycle()
   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        