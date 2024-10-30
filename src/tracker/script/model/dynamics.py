import numpy as np
import math
import casadi

# z = [u, x]
# u = [ax, ay, az]
# x = [pos_x, pos_y, pos_z, vx, vy, vz]
## yuwei: check it, it's correct
def second_order_dynamics(u, x, dt = 0.1):
    dim = 2
    integral = np.zeros(dim*2)
    integral[:dim] = 0.5 * dt**2 * u
    integral[dim:] = dt * u

    phi = np.eye(dim*2)
    for i in range(dim):
        phi[i, i+dim] = dt

    return np.dot(phi, x) + integral


def second_order_dynamics_casadi(u, x, dt=0.1):
    """
    using casadi to implement second order dynamics
    """
    dim = 2
    integral = casadi.SX.zeros(dim * 2)
    integral[:dim] = 0.5 * dt ** 2 * u
    integral[dim:] = dt * u

    phi = casadi.SX.eye(dim * 2)
    for i in range(dim):
        phi[i, i + dim] = dt

    return casadi.mtimes(phi, x) + integral



def first_order_dynamics(u, x, dt = 0.1):
    """
    u = [vx1,vy1, vx2, vy2, ...]
    x = [x1, y1, x2, y2, ...]
    """
    return x + u * dt


# dx = [v; a; deuler];
# x = [pos; vel; euler]
# u = [f; angular_rate]
def bebop_dynamics(u, x, dt = 0.1):

    # Constants
    m = 0.5
    g = 9.81
    Ix = 0.0023
    Iy = 0.0023
    Iz = 0.004
    d = 0.23

    # Forces
    f = u[0]
    w = u[1:]

    # States
    pos = x[0:3]
    vel = x[3:6]
    euler = x[6:]

    # Rotation matrix
    R = np.array([[math.cos(euler[1])*math.cos(euler[2]), math.sin(euler[0])*math.sin(euler[1])*math.cos(euler[2]) - math.cos(euler[0])*math.sin(euler[2]), math.cos(euler[0])*math.sin(euler[1])*math.cos(euler[2]) + math.sin(euler[0])*math.sin(euler[2])],
                  [math.cos(euler[1])*math.sin(euler[2]), math.sin(euler[0])*math.sin(euler[1])*math.sin(euler[2]) + math.cos(euler[0])*math.cos(euler[2]), math.cos(euler[0])*math.sin(euler[1])*math.sin(euler[2]) - math.sin(euler[0])*math.cos(euler[2])],
                  [-math.sin(euler[1]), math.sin(euler[0])*math.cos(euler[1]), math.cos(euler[0])*math.cos(euler[1])]])
    
    # Acceleration
    a = np.array([0, 0, f/m]) - np.array([0, 0, g])
    a = np.dot(R, a)

    # Angular acceleration
    w = np.array([0, w[0], w[1]])
    w = np.dot(np.linalg.inv(np.array([[1, math.sin(euler[0])*math.tan(euler[1]), math.cos(euler[0])*math.tan(euler[1])],
                                       [0, math.cos(euler[0]), -math.sin(euler[0])],
                                       [0, math.sin(euler[0])/math.cos(euler[1]), math.cos(euler[0])/math.cos(euler[1])]])), w)
    
    deuler = np.array([w[1], w[2], math.sin(euler[0])*math.tan(euler[1])*w[1] + math.cos(euler[0])*math.tan(euler[1])*w[2]])

    # Update states
    posNext = pos + vel * dt
    velNext = vel + a * dt
    eulerNext = euler + deuler * dt

    return np.concatenate((posNext, velNext, eulerNext))
    