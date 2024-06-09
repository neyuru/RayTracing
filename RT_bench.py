# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:15:02 2019

@author: yorch
"""

# Copyright (c) 2023 Jorge A. Ramos O.
#
# Permission is granted to any person to use, copy, modify, merge, distribute, sublicense,
# and/or sell copies of this Software, subject to the following condition:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
#
# The full license is included in the LICENSE file in the project root.
# If not, see <https://opensource.org/licenses/MIT>.

"""
To gain a deeper understanding of the MxSA, AnDF, or HySA algorithms implemented
in this software, it is recommended to read:

Title: On ray tracing for sharp changing media
Authors: Jorge A. Ramos Oliveira, Arturo Baltazar, Mario Castelán
Publication: J. Acoust. Soc. Am. 1 September 2019; 146 (3): 1595–1604
Link/DOI: https://doi.org/10.1121/1.5125133

In addition to the methodologies covered in the mentioned article, this software
also introduces a new method for tracing rays, termed the 'momentum algorithm',
that permits treatment of anisotropic media.
"""


import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import pool

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from scipy import interpolate
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


#######################
#      CONSTANTS      #
#######################
''' THE FOLLOWING LINES SHOULDN'T NORMALLY BE EDITED '''

# 'interface' Scenario Parameters
THCK_PARAM = 0.005                                          # Thickness parameter defining the interface thickness.
A = (1 + np.sqrt(2))/2 - 99*(np.sqrt(2) - 1)/200            # Auxiliary number for SIGMA calculation.
SIGMA = -2 * THCK_PARAM * np.log((A - 1)/(np.sqrt(2) - A))  # True thickness of an interface. It is assumed that this is the smallest feature in a simulation.

# Golden Search Parameters
DELTA_G = np.pi/2                                           # Defines the half interval size for golden search. The whole search interval is +/- DELTA_G.
GOLD_RATIO = (np.sqrt(5) - 1) / 2                           # Golden ratio used in the golden search algorithm.
GOLD_TOL = np.sqrt(sys.float_info.epsilon)                  # Tolerance value for golden search, derived from machine epsilon.

# Simulation Parameters
MAX_DEVIATION = 0.2                                         # Maximum acceptable average deviation error for a ray in the 'interface' scenario.
THREADS = multiprocessing.cpu_count()                       # Number of system threads available for parallel processing.

# Dummy list variable
VECTOR_LIST = []

## EDIT THE FOLLOWING LINES AS NEEDED ##
# Grid and Step Size Parameters
DELTA = SIGMA/3                                             # Grid size as a fraction of the interface width (SIGMA). This grid is going to be used to sample a medium,
                                                            # thus, a value equal than or higher than 2 is highly recommended.
DELTA_S_DIVISOR = 20                                        # Default divisor for SIGMA to calculate DELTA_S. It is strongly encouraged to select a value higher than 1.
                                                            # During the execution of the script, a 'suitable' value can be automatically calculated if the user so chooses.
DELTA_S = SIGMA/DELTA_S_DIVISOR                             # Default ray advancement step size as a fraction of the interface width (SIGMA).
N = 10                                                      # For the Fish-Eye test, N equals the turns that the ray moves around the unit circle. Must be integer > 0
                                                            # Suggestion: use N=1 to test precision of algorithm and N=10 to benchmark completion times
DELTA_S_DIVISOR_FISHEYE = 90                                # For the Fish-Eye test, equals to the number of segments the perimeter of the unit circle is going to be divided.

# DELTA_S search parameters
# These values where carefully selected to lie within a close margin of the optimal value. Increasing the search interval would not affect the results but it will definitely
# increase computational time. Decreasing the search interval would propbably leave out the optimal DELTA_S value.
DELTA_STEP = 0.01                                           # Increment used by the algorithm to search for an optimal DELTA_S (Sharp interface).
DELTA_S_DIVISOR_UPPER_LIMIT = 3                             # Upper bound for DELTA_S_DIVISOR during DELTA_S search.
DELTA_S_DIVISOR_LOWER_LIMIT = 1 + DELTA_STEP                # Lower bound for DELTA_S_DIVISOR during DELTA_S search.
DELTA_STEP_FISHEYE = 1                                      # Increment used by the algorithm to search for an optimal DELTA_S (Fisheye).
DELTA_S_DIVISOR_FISHEYE_UPPER_LIMIT = 303                   # Upper bound for DELTA_S_DIVISOR_FISHEYE during DELTA_S search.
DELTA_S_DIVISOR_FISHEYE_LOWER_LIMIT = 4                     # Lower bound for DELTA_S_DIVISOR_FISHEYE during DELTA_S search.
DELTA_STEP_VERT = 0.005                                     # Increment used by the algorithm to search for an optimal DELTA_S (Vertically heterogeneous).
DELTA_S_DIVISOR_VERT_UPPER_LIMIT = 2                        # Upper bound for DELTA_S_DIVISOR_VERT during DELTA_S search.
DELTA_S_DIVISOR_VERT_LOWER_LIMIT = 1/40                     # Lower bound for DELTA_S_DIVISOR_VERT during DELTA_S search.


#######################
#  DEFINED FUNCTIONS  #
#######################

# AVAILABLE SCENARIOS
# -- Sharp interface
def interface(a, b):
    r = np.sqrt(2) - (np.sqrt(2) - 1)/(1 + np.exp(-b/THCK_PARAM))
    return r
# -- Maxwell's Fisheye
def fisheye(a, b):
    r = 1/(1 + np.power(a, 2) + np.power(b, 2))
    return r
# -- Vertically heterogeneous
def vert_heterogeneous(a, b):
    v = 18 + 2*b
    return 1/v
# -- Angle dependent part: intended to be multiplied to an isotropic function to convert it to an anisotropic medium
def anisotropy(theta, gamma):
    return np.sqrt((gamma*np.sin(theta))**2 + np.cos(theta)**2)


# HELPER FUNCTIONS
def remove_outliers_iqr(data):
    """
    Removes outliers in a vector of data according to the Interquartile Range method.

    Parameters:
    - data: The data to be trimmed.

    Returns:
    - The filtered data.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]


def n_gradient(vector, grd, z):
    """
    Calculates the index of refraction and its gradient over a given point.

    Parameters:
    - vector: The given position.
    - grd: The callable bi-cubic function containing the gradient of n.
    - z: The callable bi-linear function representing the index of refraction n.

    Returns:
    - The value of n at the given point and its gradient.
    """
    n = z(vector[1], vector[0])[0][0]
    gradnx = grd[1](vector[1], vector[0])[0][0]
    gradny = grd[0](vector[1], vector[0])[0][0]
    return n, np.array([gradnx, gradny])


def dn_dang(ang, gamma):
    """
    Calculates the derivative of the angle-dependent part of the index of refraction with respect of group angle (not currently used).

    Parameters:
    - ang: The current group angle.

    Returns:
    - The derivative.
    """
    h = GOLD_TOL*ang
    a = anisotropy(ang - h, gamma)
    d = anisotropy(ang + h, gamma)
    return (d - a)/(2*h)


def golden(func,a,b):
    """
    Searches for the minimum value in a function inside an interval.

    Parameters:
    - func: The callable function to be minimized.
    - a: The intervals' lower bound.
    - b: The intervals' upper bound.

    Returns:
    - The minimum value of func between a and b.
    """
    c=b-(b-a)*GOLD_RATIO
    d=a+(b-a)*GOLD_RATIO

    while abs(c-d)>GOLD_TOL:
        if func(c)<func(d):
            b=d
        else:
            a=c

        c=b-(b-a)*GOLD_RATIO
        d=a+(b-a)*GOLD_RATIO

    return (b+a)/2


def impulse_t(a, b, step):
    """
    Calculates the impulse integrals using the trapezoidal rule.

    Parameters:
    - a: The gradient of n at the starting point.
    - b: The gradient of n at the final point.
    - step: The current step-size.

    Returns:
    - The value of the impulse integral.
    """
    return step*(a+b)/2


def moment(n, theta, gamma, opt_vec):
    """
    Calculates the moment of a ray in a certain point in its trajectory.

    Parameters:
    - n: The current isotropic index of refraction.
    - theta: The ray's current group angle.
    - opt_vec: if calculating x-moment, opt_vec=[cos(theta),-sen(theta)**2], otherwise opt_vec=[sen(theta),cos(theta)**2]

    Returns:
    - The moment in a certain direction.
    """
    coef=anisotropy(theta, gamma)
    return n*coef*opt_vec[0]*(1 + opt_vec[1]*(gamma**2-1)/(coef**2))

def moments(theta, n, i_unitv, gamma):
    """
    Calculates the moment vector of a ray in a certain point in its trajectory.

    Parameters:
    - theta: The ray's current group angle.
    - n: The current index of refraction.
    - i_unitv: The ray's current  unit tangent vector.

    Returns:
    - The moment vector of a ray.
    """
    # dn = dn_dang(theta, gamma)       # in an isotropic medium, this value is zero
    return np.array([moment(n, theta, gamma, np.array([i_unitv[0], -i_unitv[1]**2])), moment(n, theta, gamma, np.array([i_unitv[1], i_unitv[0]**2]))])

def constants(user_choice):
    """
    Returns the appropriate constants depending on user choice.

    Parameters:
    - user_choice: Self explanatory.

    Returns:
    - Some constants.
    """
    if user_choice == "1":
        gamma = 1                                                   # this means isotropy
        ray_count = 42                                              # number of rays to be simulated
        theta_v = np.linspace(2*(np.pi/60), np.pi/2, ray_count+1)   # initial shooting angles of the simulated rays
        pos_x = np.ones((ray_count))*-2                             # initial x position for the rays
        s = 80                                                      # maximum allowed travelled distance (hopefully the rays abandon the boundaries first)
        limx_i, limx_s, limy_i, limy_s = -2, 20, -2, 4              # boundaries of the simulation
        op_interface, op_fish, op_vert_heterogeneous, op_anisotropy = 1, 0, 0, 0
    elif user_choice == "2":
        gamma = 1                                                   # this means isotropy
        ray_count = 1                                               # number of rays to be simulated
        theta_v = np.linspace(np.pi/2, np.pi/2, 1)                  # initial shooting angles of the simulated rays
        pos_x = np.array((1, 0))                                    # initial position for the rays
        s = N * (2*np.pi)                                           # N is a global variable that indicates the turns around the unit circle
        limx_i, limx_s, limy_i, limy_s = -1.5, 1.5, -1.5, 1.5       # boundaries of the simulation
        op_interface, op_fish, op_vert_heterogeneous, op_anisotropy = 0, 1, 0, 0
    elif user_choice == "3":
        gamma = 1                                                   # this means isotropy
        ray_count = 31                                              # number of rays to be simulated
        theta_v = np.linspace(0, np.pi/2, ray_count)                # initial shooting angles of the simulated rays
        pos_x = np.ones((ray_count))*-2                             # initial x position for the rays
        s = 80                                                      # maximum allowed travelled distance (hopefully the rays abandon the boundaries first)
        limx_i, limx_s, limy_i, limy_s = -2, 5, -2.5, 1             # boundaries of the simulation. These were set so that the interpolation of the wavefront
                                                                    # line could always have more than one point to work with the window set as
                                                                    # limx_i, limx_s, limy_i, limy_s = -2, 4, -2, 0. Set these limits when displaying a movie
                                                                    # is not important and the objective is to benchmark the scenario.
        op_interface, op_fish, op_vert_heterogeneous, op_anisotropy = 0, 0, 1, 0
    elif user_choice == "4":
        gamma = 3                                                   # this means anisotropy
        ray_count = 31                                              # number of rays to be simulated
        theta_v = np.linspace(0, np.pi/2, ray_count)                # initial shooting angles of the simulated rays
        pos_x = np.ones((ray_count))*-2                             # initial x position for the rays
        s = 80                                                      # maximum allowed travelled distance (hopefully the rays abandon the boundaries first)
        limx_i, limx_s, limy_i, limy_s = -2, 5, -2.5, 1             # boundaries of the simulation. These were set so that the interpolation of the wavefront
                                                                    # line could always have more than one point to work with the window set as
                                                                    # limx_i, limx_s, limy_i, limy_s = -2, 4, -2, 0. Set these limits when displaying a movie
                                                                    # is not important and the objective is to benchmark the scenario.
        op_interface, op_fish, op_vert_heterogeneous, op_anisotropy = 0, 0, 0, 1
    return gamma, ray_count, theta_v, pos_x, s, limx_i, limx_s, limy_i, limy_s, op_interface, op_fish, op_vert_heterogeneous, op_anisotropy


# RAY ADVANCEMENT
# -- Linear approximation
def first_order_taylor(i_vpos, v_un, step):
    """
    Calculates a ray's next position given the current one by using a simple linear approximation.

    Parameters:
    - i_vpos: The current position.
    - v_un: The current rays' unit tangent vector.
    - step: The current step-size.

    Returns:
    - The final position.
    """
    return i_vpos + v_un*step
# -- Taylor series approximation
def second_order_taylor(i_vpos, i_unitv, step, init_n, i_grad, grd, z):
    """
    Calculates a ray's next position given the current one by using a Taylor series expansion of dr/ds.

    Parameters:
    - i_vpos: The current position.
    - i_unitv: The current rays' unit tangent vector.
    - step: The current step-size.
    - init_n: The current index of refraction.
    - i_grad: The current index gradient.
    - grd: The callable bi-cubic function of the gradient of n.
    - z: The callable bi-linear function of the index of refraction n.

    Returns:
    - The final position, index of refraction and gradient of n.
    """
    f_vpos = first_order_taylor(i_vpos, i_unitv, step) + (i_grad - np.dot(i_grad, i_unitv)*i_unitv)*(step**2)/(2*init_n)
    # final_n, f_grad = n_gradient(f_vpos, grd, z)
    # return f_vpos, final_n, f_grad
    return f_vpos
# -- Two-point "curvature correction"
def curvature_t(i_angle, i_grad, i_unitv, init_n, i_vpos, grd, z, step):
    """
    Uses the ray's current curvature and the trapezoidal rule to calculate the ray's next position.

    Parameters:
    - i_angle: The current group (ray) angle.
    - i_grad: The current index gradient.
    - i_unitv: The current rays' unit tangent vector.
    - init_n: The current index of refraction.
    - i_vpos: The current position.
    - grd: The callable bi-cubic function of the gradient of n.constants
    - z: The callable bi-linear function of the index of refraction n.
    - step: The current step-size.

    Returns:
    - The final: index of refraction, gradient, position and a flag that indicates if the curvature is negligible (false) or not (true).
    """
    curv = np.linalg.norm((i_grad - np.dot(i_grad, i_unitv)*i_unitv))/init_n

    ignore = True
    if curv < GOLD_TOL:
        f_vpos = first_order_taylor(i_vpos, i_unitv, step)
        ignore = False
    else:
        d_curv = curv*step
        if np.cross(i_grad, i_unitv) > 0:
            f_vpos = i_vpos + np.array((np.sin(i_angle) - np.sin(i_angle - d_curv), np.cos(i_angle - d_curv) - np.cos(i_angle)))/curv
        else:
            f_vpos = i_vpos + np.array((np.sin(i_angle + d_curv) - np.sin(i_angle), -np.cos(i_angle + d_curv) + np.cos(i_angle)))/curv

    return f_vpos, ignore


# ANGLE DETERMINATION
# -- Finite difference solution to group (ray) angle.
def finite_diff():
    x = 11*VECTOR_LIST[3] - 18*VECTOR_LIST[2] + 9*VECTOR_LIST[1] - 2*VECTOR_LIST[0]
    return np.arctan2(x[1],x[0])
# -- "Angular Displacement Form" solution to group (ray) angle.
def tfinal_2o(i_angle, step, init_n, final_n, i_grad, f_grad):
    """
    Calculates the subsequent shooting angle given the previous one using Runge-Kutta of 2nd order.

    Parameters:
    - i_angle: Initial angle.
    - step: The current step size.
    - init_n: The initial index of refraction.
    - final_n: The final index of refraction.
    - i_grad: The initial gradient of n.
    - f_grad: The final gradient of n.

    Returns:
    - The final angle.
    """
    k1 = step*(np.cos(i_angle)*i_grad[1] - np.sin(i_angle)*i_grad[0])/init_n
    k2 = step*(np.cos(i_angle + k1)*f_grad[1] - np.sin(i_angle + k1)*f_grad[0])/final_n
    return i_angle + (k1 + k2)/2
# -- Two-point analitic solution to the cost function.
def theta_cost_t(init_n, i_angle, i_grad, f_grad, step):
    """
    Calculates the subsequent shooting angle given the previous one using the two-point cost function.

    Parameters:
    - init_n: The initial index of refraction.
    - i_angle: Initial angle.
    - i_grad: The initial gradient of n.
    - f_grad: The final gradient of n.
    - step: The current step-size.

    Returns:
    - The final angle.
    """
    return np.arctan2(init_n*np.sin(i_angle) + impulse_t(i_grad[1], f_grad[1], step), init_n*np.cos(i_angle) + impulse_t(i_grad[0], f_grad[0], step))


# PRE-PROCESING FUNCTIONS
# -- Creation of the grid
def genZ(xi, xs, yi, ys):
    """
    Calculates the index of refraction grid.

    Parameters:
    - xi: The starting x position.
    - xs: The final x position.
    - yi: The starting y position.
    - ys: The final y position.

    Returns:
    - The x and y coordinates of the grid, the matrices of x and y coordinates and the index of refraction grid.
    """

    qx = int((xs - xi + 6)/DELTA + 1)
    qy = int((ys - yi + 6)/DELTA + 1)

    x, y = np.linspace(xi - 3, xs + 3, qx), np.linspace(yi - 3, ys + 3, qy)
    X, Y = np.meshgrid(x, y)

    ZZ = f(X, Y)
    return x, y, X, Y, ZZ
# -- Creation of the splines.
def interpolacion(x, y, Z, X, Y):
    """
    Uses numpy's RectBivariateSpline to create an interpolation function of n and its gradient.

    Parameters:
    - x: The grid's x coordinates.
    - y: The grid's y coordinates.
    - Z: The callable index of refraction function.
    - X: A meshgrid of x coordinates.
    - Y: A meshgrid of y coordinates.

    Returns:
    - The bi-linear and bi-cubic splines of n and its gradient respectively. Additionally, the hessian of n.
    """
    # First and second derivatives of n.
    GradX,GradY = np.gradient(Z, DELTA, edge_order=2)
    GradXX,GradXY = np.gradient(GradX, DELTA, edge_order=2)
    GradXY,GradYY = np.gradient(GradY, DELTA, edge_order=2)

    # Generate the splines.
    z = interpolate.RectBivariateSpline(y, x, Z, kx=1, ky=1)
    funcgradX = interpolate.RectBivariateSpline(y, x, GradX, kx=3, ky=3)
    funcgradY = interpolate.RectBivariateSpline(y, x, GradY, kx=3, ky=3)
    grd = (funcgradX, funcgradY)
    funcgradXX = interpolate.RectBivariateSpline(y, x, GradXX, kx=3, ky=3)
    funcgradXY = interpolate.RectBivariateSpline(y, x, GradXY, kx=3, ky=3)
    funcgradYY = interpolate.RectBivariateSpline(y, x, GradYY, kx=3, ky=3)
    hess = np.array([[funcgradYY, funcgradXY], [funcgradXY, funcgradXX]])       # currently, there's no use for the hessian but it is calculated nonetheless

    return z, grd, hess


# RAY STEP ADVANCEMENT
# -- First order Taylor and analytic solution to cost function for angle determination.
def op1(i_angle, init_n, i_grad, i_unitv, i_vpos, m_i, grd, z, step):
    """
    Advances a ray one step.

    Parameters:
    - i_angle: The current group angle.
    - init_n: The initial index of refraction.
    - i_grad: The initial gradient of n.
    - i_unitv: The initial unit tangent vector.
    - i_vpos: The initial ray position.
    - m_i: the initial ray moment.
    - grd: The callable bi-cubic function containing the gradient of n.
    - z: The callable bi-linear function representing the index of refraction n.
    - step: The current step-size.

    Returns:
    - The final values of: position, gorup angle, index of refraction, gradient and the anisotropic component of n.
    """
    f_vpos = first_order_taylor(i_vpos, i_unitv, step)                          # ray advancement
    final_n, f_grad = n_gradient(f_vpos, grd, z)
    f_angle = theta_cost_t(init_n, i_angle, i_grad, f_grad, step)               # angle determination

    return f_vpos, f_angle, final_n, f_grad
# -- First order Taylor and solution to differential equation d_theta/d_s.
def op2(i_angle, init_n, i_grad, i_unitv, i_vpos, m_i, grd, z, step):
    """
    Advances a ray one step.

    Parameters:
    - i_angle: The current group angle.
    - init_n: The initial index of refraction.
    - i_grad: The initial gradient of n.
    - i_unitv: The initial unit tangent vector.
    - i_vpos: The initial ray position.
    - m_i: the initial ray moment.
    - grd: The callable bi-cubic function containing the gradient of n.
    - z: The callable bi-linear function representing the index of refraction n.
    - step: The current step-size.

    Returns:
    - The final values of: position, gorup angle, index of refraction, gradient and the anisotropic component of n.
    """
    f_vpos = first_order_taylor(i_vpos, i_unitv, step)                          # ray advancement
    final_n, f_grad = n_gradient(f_vpos, grd, z)
    f_angle = tfinal_2o(i_angle, step, init_n, final_n, i_grad, f_grad)         # angle determination

    return f_vpos, f_angle, final_n, f_grad
# -- Step by curvature method and analytic solution to cost function for angle determination (trapezoidal rule).
def op3(i_angle, init_n, i_grad, i_unitv, i_vpos, m_i, grd, z, step):
    """
    Advances a ray one step.

    Parameters:
    - i_angle: The current group angle.
    - init_n: The initial index of refraction.
    - i_grad: The initial gradient of n.
    - i_unitv: The initial unit tangent vector.
    - i_vpos: The initial ray position.
    - m_i: the initial ray moment.
    - grd: The callable bi-cubic function containing the gradient of n.
    - z: The callable bi-linear function representing the index of refraction n.
    - step: The current step-size.

    Returns:
    - The final values of: position, gorup angle, index of refraction, gradient and the anisotropic component of n.
    """
    f_vpos, ignore = curvature_t(i_angle, i_grad, i_unitv, init_n, i_vpos, grd, z, step)                # ray advancement
    final_n, f_grad = n_gradient(f_vpos, grd, z)

    if ignore:      # the curvature is high, calculate new angle.
        f_angle = tfinal_2o(i_angle, step, init_n, final_n, i_grad, f_grad)                             # angle determination
    else:           # curvature is low
        f_angle = i_angle

    return f_vpos, f_angle, final_n, f_grad
# -- Step by curvature method and analytic solution to cost function for angle determination (trapezoidal rule).
def op4(i_angle, init_n, i_grad, i_unitv, i_vpos, m_i, grd, z, step):
    """
    Advances a ray one step.

    Parameters:
    - i_angle: The current group angle.
    - init_n: The initial index of refraction.
    - i_grad: The initial gradient of n.
    - i_unitv: The initial unit tangent vector.
    - i_vpos: The initial ray position.
    - m_i: the initial ray moment.
    - grd: The callable bi-cubic function containing the gradient of n.
    - z: The callable bi-linear function representing the index of refraction n.
    - step: The current step-size.

    Returns:
    - The final values of: position, gorup angle, index of refraction, gradient and the anisotropic component of n.
    """
    f_vpos, ignore = curvature_t(i_angle, i_grad, i_unitv, init_n, i_vpos, grd, z, step)                # ray advancement
    final_n, f_grad = n_gradient(f_vpos, grd, z)

    if ignore:      # the curvature is high, calculate new angle.
        f_angle = theta_cost_t(init_n, i_angle, i_grad, f_grad, step)                                   # angle determination
    else:           # curvature is low
        f_angle = i_angle

    return f_vpos, f_angle, final_n, f_grad
# -- Step by curvature method (trapezoidal rule) and optimization of cost function for angle determination.
def op5(i_angle, init_n, i_grad, i_unitv, i_vpos, m_i, grd, z, step):
    """
    Advances a ray one step.

    Parameters:
    - i_angle: The current group angle.
    - init_n: The initial index of refraction.
    - i_grad: The initial gradient of n.
    - i_unitv: The initial unit tangent vector.
    - i_vpos: The initial ray position.
    - m_i: the initial ray moment.
    - grd: The callable bi-cubic function containing the gradient of n.
    - z: The callable bi-linear function representing the index of refraction n.
    - step: The current step-size.

    Returns:
    - The final values of: position, gorup angle, index of refraction, gradient and the anisotropic component of n.
    """
    f_vpos, ignore = curvature_t(i_angle, i_grad, i_unitv, init_n, i_vpos, grd, z, step)                # ray advancement
    final_n, f_grad = n_gradient(f_vpos, grd, z)

    if ignore:      # the curvature is high, calculate new angle.
        func = lambda theta_opt: (final_n*np.cos(theta_opt) - init_n*i_unitv[0] - impulse_t(i_grad[0], f_grad[0], step))**2 + (final_n*np.sin(theta_opt) - init_n*i_unitv[1] - impulse_t(i_grad[1], f_grad[1], step))**2
        f_angle = golden(func, i_angle - DELTA_G, i_angle + DELTA_G)                                    # angle determination
    else:           # curvature is low.
        f_angle = i_angle

    return f_vpos, f_angle, final_n, f_grad
# -- HySA (second order Taylor & solution to differential equation d_theta/d_s)
def op6(i_angle, init_n, i_grad, i_unitv, i_vpos, m_i, grd, z, step):
    """
    Advances a ray one step.

    Parameters:
    - i_angle: The current group angle.
    - init_n: The initial index of refraction.
    - i_grad: The initial gradient of n.
    - i_unitv: The initial unit tangent vector.
    - i_vpos: The initial ray position.
    - m_i: the initial ray moment.
    - grd: The callable bi-cubic function containing the gradient of n.
    - z: The callable bi-linear function representing the index of refraction n.
    - step: The current step-size.

    Returns:
    - The final values of: position, gorup angle, index of refraction, gradient and the anisotropic component of n.
    """
    f_vpos = second_order_taylor(i_vpos, i_unitv, step, init_n, i_grad, grd, z)       # ray advancement
    final_n, f_grad = n_gradient(f_vpos, grd, z)
    f_angle = tfinal_2o(i_angle, step, init_n, final_n, i_grad, f_grad)               # angle determination

    return f_vpos, f_angle, final_n, f_grad
# -- Step by second order Taylor and finite difference for angle determination.
def op7(i_angle, init_n, i_grad, i_unitv, i_vpos, m_i, grd, z, step):
    """
    Advances a ray one step.

    Parameters:
    - i_angle: The current group angle.
    - init_n: The initial index of refraction.
    - i_grad: The initial gradient of n.
    - i_unitv: The initial unit tangent vector.
    - i_vpos: The initial ray position.
    - m_i: the initial ray moment.
    - grd: The callable bi-cubic function containing the gradient of n.
    - z: The callable bi-linear function representing the index of refraction n.
    - step: The current step-size.

    Returns:
    - The final values of: position, gorup angle, index of refraction, gradient and the anisotropic component of n.
    """
    f_vpos = second_order_taylor(i_vpos, i_unitv, step, init_n, i_grad, grd, z)       # ray advancement
    final_n, f_grad = n_gradient(f_vpos, grd, z)
    VECTOR_LIST.append(f_vpos)
    f_angle = finite_diff()                                                           # angle determination
    VECTOR_LIST.pop(0)

    return f_vpos, f_angle, final_n, f_grad
# -- Step by second order Taylor and analytic solution to cost function for angle determination.
def op8(i_angle, init_n, i_grad, i_unitv, i_vpos, m_i, grd, z, step):
    """
    Advances a ray one step.

    Parameters:
    - i_angle: The current group angle.
    - init_n: The initial index of refraction.
    - i_grad: The initial gradient of n.
    - i_unitv: The initial unit tangent vector.
    - i_vpos: The initial ray position.
    - m_i: the initial ray moment.
    - grd: The callable bi-cubic function containing the gradient of n.
    - z: The callable bi-linear function representing the index of refraction n.
    - step: The current step-size.

    Returns:
    - The final values of: position, gorup angle, index of refraction, gradient and the anisotropic component of n.
    """
    f_vpos = second_order_taylor(i_vpos, i_unitv, step, init_n, i_grad, grd, z)       # ray advancement
    final_n, f_grad = n_gradient(f_vpos, grd, z)
    f_angle = theta_cost_t(init_n, i_angle, i_grad, f_grad, step)                     # angle determination

    return f_vpos, f_angle, final_n, f_grad
# -- Step by second order Taylor and optimization of cost function for angle determination.
def op9(i_angle, init_n, i_grad, i_unitv, i_vpos, m_i, grd, z, step):
    """
    Advances a ray one step.

    Parameters:
    - i_angle: The current group angle.
    - init_n: The initial index of refraction.
    - i_grad: The initial gradient of n.
    - i_unitv: The initial unit tangent vector.
    - i_vpos: The initial ray position.
    - m_i: the initial ray moment.
    - grd: The callable bi-cubic function containing the gradient of n.
    - z: The callable bi-linear function representing the index of refraction n.
    - step: The current step-size.

    Returns:
    - The final values of: position, gorup angle, index of refraction, gradient and the anisotropic component of n.
    """
    f_vpos = second_order_taylor(i_vpos, i_unitv, step, init_n, i_grad, grd, z)       # ray advancement
    final_n, f_grad = n_gradient(f_vpos, grd, z)

    func = lambda theta_opt: (final_n*np.cos(theta_opt) - init_n*i_unitv[0] - impulse_t(i_grad[0], f_grad[0], step))**2 + (final_n*np.sin(theta_opt) - init_n*i_unitv[1] - impulse_t(i_grad[1], f_grad[1], step))**2
    f_angle = golden(func, i_angle - DELTA_G, i_angle + DELTA_G)                      # angle determination

    return f_vpos, f_angle, final_n, f_grad
# -- Step by curvature method (trapezoidal rule) and optimization of *anisotropic* cost function for angle determination.
def op10(i_angle, init_n, i_grad, i_unitv, i_vpos, coef_i, grd, z, step):
    """
    Advances a ray one step.

    Parameters:
    - i_angle: The current group angle.
    - init_n: The initial index of refraction.
    - i_grad: The initial gradient of n.
    - i_unitv: The initial unit tangent vector.
    - i_vpos: The initial ray position.
    - coef_i: The initial anisotropic coeficient.
    - grd: The callable bi-cubic function containing the gradient of n.
    - z: The callable bi-linear function representing the index of refraction n.
    - step: The current step-size.

    Returns:
    - The final values of: position, gorup angle, index of refraction, gradient and the anisotropic component of n.
    """
    f_vpos, ignore = curvature_t(i_angle, i_grad, i_unitv, init_n, i_vpos, grd, z, step)    # ray advancement
    final_n, f_grad = n_gradient(f_vpos, grd, z)

    if ignore:      # the curvature is high, calculate new angle.

        mi_x = moment(init_n, i_angle, gamma, np.array([i_unitv[0], -i_unitv[1]**2]))
        mi_y = moment(init_n, i_angle, gamma, np.array([i_unitv[1], i_unitv[0]**2]))

        func = lambda theta_opt: (moment(final_n, theta_opt, gamma, np.array([np.cos(theta_opt), -np.sin(theta_opt)**2])) - mi_x - impulse_t(coef_i*i_grad[0], anisotropy(theta_opt, gamma)*f_grad[0], step))**2 + (moment(final_n, theta_opt, gamma, np.array([np.sin(theta_opt), np.cos(theta_opt)**2])) - mi_y - impulse_t(coef_i*i_grad[1], anisotropy(theta_opt, gamma)*f_grad[1], step))**2
        f_angle = golden(func, i_angle - DELTA_G, i_angle + DELTA_G)                        # angle determination

    else:           # curvature is low.
        f_angle = i_angle

    return f_vpos, f_angle, final_n, f_grad
# -- Step by second order Taylor and optimization of *anisotropic* cost function for angle determination.
def op11(i_angle, init_n, i_grad, i_unitv, i_vpos, coef_i, grd, z, step):
    """
    Advances a ray one step.

    Parameters:
    - i_angle: The current group angle.
    - init_n: The initial index of refraction.
    - i_grad: The initial gradient of n.
    - i_unitv: The initial unit tangent vector.
    - i_vpos: The initial ray position.
    - coef_i: The initial anisotropic coeficient.
    - grd: The callable bi-cubic function containing the gradient of n.
    - z: The callable bi-linear function representing the index of refraction n.
    - step: The current step-size.

    Returns:
    - The final values of: position, gorup angle, index of refraction, gradient and the anisotropic component of n.
    """

    f_vpos = second_order_taylor(i_vpos, i_unitv, step, init_n, i_grad, grd, z)         # ray advancement
    final_n, f_grad = n_gradient(f_vpos, grd, z)

    mi_x = moment(init_n, i_angle, gamma, np.array([i_unitv[0], -i_unitv[1]**2]))
    mi_y = moment(init_n, i_angle, gamma, np.array([i_unitv[1], i_unitv[0]**2]))

    func = lambda theta_opt: (moment(final_n, theta_opt, gamma, np.array([np.cos(theta_opt), -np.sin(theta_opt)**2])) - mi_x - impulse_t(coef_i*i_grad[0], anisotropy(theta_opt, gamma)*f_grad[0], step))**2 + (moment(final_n, theta_opt, gamma, np.array([np.sin(theta_opt), np.cos(theta_opt)**2])) - mi_y - impulse_t(coef_i*i_grad[1], anisotropy(theta_opt, gamma)*f_grad[1], step))**2
    f_angle = golden(func, i_angle - DELTA_G, i_angle + DELTA_G)                        # angle determination

    return f_vpos, f_angle, final_n, f_grad

def trazar(selected_func, z, grd, show, step, divisor, user_choice):
    """
    Function to integrate the equations of motion.

    Parameters:
    - selected_func: The selected method for integration.
    - z: The callable bi-linear function representing the index of refraction n.
    - grd: The callable bi-cubic function containing the gradient of n.
    - show: boolean flag that dictates if benchmarking data is calculated for the sharp interface scenario.
    - step: The current step-size.
    - divisor: for the fish-eye test, it signifies the number of segments the unit circle is partitioned
    - user_choice: Determines the selected scenario.

    Returns:
    - The vectors s_ray, d_ray, compute_times, errors (look below code for definitions).
    """

    def store_update_results(init_vpos, final_vpos, sim_dist, real_dist, delta_step, final_index, final_grad, final_angle):
        dist = np.linalg.norm(init_vpos - final_vpos)                                   # ammount of travelled distance in one step
        sim_dist += dist                                                                # accumulate the travelled distance
        real_dist += delta_step                                                         # accumulate the expected distance
        final_unitv = np.array((np.cos(final_angle), np.sin(final_angle)))              # calculate the final unit vector
        coef_f = anisotropy(final_angle, gamma)                                         # calculate the final coeficient for anisotropy (in isotropic conditions this should be 1)
        final_momenta = moments(final_angle, final_index, final_unitv, gamma)   # calculate the ray's final momenta
        return dist, sim_dist, real_dist, final_index, final_grad, coef_f, final_vpos, final_angle, final_unitv, final_momenta

    # grab some required constants
    gamma, ray_count, theta_v, pos_x, s, limx_i, limx_s, limy_i, limy_s, op_interface, op_fish, op_vert_heterogeneous, op_anisotropy = constants(user_choice)

    '''CREATE NEW VARIABLES'''
    if op_fish:
        max_size = N * divisor                                      # the maximum number of elements in the vector that will contain the coordinates of the ray
    else:
        max_size = int(np.ceil(s/step) + 1)                         # estimate the maximum number of elements in the vector that will contain the coordinates of the rays
    compute_times = np.zeros(ray_count)                             # vector to store completion times for each ray
    d_ray = np.zeros((3, ray_count))                                # vector to store: expected arc length, actual arc length, iteration number
    s_ray = np.zeros((max_size, 6, ray_count))                      # vector to store: x and y coordinates, x and y moments, traveltime and ray angle
    n_ray = np.zeros((max_size,ray_count))                          # vector to store: ray's indices of refraction
    errors = np.zeros(ray_count)                                    # vector to store outbound angle errors (only for sharp interface)
    loop_iter = 1                                                   # stores the initial iteration number for the integration

    for k in range (0, ray_count):
        '''INITIAL CONDITIONS'''
        if op_fish:
            i_vpos = pos_x.astype(np.float64)                       # initial position
        else:
            i_vpos = np.array((pos_x[k], -2)).astype(np.float64)    # initial position
        i_angle = theta_v[k]                                        # initial angle
        i_unitv = np.array((np.cos(i_angle), np.sin(i_angle)))      # initial unit tangent vector
        init_n,i_grad = n_gradient(i_vpos, grd, z)                  # initial index and its gradient
        coef_i = anisotropy(i_angle, gamma)                         # initial coeficient of anisotropy (should be one for isotropic cases)


        '''VARIABLE INITIALIZATION'''
        s_ray[0,0,k] = i_vpos[0]
        s_ray[0,1,k] = i_vpos[1]
        m_i = moments(i_angle,init_n, i_unitv, gamma)
        s_ray[0,2,k], s_ray[0,3,k] = m_i
        s_ray[0,5,k] = i_angle
        dist_sim, dist_real = 0, 0
        n_ray[0,k] = coef_i*init_n

        #############################
        ## INTEGRATION STARTS HERE ##
        #############################
        t1 = time.perf_counter()

        if selected_func == op7:
            ## THE METHOD OF FINITE DIFFERENCES HAS BEEN SELECTED FOR ANGLE DETERMINATION. ##
            ## A THIRD ORDER BACKWARD DIFFERENCE IS IMPLEMENTED. HENCE, THE RAY WILL FIRST ##
            ## BE ADVANCED TWO STEPS MANUALLY PRIOR TO AUTOMATIC DIFFERENTIATION.          ##
            loop_iter = 3                                           # the first two iterations are performed here. Start automatic integration loop at 3
            VECTOR_LIST.append(i_vpos)                              # populate the list with the initial position

            f_vpos = second_order_taylor(i_vpos, i_unitv, step, init_n, i_grad, grd, z)    # ray advancement
            final_n, f_grad = n_gradient(f_vpos, grd, z)
            VECTOR_LIST.append(f_vpos)                                                                      # populate the list with the second position
            x = VECTOR_LIST[1] - VECTOR_LIST[0]
            f_angle = np.arctan2(x[1], x[0])                                                                # first order angle determination
            # STORE AND UPDATE RESUTLS
            dist, dist_sim, dist_real, init_n, i_grad, coef_i, i_vpos, i_angle, i_unitv, m_i = store_update_results(i_vpos, f_vpos, dist_sim, dist_real, step, final_n, f_grad, f_angle)
            s_ray[1,0:2,k] = i_vpos
            s_ray[1,2,k], s_ray[1, 3, k] = m_i
            n_ray[1,k] = coef_i*init_n
            s_ray[1,4,k] = s_ray[0, 4, k] + dist*(n_ray[0,k] + n_ray[1,k])/2
            s_ray[1,5,k] = i_angle

            f_vpos = second_order_taylor(i_vpos, i_unitv, step, init_n, i_grad, grd, z)    # ray advancement
            final_n, f_grad = n_gradient(f_vpos, grd, z)
            VECTOR_LIST.append(f_vpos)                                                                      # populate the list with the third position
            x = 3*VECTOR_LIST[2] - 4*VECTOR_LIST[1] + VECTOR_LIST[0]
            f_angle = np.arctan2(x[1], x[0])                                                                # second order angle determination
            # STORE AND UPDATE RESUTLS
            dist, dist_sim, dist_real, init_n, i_grad, coef_i, i_vpos, i_angle, i_unitv, m_i = store_update_results(i_vpos, f_vpos, dist_sim, dist_real, step, final_n, f_grad, f_angle)
            s_ray[2, 0:2, k] = i_vpos
            s_ray[2, 2, k], s_ray[2, 3, k] = m_i
            n_ray[2,k] = coef_i*init_n
            s_ray[2, 4, k] = s_ray[1, 4, k] + dist*(n_ray[1,k] + n_ray[2,k])/2
            s_ray[2,5,k] = i_angle

        for i in range (loop_iter, max_size):
            ## AUTOMATIC INTEGRATION ##
            f_vpos, f_angle, final_n, f_grad = selected_func(i_angle, init_n, i_grad, i_unitv, i_vpos, coef_i, grd, z, step)
            # STORE AND UPDATE RESUTLS
            dist, dist_sim, dist_real, init_n, i_grad, coef_i, i_vpos, i_angle, i_unitv, m_i = store_update_results(i_vpos, f_vpos, dist_sim, dist_real, step, final_n, f_grad, f_angle)
            s_ray[i,0:2,k] = i_vpos
            s_ray[i,2,k], s_ray[i, 3, k] = m_i
            n_ray[i,k] = coef_i*init_n
            s_ray[i,4,k] = s_ray[i - 1, 4, k] + dist*(n_ray[i - 1,k] + n_ray[i,k])/2
            s_ray[i,5,k] = i_angle

            # THIS WILL END THE INTEGRATION IF A RAY HAS REACHED THE SIMULATION BOUNDARIES
            if i_vpos[0] > limx_s or i_vpos[0] < limx_i or i_vpos[1] > limy_s or i_vpos[1] < limy_i:
                break

        t2 = time.perf_counter()                                    # end timing the integration
        compute_times[k-1] = t2 - t1                                # this is the time taken to integrate the kth ray
        #############################
        ##  INTEGRATION ENDS HERE  ##
        #############################

        # STORE RESUTLS
        d_ray[0, k] = dist_real
        d_ray[1, k] = dist_sim
        d_ray[2, k] = i#-2
        if op_fish:
            dist_real = 2*np.pi
        if selected_func == op7:
            VECTOR_LIST.clear()

        if op_interface:
            """
            For the sharp interface scenario, this section calculates the error between the expected outward angle (as predicted by Snell's law) and the angle obtained in the simulation.
            These calculations assume that the last 10% or more of a ray's trajectory falls within the simulation boundaries after interacting with the interface. If simulation parameters
            (scenario dimensions or ray tracing settings) are modified, ensure the validity of these calculations.
            """
            if theta_v[k] < np.pi/4:
                angreal = 90 - 180*theta_v[k]/np.pi                                             # Convert the shooting angle to degrees and calculate the incident angle to the interface (reflection).
            else:
                if theta_v[k] == np.pi/4:
                    angreal = 0
                else:
                    angreal = 180*np.arcsin(np.sqrt(2)*np.sin(np.pi/2 - theta_v[k]))/np.pi      # Convert the shooting angle to degrees and calculate the incident angle to the interface (refraction).

            dummy = np.array((s_ray[:, 0, k][0:int(i) + 1], s_ray[:, 1, k][0:int(i) + 1]))      # Trims the s_ray coordinate vector to only include the calculated coordinates up to the ith index.
                                                                                                # This removes any excess positions from the original size of 'max_size' in s_ray, retaining only the relevant
                                                                                                # coordinates for the kth ray, ensuring accuracy and optimizing memory usage.
            distx = dummy[0, int(9.5*(i)/10)] - dummy[0, int(9*(i)/10)]                         # Calculates the travelled distance in the x coordinate in the second to last 5% of the rays' trajectory
            disty = dummy[1, int(9.5*(i)/10)] - dummy[1, int(9*(i)/10)]                         # Calculates the travelled distance in the y coordinate in the second to last 5% of the rays' trajectory

            angsim = 180*np.arctan(np.abs(distx/disty))/np.pi                                   # This is the simulated outward angle.

            error = np.abs((angsim - angreal)/1)
            errors[k] = error

            if show:
                a = s_ray[i, 0, k]                                              # The last obtained x coordinate for the k-th ray.
                b = s_ray[i, 1, k]                                              # The last obtained y coordinate for the k-th ray.
                c = angsim                                                      # This is the simulated outward angle.
                d = angreal                                                     # This is the expected outward angle.
                e = error                                                       # Error between real and expected.
                f = theta_v[k]*180/np.pi                                        # Initial shooting angle.

                def format_num(num):

                    if num < 0:
                        # If negative, account for minus sign
                        if abs(num) < 10:
                            # If the absolute value is less than 10
                            return "{: >10.8f}".format(num)
                        else:
                            return "{: >10.7f}".format(num)
                    else:
                        # If positive
                        if num < 10:
                            return "{: >10.9f}".format(num)
                        else:
                            return "{: >10.8f}".format(num)

                print(f"Coords: [ {format_num(a)} , {format_num(b)} ] | SimAng: {format_num(c)} | SnellAng: {format_num(d)} | Err: {format_num(e)} | InitAng: {format_num(f)}")


    return s_ray, d_ray, compute_times, errors

def search_delta(option, z, grd, step, divisor, user_choice):
    rays, _, _, errors = trazar(option, z, grd, False, step, divisor, user_choice)

    if op_interface:
        return np.mean(errors), np.max(errors)                                                              # return the mean deviation error among all rays with respect Snell's law
    if op_fish:
        return 100*np.linalg.norm(np.array([1, 0]) - np.array((rays[-1,0][0],rays[-1,1][0])))/(2*np.pi)     # return the percentage error of the last obtained coordinate
    if op_vert_heterogeneous or op_anisotropy:
        return rays[:, 2, :]                                                                                # return each ray's x-moment


def main(user_choice, op_interface, op_fish, op_vert_heterogeneous, op_anisotropy):
    global DELTA_S, DELTA_S_DIVISOR_FISHEYE, MAX_DEVIATION

    def graficar():

        ''' PLOT OF SCENARIO '''
        # Enable LaTeX rendering for all text
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams['font.family'] = 'serif'

        fig, ax = plt.subplots()

        ax.tick_params(axis='x', labelsize=24)
        ax.tick_params(axis='y', labelsize=24)

        if op_fish:
            v_ray = np.array((s_ray[:, 0], s_ray[:, 1]))
            ax.plot(v_ray[0], v_ray[1], 'r', linewidth=1.5)

            ax.set_xlim([limx_i, limx_s])
            ax.set_ylim([limy_i, limy_s])
        elif (op_vert_heterogeneous or op_anisotropy):

            while True:
                user_input = input('\n\033[92mDo you want to make a static (enter 1) or movie (enter 2) plot?: \033[0m ').strip()
                if user_input == '1':
                    for travel_time in np.arange(0.05, 0.6, 0.05):
                        valid_ray_coord = []                                        # list to store the coordinates of all rays with the desired travel time
                        angle_vector = []                                           # list to store the group angle at the desired travel time

                        for i in range(ray_count):
                            v_ray = np.array((s_ray[:, 0, i][0:int(d_ray[2, i])+1], s_ray[:, 1, i][0:int(d_ray[2, i])+1], s_ray[:, 5, i][0:int(d_ray[2, i])+1]))
                            t_ray = s_ray[:, 4, i][0:int(d_ray[2, i]) + 1]

                            # Check if this ray's data contains the desired travel time
                            if np.max(t_ray) >= travel_time:
                                x_interpolator = PchipInterpolator(t_ray, v_ray[0])
                                y_interpolator = PchipInterpolator(t_ray, v_ray[1])
                                angle_interpolator = PchipInterpolator(t_ray, v_ray[2])
                                x = x_interpolator(travel_time)
                                y = y_interpolator(travel_time)
                                angle = angle_interpolator(travel_time)

                                angle_vector.append(angle)
                                valid_ray_coord.append([x, y])

                            # Only plot every other ray. Un-plotted rays still aid in improving interpolation of wavefronts.
                            if i % 2 == 0:
                                ax.plot(v_ray[0], v_ray[1], color='r', linewidth=1.5, zorder=2)     # make a plot of the ray
                                ax.plot(x, y, color='g', marker='o', markersize=5, zorder=3)        # draw a point in the ray that matches the desired travel time

                        if len(valid_ray_coord) > 1:                                # ensure there are enough points to interpolate
                            valid_ray_coord = np.array(valid_ray_coord)             # convert valid_ray_coord to numpy array for processing
                            angle_vector = np.array(angle_vector)                   # convert angle_vector to numpy array for processing

                            indices = np.argsort(valid_ray_coord[:, 1])             # sort wavefront points based on their y-values (or x-values if preferred) to strictly increasing for interpolation

                            ray_coord_sorted = valid_ray_coord[indices]             # sorted wavefront points

                            pchip_interpolator = PchipInterpolator(ray_coord_sorted[:, 1], ray_coord_sorted[:, 0])  # create the wavefront PCHIP interpolator with sorted valid points
                            derivative_interp = pchip_interpolator.derivative()
                            dy_dx_original = derivative_interp(ray_coord_sorted[:, 1])                              # calculate the derivative of the interp (tangents to wavefront)

                            # Calculate tangent and normal angles
                            tangent_angles = np.pi/2 - np.arctan(dy_dx_original)
                            normal_angles = tangent_angles - np.pi / 2

                            BLUE_COLOR = "\033[94m"
                            RESET_COLOR = "\033[0m"
                            RED_COLOR = "\033[91m"
                            print(f"{BLUE_COLOR}\nTravel Time: {travel_time:.2f}{RESET_COLOR} - all angles expressed in radians")
                            for idx, (angle_diff, angle, normal_angle) in enumerate(zip(np.absolute(angle_vector - normal_angles), angle_vector, normal_angles)):
                                # Align positive and negative numbers
                                angle_diff_str = f"{angle_diff: .4f}"
                                angle_str = f"{angle: .4f}"
                                normal_angle_str = f"{normal_angle: .4f}"
                                # Use formatted string with a fixed width for ray numbers
                                ray_number_str = f"Ray {idx + 1:<2}:"
                                print(f"{RED_COLOR}{ray_number_str}{RESET_COLOR} Angle Diff = {angle_diff_str}, Ray Angle = {angle_str}, Normal Angle = {normal_angle_str}")


                            # Interpolate x-values for plotting
                            y_fine = np.linspace(min(ray_coord_sorted[:, 1]), max(ray_coord_sorted[:, 1]), 100)
                            x_fine = pchip_interpolator(y_fine)
                            ax.plot(x_fine, y_fine, 'b--', zorder=4)        # create a plot for the wavefront

                            ax.set_xlim([limx_i, 4])
                            ax.set_ylim([-2, 0])

                    break
                elif user_input == '2':
                    lines = []  # List to keep track of lines to update during animation
                    # Initialize ray plots (constant across frames)
                    for i in range(ray_count):
                        v_ray = np.array((s_ray[:, 0, i][0:int(d_ray[2, i])+1], s_ray[:, 1, i][0:int(d_ray[2, i])+1], s_ray[:, 5, i][0:int(d_ray[2, i])+1]))
                        # Only plot every other ray
                        if i % 2 == 0:
                            line, = ax.plot(v_ray[0], v_ray[1], 'r', linewidth=1.5, zorder=2)
                            lines.append(line)

                    wavefront_line, = ax.plot([], [], 'b--', zorder=4)      # line for wavefront (updated each frame)

                    green_dots = []                                         # initialize an empty list to keep track of the green dots

                    # Animation update function
                    def update(frame):
                        travel_time = 0.01 + frame * 0.01
                        valid_ray_coord = []

                        # Clear the previous green dots from the axes for each frame
                        for dot in green_dots:
                            dot.remove()
                        green_dots.clear()

                        for i in range(ray_count):
                            v_ray = np.array((s_ray[:, 0, i][0:int(d_ray[2, i])+1], s_ray[:, 1, i][0:int(d_ray[2, i])+1], s_ray[:, 5, i][0:int(d_ray[2, i])+1]))
                            t_ray = s_ray[:, 4, i][0:int(d_ray[2, i]) + 1]
                            if np.max(t_ray) >= travel_time:

                                x_interpolator = PchipInterpolator(t_ray, v_ray[0])
                                y_interpolator = PchipInterpolator(t_ray, v_ray[1])
                                x = x_interpolator(travel_time)
                                y = y_interpolator(travel_time)

                                valid_ray_coord.append([x, y])

                                if i % 2 == 0:
                                    # Plot and store new green dots for every other ray
                                    dot, = ax.plot(x, y, 'go', markersize=5, zorder=3)
                                    green_dots.append(dot)

                        if valid_ray_coord:
                            valid_ray_coord = np.array(valid_ray_coord)
                            indices = np.argsort(valid_ray_coord[:, 1])
                            ray_coord_sorted = valid_ray_coord[indices]
                            pchip_interpolator = PchipInterpolator(ray_coord_sorted[:, 1], ray_coord_sorted[:, 0])

                            y_fine = np.linspace(min(ray_coord_sorted[:, 1]), max(ray_coord_sorted[:, 1]), 100)
                            x_fine = pchip_interpolator(y_fine)
                            wavefront_line.set_data(x_fine, y_fine)

                        return lines + green_dots + [wavefront_line]


                    # Create animation
                    ani = FuncAnimation(fig, update, frames=45, blit=True)

                    ax.set_xlim([limx_i, 4])
                    ax.set_ylim([-2, 0])

                    break
                else:
                    print("Invalid input. Please enter 1 or 2.")

            ax.annotate('', xy=(-2, -2), xytext=(4.1, -2), arrowprops=dict(arrowstyle="<|-", color='black', linewidth=1.5, mutation_scale=30))
            ax.text(4.05, -1.95, '$x$', fontsize=26, ha='left')

            ax.annotate('', xy=(-1.999, -2), xytext=(-1.999, 0.1), arrowprops=dict(arrowstyle="<|-", color='black', linewidth=1.5, mutation_scale=30))
            ax.text(-1.95, 0.05, '$y$', fontsize=26, va='bottom')

        else:
            for i in range(ray_count):
                v_ray = np.array((s_ray[:, 0, i][0:int(d_ray[2, i])+1], s_ray[:, 1, i][0:int(d_ray[2, i])+1]))
                ax.plot(v_ray[0], v_ray[1], color='r', linewidth=1.5)

            ax.set_xlim([limx_i, limx_s])
            ax.set_ylim([limy_i, limy_s])

            ax.annotate('', xy=(-2, -2), xytext=(20.5, -2), arrowprops=dict(arrowstyle="<|-", color='black', linewidth=1.5, mutation_scale=30))
            ax.text(20.2, -1.8, '$x$', fontsize=26, ha='left')

            ax.annotate('', xy=(-1.999, -2), xytext=(-1.999, 4.5), arrowprops=dict(arrowstyle="<|-", color='black', linewidth=1.5, mutation_scale=30))
            ax.text(-1.8, 4.2, '$y$', fontsize=26, va='bottom')

        ax.set_aspect('equal')
        pcm = ax.pcolormesh(X, Y, Z, cmap='Greys')

        divider = make_axes_locatable(ax)                                                   # create a colorbar axis
        cax = divider.append_axes("bottom", size="8%", pad=0.4)                             # adjust size and pad as needed

        cbar = fig.colorbar(pcm, cax=cax, orientation='horizontal')
        cbar.set_ticks([cbar.vmin, cbar.vmax])                                              # set colorbar ticks
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))                       # set colorbar tick labels to 2 decimal points
        cbar.ax.tick_params(labelsize=24)                                                   # set colorbar tick label font size
        cbar.ax.set_xlabel("Refractive index $n$", fontsize=28, labelpad=-15)

        # Rotate colorbar tick labels to horizontal and align them with the ticks
        for label in cbar.ax.xaxis.get_majorticklabels():
            label.set_rotation('horizontal')
            label.set_horizontalalignment('center')

        if op_interface:
            ax.set_xticks(np.arange(limx_i, limx_s + 2, 2))                                 # adjust these numbers as necessary
            ax.set_yticks(np.arange(limy_i, limy_s + 1, 1))

        plt.tight_layout()

        if (op_vert_heterogeneous or op_anisotropy) and user_input == '2':
            while True:
                movie = input('\n\033[92mDo you want to save a video? (Y/N):\033[0m ').strip().lower()
                if movie == 'y':
                    ani.save('wavefront_propagation.mp4', writer='ffmpeg')
                    break
                elif movie == 'n':
                    break
                else:
                    print("Invalid input. Please enter Y or N.")

        plt.show()


        if op_vert_heterogeneous or op_anisotropy:
            ''' RAY PARAMETER (MOMENTA) PLOT '''
            fig, ax = plt.subplots()

            ax.tick_params(axis='x', labelsize=24)
            ax.tick_params(axis='y', labelsize=24)

            for i in range(1, ray_count - 1):
                if i % 2 == 0:
                    v_ray = np.array((s_ray[0:int(d_ray[2, i]) + 1, 2, i], s_ray[0:int(d_ray[2, i]) + 1, 3, i]))

                    x_values = np.arange(len(v_ray[0])) * DELTA_S
                    ax.plot(x_values, v_ray[0], color='r', linewidth=1.5)
                    avg_param = np.mean(v_ray[0])
                    range_param = np.std(v_ray[0])
                    # comment the following line for less verbosity
                    ax.annotate(f"Ray {int(i/2)}, $\Delta p_x$={100*np.abs(v_ray[0][0]-avg_param)/v_ray[0][0]:.2e}\%, $\overline p_x$={avg_param:.2e}, CV={100*range_param/avg_param:.4f}\%", xy=(x_values[-1], v_ray[0][-1]), textcoords="offset points", xytext=(-100,6), ha='center', fontsize=16, weight='bold')
                    # uncomment the following line for less verbosity
                    # ax.annotate(f"Ray {int(i/2)}, {v_ray[0][0]:.2e}", xy=(x_values[-1], v_ray[0][-1]), textcoords="offset points", xytext=(-10,5), ha='center', fontsize=20, weight='bold')

            ax.set_xlabel('Ray length', fontsize=24)
            ax.set_ylabel('Ray parameter $p_x$', fontsize=24)

            plt.show()

        return 0

    scenarios = [
        "the sharp interface scenario",
        "the fish-eye scenario",
        "the isotropic vertically heterogeneous scenario",
        "the anisotropic vertically heterogeneous scenario"
        ]

    if not(op_anisotropy):
        """
        User has selected an isotropic scenario. Proceed to prompt for a suitable algorithm
        """
        while True:
            messages = [
                " 1st order Taylor  + analytical 2-point momentum-impulse",
                " 1st order Taylor  + d_theta/d_s Runge-Kutta (AnDF)",
                " 2-point curvature + d_theta/d_s Runge-Kutta",
                " 2-point curvature + analytical 2-point momentum-impulse",
                " 2-point curvature + optimized  2-point momentum-impulse",
                " 2nd order Taylor  + d_theta/d_s Runge-Kutta (HySA)",
                " 2nd order Taylor  + 4-point difference method (MxSA)",
                " 2nd order Taylor  + analytical 2-point momentum-impulse",
                " 2nd order Taylor  + optimized  2-point momentum-impulse"
                ]

            print("\033[1m\nChoose an Algorithm Option. (Methods for ray advancement + angle determination)\033[0m")
            print("───────────────────────────────────────────────────────────────────────────────")
            print(f"• \033[1;94m1.\033[0m Run {messages[0]}...")
            print(f"• \033[1;94m2.\033[0m Run {messages[1]}...")
            print(f"• \033[1;94m3.\033[0m Run {messages[2]}...")
            print(f"• \033[1;94m4.\033[0m Run {messages[3]}...")
            print(f"• \033[1;94m5.\033[0m Run {messages[4]}...")
            print(f"• \033[1;94m6.\033[0m Run {messages[5]}...")
            print(f"• \033[1;94m7.\033[0m Run {messages[6]}...")
            print(f"• \033[1;94m8.\033[0m Run {messages[7]}...")
            print(f"• \033[1;94m9.\033[0m Run {messages[8]}...")

            user_choice_1 = input('\n\033[92mEnter the number of your choice:\033[0m ')


            if user_choice_1 == "1":
                option = op1
                break
            elif user_choice_1 == "2":
                option = op2
                break
            elif user_choice_1 == "3":
                option = op3
                break
            elif user_choice_1 == "4":
                option = op4
                break
            elif user_choice_1 == "5":
                option = op5
                break
            elif user_choice_1 == "6":
                option = op6
                break
            elif user_choice_1 == "7":
                option = op7
                break
            elif user_choice_1 == "8":
                option = op8
                break
            elif user_choice_1 == "9":
                option = op9
                break
            else:
                print("Invalid choice. Please choose 1, 2, 3, 4, 5, 6, 7, 8 or 9.")
                continue
    else:
        """
        User has selected an anisotropic scenario. Optimization algorithm for theta is the only suitable choice
        """
        while True:
            messages = [
                " 2-point curvature + optimized 2-point momentum-impulse",
                " 2nd order Taylor  + optimized 2-point momentum-impulse"
                ]

            print("\033[1m\nChoose an Algorithm Option. (Methods for ray advancement + angle determination)\033[0m")
            print("───────────────────────────────────────────────────────────────────────────────")
            print(f"• \033[1;94m1.\033[0m Run {messages[0]}...")
            print(f"• \033[1;94m2.\033[0m Run {messages[1]}...")

            user_choice_1 = input('\n\033[92mEnter the number of your choice:\033[0m ')


            if user_choice_1 == "1":
                option = op10
                break
            elif user_choice_1 == "2":
                option = op11
                break
            else:
                print("Invalid choice. Please choose 1 or 2.")
                continue

    while True:
        delta_s_search = input('\n\033[92mDo you want to find an appropriate DELTA_S? (Y/N):\033[0m ').strip().lower()

        if delta_s_search == 'y':
            print("\nFINDING SUITABLE DIVISOR...")

            if op_interface:
                divisors = np.arange(DELTA_S_DIVISOR_UPPER_LIMIT, DELTA_S_DIVISOR_LOWER_LIMIT - DELTA_STEP, -DELTA_STEP)
                delta_s_options = SIGMA / divisors                                  # search interval divided in steps of DELTA_STEP
            elif op_fish:
                MAX_DEVIATION = 5
                divisors = np.arange(DELTA_S_DIVISOR_FISHEYE_UPPER_LIMIT, DELTA_S_DIVISOR_FISHEYE_LOWER_LIMIT - DELTA_STEP_FISHEYE, -DELTA_STEP_FISHEYE)
                delta_s_options = 2*np.pi / divisors                                # search interval divided in steps of DELTA_STEP_FISHEYE
            else:
                MAX_DEVIATION = 0.05
                divisors = np.arange(DELTA_S_DIVISOR_VERT_UPPER_LIMIT, DELTA_S_DIVISOR_VERT_LOWER_LIMIT - 2*DELTA_STEP, -DELTA_STEP)
                delta_s_options = SIGMA / divisors                                  # search interval divided in steps of DELTA_STEP_FISHEYE


            num_opt = len(delta_s_options)                                          # the number of options for DELTA_S
            error_prom = np.zeros_like(delta_s_options)                             # create the vector to store deviation errors
            with ProcessPoolExecutor(max_workers=THREADS-2) as executor:            # THREADS-2 processes will be created to concurrently calculate the deviation error for each DELTA_S in delta_s_options
                results = list(executor.map(search_delta, [option]*num_opt, [z]*num_opt, [grd]*num_opt, delta_s_options, divisors+1, [user_choice]*num_opt))

            if op_interface:
                error_prom = [result[0] for result in results]
                max_error = [result[1] for result in results]
                def find_index(errors, max_errors):
                    # If all errors are either above or below the threshold, return None
                    if not any(error > MAX_DEVIATION for error in errors) or not any(error < MAX_DEVIATION for error in errors):
                        return None

                    for i in reversed(range(len(errors))):
                        if errors[i] < MAX_DEVIATION and max_errors[i] < 0.8:           # check for condition on both avg errors and max_errors
                            if all(e < MAX_DEVIATION for e in errors[:i]) and all(e < 0.8 for e in max_errors[:i]):            # check previous errors to ensure all are below the threshold
                                return i
                    return None

                index = find_index(error_prom, max_error)

            elif op_fish:
                error_prom = [result for result in results]

                def find_index(errors):
                    # If all errors are either above or below the threshold, return None
                    if not any(error > MAX_DEVIATION for error in errors) or not any(error < MAX_DEVIATION for error in errors):
                        return None

                    for i in range(len(errors)):
                        if errors[i] > MAX_DEVIATION:                                   # check for first error above MAX_DEVIATION %
                            return i - 1
                    return None

                index = find_index(error_prom)

            else:
                x_moment = [result for result in results]

                scenario_average_CV = np.zeros_like(delta_s_options)
                for j in range(len(x_moment)):
                    ray_moment_CVs = np.zeros(ray_count - 2)
                    for i in range(1, ray_count - 1):
                        masked = np.ma.masked_equal(x_moment[j][:,i],0).compressed()    # exclude zeros from x_moment array
                        ray_moment_CVs[i - 1] = 100*np.std(masked)/np.mean(masked)
                    scenario_average_CV[j] = np.mean(ray_moment_CVs)


                def find_index(errors):
                    # If all errors are either above or below the threshold, return None
                    if not any(error > MAX_DEVIATION for error in errors) or not any(error < MAX_DEVIATION for error in errors):
                        return None

                    for i in range(len(errors)):
                        if i > 1:
                            if errors[i] > MAX_DEVIATION:                                   # check for first error above MAX_DEVIATION %
                                if all(e < MAX_DEVIATION for e in errors[:i - 1]):            # check previous errors to ensure all are below the threshold
                                    return i - 1
                    return None

                index = find_index(scenario_average_CV)

            if index is not None:
                if op_fish:
                    DELTA_S_DIVISOR_FISHEYE = round(divisors[index])
                    print("Found best divisor! Using DELTA_S = 2*pi / {:.0f}".format(DELTA_S_DIVISOR_FISHEYE))
                    DELTA_S = 2*np.pi / DELTA_S_DIVISOR_FISHEYE
                else:
                    DELTA_S_DIVISOR = round(divisors[index],2)
                    print("Found best divisor! Using DELTA_S = SIGMA / {:.2f}".format(DELTA_S_DIVISOR))
                    DELTA_S = SIGMA / DELTA_S_DIVISOR

                t1 = time.perf_counter()
                s_ray, d_ray, compute_times, errors = trazar(option, z, grd, True, DELTA_S, DELTA_S_DIVISOR_FISHEYE+1, user_choice)
                t2 = time.perf_counter()

                print("\nRESULTS")
                if op_fish:
                    print("Closure error ", 100*np.linalg.norm(np.array([1, 0]) - s_ray[-1, 0:2, 0])/(2*np.pi), "%")
                elif op_interface:
                    error_prom = np.mean(errors)
                    print("Average ray error: ", error_prom, "degrees")
                else:
                    ray_moment_CVs = np.zeros(ray_count - 2)
                    for i in range(1, ray_count - 1):
                        masked = np.ma.masked_equal(s_ray[:,2,i],0).compressed()
                        ray_moment_CVs[i - 1] = 100*np.std(masked)/np.mean(masked)
                    print("Average ray Coefficient of Variation: ", np.mean(ray_moment_CVs))
                break
            else:
                print("\nNo suitable divisor was found. Try using another search interval (*_UPPER_LIMIT, *_LOWER_LIMIT). Exiting...")
                sys.exit()

        elif delta_s_search == 'n':
            while True:
                selection = input(f'\n\033[92mDo you want to use the calibrated DELTA_S? If not, the default DELTA_S will be used. (Y/N)?:\033[0m ').strip().lower()
                if selection == 'y':
                    if op_interface or op_vert_heterogeneous:
                        if user_choice_1 == "1":                # The following DELTA_S values were determined based on simulations that used a grid size of SIGMA/3.
                            DELTA_S = SIGMA / 38.64             # For each algorithm, this script finds an optimal DELTA_S value extracted within the interval
                        elif user_choice_1 == "2":              # (DELTA_S_DIVISOR_LOWER_LIMIT, DELTA_S_DIVISOR_UPPER_LIMIT). Particularly for algorithms 1, 2, and 7,
                            DELTA_S = SIGMA / 38.37             # a high-resolution integration (low DELTA_S values) was required to achieve a mean percentage error of
                        elif user_choice_1 == "3":              # MAX_DEVIATION = 0.2 (an arbitrary threshold) and maximum angle error of 0.8 in the interface scenario.
                            DELTA_S = SIGMA / 2.34              # Thus, the search interval for these algorithms needs to be kept narrow around the recorded values to
                        elif user_choice_1 == "4":              # ensure a timely response. Be warned that expanding this interval increases the search time considerably,
                            DELTA_S = SIGMA / 2.53              # potentially taking hours for the above mentioned algorithms.
                        elif user_choice_1 == "5":
                            DELTA_S = SIGMA / 2.53
                        elif user_choice_1 == "6":
                            DELTA_S = SIGMA / 2.55
                        elif user_choice_1 == "7":
                            DELTA_S = SIGMA / 30.05
                        elif user_choice_1 == "8":
                            DELTA_S = SIGMA / 2.74
                        elif user_choice_1 == "9":
                            DELTA_S = SIGMA / 2.74
                    elif op_fish:
                        if user_choice_1 == "1":                # There are two sets of calibrated values in this section. Manually edit and choose from one. Both of them
                            DELTA_S_DIVISOR_FISHEYE = 4587      # determined using a grid size of SIGMA/3. The current values in the left correspond to the number of segments
                        elif user_choice_1 == "2":              # a unit circle must be partitioned so that the resulting length (DELTA_S) is approximately the same as the
                            DELTA_S_DIVISOR_FISHEYE = 4556      # interface scenario above. The other set shown below, was determined so that all methods resulted in a
                        elif user_choice_1 == "3":              # closure error of MAX_DEVIATION = 5 percent or less when simulating N = 10 turns around the fisheye's unit
                            DELTA_S_DIVISOR_FISHEYE = 278       # circle. Time considerations from the interface scenario remain valid for the first set of values, so it is
                        elif user_choice_1 == "4":              # recommended to define an adequate narrow search interval:
                            DELTA_S_DIVISOR_FISHEYE = 300       # (DELTA_S_DIVISOR_FISHEYE_LOWER_LIMIT, DELTA_S_DIVISOR_FISHEYE_UPPER_LIMIT) around these if time is scarse.
                        elif user_choice_1 == "5":
                            DELTA_S_DIVISOR_FISHEYE = 300
                        elif user_choice_1 == "6":
                            DELTA_S_DIVISOR_FISHEYE = 303
                        elif user_choice_1 == "7":              # The second set is as follows: 149, 169, 182, 179, 179, 182, 191, 179, 179 respectively.
                            DELTA_S_DIVISOR_FISHEYE = 3567
                        elif user_choice_1 == "8":
                            DELTA_S_DIVISOR_FISHEYE = 325
                        elif user_choice_1 == "9":
                            DELTA_S_DIVISOR_FISHEYE = 325
                        DELTA_S = 2*np.pi / DELTA_S_DIVISOR_FISHEYE
                    else:
                        if user_choice_1 == "1":
                            DELTA_S = SIGMA / 2.53
                        else:
                            DELTA_S = SIGMA / 2.74
                    break
                elif selection == 'n':
                    break
                else:
                    print("Invalid input. Please enter 'Y' or 'N'.")

            t1 = time.perf_counter()
            s_ray, d_ray, compute_times, errors = trazar(option, z, grd, False, DELTA_S, DELTA_S_DIVISOR_FISHEYE+1, user_choice)
            t2 = time.perf_counter()

            print("\nRESULTS")
            if op_fish:
                print("Closure error ", 100*np.linalg.norm(np.array([1, 0]) - s_ray[-1, 0:2, 0])/(2*np.pi), "%")
            elif op_interface:
                s_ray, d_ray, compute_times, errors = trazar(option, z, grd, True, DELTA_S, DELTA_S_DIVISOR_FISHEYE+1, user_choice)     # display ray errors
                error_prom = np.mean(errors)
                print("Average ray error: ", error_prom, "degrees")
            else:
                ray_moment_CVs = np.zeros(ray_count - 2)
                for i in range(1, ray_count - 1):
                    masked = np.ma.masked_equal(s_ray[:,2,i],0).compressed()
                    ray_moment_CVs[i - 1] = 100*np.std(masked)/np.mean(masked)
                print("Average ray Coefficient of Variation: ", np.mean(ray_moment_CVs))


            break

        else:
            print("Invalid input. Please enter 'Y' or 'N'.")

    print("Total travelled distance: ", np.sum(d_ray[1,:]))
    while True:
        warmup = 100
        trial = 100
        approx_seconds = (t2-t1)*(warmup+trial)*(THREADS/2-1)
        approx_minutes = np.round(approx_seconds / 60, 1)

        print("\033[1m\nBenchmarking Process\033[0m")
        print("────────────────────")
        print(f"• \033[1mPurpose:\033[0m To accurately measure the execution time per scenario, {warmup} warmup trials")
        print(f"           will be run to prepare the system, followed by {trial} repetitions of the experiment.")
        print(f"• \033[1mEstimated Duration:\033[0m Approximately {approx_minutes} minutes, granted the CPU doesn't slow down due to")
        print(f"                      overheating (thermal throttle), energy or performance settings")
        print(f"                      (e.g. a laptop), or similar.")
        print(f"• \033[1mNote:\033[0m For optimal results, it is recommended to close any unused running programs.")

        user_input = input('\n\033[92mDo you want to proceed with the benchmark? (Y/N):\033[0m ').strip().lower()

        if user_input == 'n':
            break
        elif user_input == 'y':

            print("\nWarming up...", end='', flush=True)                                    # Warmup stage, trying to prime CPU cache
            # for j in range(0, warmup):
            #     with ProcessPoolExecutor(max_workers=THREADS-2) as executor:                # Leave 2 cpu threads to maintain system responsiveness.
            #         _ = [executor.submit(trazar, option, z, grd, False, DELTA_S, DELTA_S_DIVISOR_FISHEYE+1, user_choice) for _ in range(THREADS-2)]
            print(" done")

            print(f"Benchmarking{messages[int(user_choice_1) - 1]} in {scenarios[int(user_choice) - 1]}...", end='', flush=True)        # Start the benchmark process.

            benchmarks = []
            arr = np.zeros(trial*(int(THREADS/2)-1))                                        # Array where each of its elements stores the completion time in a single experiment.
            while True:
                index = 0                                                                   # Keep track of the current index in arr
                for j in range(0, trial):
                    with ProcessPoolExecutor(max_workers=int(THREADS/2)-1) as executor:     # Leave 2 cpu threads for sys admin tasks and to minimize variation.
                        futures = [executor.submit(trazar, option, z, grd, False, DELTA_S, DELTA_S_DIVISOR_FISHEYE+1, user_choice) for _ in range(int(THREADS/2)-1)]
                    results = [future.result() for future in futures]

                    for result in results:
                        arr[index] = np.sum(result[2])                                      # Store completion time of 1 scenario with a worker_process.
                        index += 1

                cleaned_arr = remove_outliers_iqr(arr)                                      # Filter outliers.
                new_size = len(cleaned_arr)
                benchmarks.append(np.median(cleaned_arr[int(-0.3 * new_size):]))            # Calculate the median of the last 30% filtered array and append it to "benchmarks" list.

                if len(benchmarks) >= 2:
                    diff_percent = 100 * abs(benchmarks[-1] - benchmarks[-2]) / max(benchmarks[-1], benchmarks[-2])
                    if diff_percent < 0.5:
                        break                                                               # Exit the loop if the last two benchmarks have less than 0.5% difference.

            final_result = np.mean(benchmarks[-2:])                                         # Average of the last two benchmarks
            print(" done")

            print("\nCompletion time per scenario:", final_result, "seconds.")
            break
        else:
            print("\nInvalid input. Please enter 'Y' or 'N'.")

    graficar()
    return 0


if __name__ == "__main__":

    # Initialize every scenario as 'not selected'.
    op_interface, op_fish, op_vert_heterogeneous, op_anisotropy = 0, 0, 0, 0

    while True:
        print("\033[1m\nChoose a Test Option\033[0m")
        print("────────────────────")
        print("• \033[1;94m1.\033[0m Sharp changes in n(x,y)")
        print("• \033[1;94m2.\033[0m Gradual changes in n(x,y)")
        print("• \033[1;94m3.\033[0m Vertically heterogeneous - isotropic")
        print("• \033[1;94m4.\033[0m Vertically heterogeneous - anisotropic")

        user_choice = input('\n\033[92mEnter the number of your choice:\033[0m ')

        if user_choice == "1":
            print(f"Sharp interface scenario will be used.")
            f = interface                                                   # assign scenario to f so it will be available to function genZ
            break
        elif user_choice == "2":
            print(f"Maxwell's Fish-Eye scenario will be used.")
            f = fisheye                                                     # assign scenario to f so it will be available to function genZ
            break
        elif user_choice == "3":
            print(f"Isotropic vertically heterogeneous scenario will be used.")
            f = vert_heterogeneous                                          # assign scenario to f so it will be available to function genZ
            break
        elif user_choice == "4":
            print(f"Running anisotropic vertically heterogeneous scenario...")
            f = vert_heterogeneous                                          # assign scenario to f so it will be available to function genZ
            break
        else:
            print("Invalid choice. Please choose 1, 2, 3 or 4.")
            continue
    gamma, ray_count, theta_v, pos_x, s, limx_i, limx_s, limy_i, limy_s, op_interface, op_fish, op_vert_heterogeneous, op_anisotropy = constants(user_choice)

    # Create the callable bi-linear and bi-cubic functions z and grd (spatial index of refraction and its gradient respectively)
    linx, liny, X, Y, Z = genZ(limx_i, limx_s, limy_i, limy_s)
    z, grd, hess = interpolacion(linx, liny, Z, X, Y)

    main(user_choice, op_interface, op_fish, op_vert_heterogeneous, op_anisotropy)
