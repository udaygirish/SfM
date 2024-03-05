# Given 6 3D-2D Correspondences, Camera Pose(C,R) refine camera pose using
# minimization of reprojection error using scipy.optimize.least_squares
import os 
import sys
sys.path.append("../")
import numpy as np
import cv2
from scipy.optimize import least_squares, minimize, differential_evolution
from scipy.spatial.transform import Rotation

import numpy as np
from scipy.optimize import minimize
def project_points(X, R, t, K):
    """Project 3D points X using rotation R, translation t, and camera matrix K."""
    P = np.dot(K, np.hstack((R, t.reshape(-1, 1))))
    projected_points = np.dot(P, np.vstack((X.T, np.ones((1, X.shape[0])))))
    projected_points /= projected_points[2, :]
    return projected_points[:2, :].T

def reprojection_error_euler(params, X, x, K):
    """Reprojection error for PnP using Euler angles."""
    R = euler_angles_to_rotation_matrix(params[:3])
    t = params[3:]
    projected_points = project_points(X, R, t, K)
    return np.sum((projected_points - x)**2)

def reprojection_error_quaternion(params, X, x, K):
    """Reprojection error for PnP using quaternions."""
    q = params[:4]
    t = params[4:]
    R = Rotation.from_quat(q).as_matrix()
    projected_points = project_points(X, R, t, K)
    return np.sum((projected_points - x)**2)

def euler_angles_to_rotation_matrix(euler_angles):
    """Convert Euler angles to a 3x3 rotation matrix."""
    phi, theta, psi = euler_angles
    Rx = np.array([[1, 0, 0],[0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],[0, 1, 0],[-np.sin(theta), 0, np.cos(theta)]])
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],[np.sin(psi), np.cos(psi), 0],[0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def euler_angles_from_rotation_matrix(R):
    """Convert a 3x3 rotation matrix to Euler angles."""
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def NonLinearPnP_Rot(K, initial_R, initial_t, X, image_points):
    """Nonlinear refinement of rotation and translation using PnP with Euler angles."""
    initial_params = np.concatenate([euler_angles_from_rotation_matrix(initial_R), initial_t])
    
    result_euler = least_squares(reprojection_error_euler, initial_params, args=(X, image_points, K))
    optimized_params_euler = result_euler.x
    optimized_R_euler = euler_angles_to_rotation_matrix(optimized_params_euler[:3])
    optimized_t_euler = optimized_params_euler[3:]

    return optimized_R_euler, optimized_t_euler

def NonLinearPnP_Quat(K, initial_R, initial_t, X, image_points):
    """Nonlinear refinement of rotation and translation using PnP with quaternions."""
    initial_params_quaternion = np.concatenate([Rotation.from_matrix(initial_R).as_quat(), initial_t])
    
    result_quaternion = least_squares(reprojection_error_quaternion, initial_params_quaternion, args=(X, image_points, K))
    optimized_params_quaternion = result_quaternion.x
    optimized_R_quaternion = Rotation.from_quat(optimized_params_quaternion[:4]).as_matrix()
    optimized_t_quaternion = optimized_params_quaternion[4:]

    return optimized_R_quaternion, optimized_t_quaternion