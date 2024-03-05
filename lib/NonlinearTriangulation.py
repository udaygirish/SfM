# Implement the Non linear Triangulation 
# From Initial estimates of Linear Triangulation and Camera Projection matrix and the linearlty triangulated points
# use scipy.optimize.least_squares to minimize the reprojection error
import os 
import sys
sys.path.append("../")
import numpy as np
from scipy.optimize import least_squares, differential_evolution
from helpers.cv2_helpers import *
from lib.helper_funcs import *

def compute_residual(X, pts1, pts2, P1, P2):
    reprojected_pts1 = reproject_3D_2D_P(P1, X)
    reprojected_pts2 = reproject_3D_2D_P(P2, X)
    
    

    # # Normalize the reprojected points
    # # Shape from (2,) to (3,)
    # reprojected_pts1 = np.hstack((reprojected_pts1, 1))
    # reprojected_pts2 = np.hstack((reprojected_pts2, 1))
    
    # reprojected_pts1 = (T1 @ reprojected_pts1.T).T
    # reprojected_pts2 = (T2 @ reprojected_pts2.T).T
    
    # # Remove the 1 from the last column
    # reprojected_pts1 = reprojected_pts1[:2]
    # reprojected_pts2 = reprojected_pts2[:2]
    
    # print("Pts1: ", pts1)
    # print("Reprojected Pts1: ", reprojected_pts1)
    # print("Pts2: ", pts2)
    # print("Reprojected Pts2: ", reprojected_pts2)
    
    
    reprojection_error1 = reprojection_error(reprojected_pts1, pts1)
    reprojection_error2 = reprojection_error(reprojected_pts2, pts2)
    
    reprojection_error1 = np.array([reprojection_error1])
    reprojection_error2 = np.array([reprojection_error2])
    total_reprojection_error = np.sum((reprojection_error1, reprojection_error2))
    #total_reprojection_error = reprojection_error2
    # total_reprojection_error = * total_reprojection_error
    return total_reprojection_error

def nonLinearTriangulation(K, C1, R1, C2, R2, pts1, pts2, X_init):
    """
    K: Camera Intrinsic Matrix
    C1, R1: Camera Pose 1
    C2, R2: Camera Pose 2
    pts1: 2D points in Image 1
    pts2: 2D points in Image 2
    X_init: Initial Estimate of 3D points
    """
    
    I = np.eye(3)
    C1 = C1.reshape(-1,1)
    C2 = C2.reshape(-1,1)
    
    P1 = K @ R1 @ np.hstack((I, -C1))
    P2 = K @ R2 @ np.hstack((I, -C2))
    
    print("P1: ", P1)
    print("P2: ", P2)
    X_optimized = []
    
    # Normalize the 2D points
    # pts1, T1 = normalize_points(pts1)
    # pts2, T2 = normalize_points(pts2)
    
    # X_norm , T3 = normalize_points_3d(X_init)
    
    # pts1 = pts1[:,:2]
    # pts2 = pts2[:,:2]
    # X_norm = X_norm[:,:3]

    
    for i, X_i in enumerate(X_init):
        pts1_ = pts1[i]
        pts2_ = pts2[i]
        res = least_squares(compute_residual, x0=X_i, args=(pts1_, pts2_, P1, P2))
        #res = differential_evolution(compute_residual, args=(pts1_, pts2_, P1, P2), bounds=[(0, 50), (0, 50), (0, 50)])
        X_optimized.append(res.x)
        
    X_optimized = np.array(X_optimized)
    
    return X_optimized 

