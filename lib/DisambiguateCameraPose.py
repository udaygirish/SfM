# Given four camera pose configurations and their triangulated points, 
# find the correct camera pose configuration and the corresponding 3D points.
# use the condition r3*(X-C) > 0 to disambiguate the camera pose.
import os 
import sys
sys.path.append("../")
import numpy as np


def disambiguateCameraPose(C_set, R_set, X_set):
    """
    C_set: Set of Camera Poses (shape: Nx3)
    R_set: Set of Rotation Matrices (shape: Nx3x3)
    X_set: Set of Triangulated 3D Points (shape: 3xM)
    """
    
    Correct_C = None
    Correct_R = None
    Correct_X = None
    max_inliers = []
    
    # Iterate over each camera pose configuration
    for C, R, X in zip(C_set, R_set, X_set):
        
        # Check two conditions
        # 1. Z coordinate of the 3D point should be positive
        # 2. r3*(X-C) > 0
        
        # Condition 1
        condition1 = X[:,2] > 0
        condition2 = R[:,2].T @ (X.T - C.reshape(-1,1)) > 0
        
        inliers = np.logical_and(condition1, condition2)
        
        if np.sum(inliers) > np.sum(max_inliers):
            max_inliers = inliers
            Correct_C = C
            Correct_R = R
            Correct_X = X
            max_inliers = inliers
            
    return Correct_C, Correct_R, Correct_X
