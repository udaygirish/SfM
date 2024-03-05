# N>6 3D-2D correspondences, camera pose 
# Implement Linear PnP and optimize error with RANSAC
import os 
import sys
from tqdm import tqdm
sys.path.append("../")
import numpy as np
import cv2
from lib.LinearPnP import linearPnP

# PnP using RANSAC
def PnPRANSAC(K, pts3d, pts2d, threshold=8.0, max_iters=1000):
    """
    K: Camera Intrinsic Matrix
    pts3d: 3D points in world coordinates
    pts2d: 2D points in image coordinates
    threshold: RANSAC threshold
    max_iters: Maximum number of iterations
    """
    # Convert 2D points to homogeneous coordinates
    pts2d = np.hstack((pts2d, np.ones((pts2d.shape[0], 1))))
    
    # Convert 3D points to homogeneous coordinates
    pts3d = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    
    # Normalize 2D points
    #pts2d = np.linalg.inv(K) @ pts2d.T
    #pts2d = pts2d.T
    
    # Number of correspondences
    N = pts3d.shape[0]
    
    # Initialize the best model
    best_model = None
    best_inliers = 0
    
    print("Running PnP RANSAC")
    
    for i in tqdm(range(max_iters)):
        # Randomly select 6 correspondences
        idx = np.random.choice(N, 6, replace=False)
        pts3d_sample = pts3d[idx]
        pts2d_sample = pts2d[idx]
        
        # Compute the camera pose using linear PnP
        R, C = linearPnP(K, pts3d_sample, pts2d_sample)
        
        # Compute the projection matrix
        P = K @ np.hstack((R, C.reshape(-1, 1)))
        
        # Project the 3D points to 2D
        pts2d_proj = (P @ pts3d.T).T
        pts2d_proj = pts2d_proj/pts2d_proj[:,-1].reshape(-1,1)
        # Compute the reprojection error
        #error = np.linalg.norm(pts2d_proj[:,:-1] - pts2d[:,:-1], axis=1)
        # Separate the error into x and y components and add after squaring
        error = np.sum((pts2d_proj[:,:-1] - pts2d[:,:-1])**2, axis=1)
        
        # Count the number of inliers
        inliers = np.sum(error < threshold)
        
        # Update the best model
        if inliers > best_inliers:
            best_inliers = inliers
            best_model = (R, C)
    
    return best_model