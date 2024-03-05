# Build visibility matrix
# Concept is construct I*J binary matrix where I is number of cameras and J is number of 3D points
# Vij is one if jth point is visible in ith camera, else zero
import os 
import sys
sys.path.append("../")
import numpy as np
import cv2
from helpers.cv2_helpers import *
from lib.helper_funcs import *
from scipy.sparse import lil_matrix

# def buildVisibilityMatrix(C_set, R_set, X_set, K, W, H):
#     """
#     C_set: Set of Camera Poses
#     R_set: Set of Rotation Matrices
#     X_set: Set of Triangulated 3D Points
#     K: Camera Intrinsic Matrix
#     W: Image Width
#     H: Image Height
#     """
#     # Number of cameras
#     num_cameras = len(C_set)
    
#     # Number of 3D points
#     num_points = X_set[0].shape[0]
    
#     # Initialize the visibility matrix
#     V = np.zeros((num_cameras, num_points))
    
#     # For each camera
#     for i in range(num_cameras):
#         # Reproject the 3D points to 2D
#         pts2d_proj = reproject_3D_2D(K, C_set[i], R_set[i], X_set[i])
        
#         print(pts2d_proj.shape)
#         pts2d_proj = pts2d_proj.reshape(-1, 2)
#         # Check if the points are within the image boundaries
#         mask = (pts2d_proj[:,0] >= 0) & (pts2d_proj[:,0] < W) & (pts2d_proj[:,1] >= 0) & (pts2d_proj[:,1] < H)
        
#         # Ensure the mask has the correct length (matching the number of 3D points)
#         mask = np.pad(mask, (0, num_points - len(mask)), 'constant', constant_values=(False,))
#         # Update the visibility matrix
#         V[i] = mask
    
#     return V

def create_visibility_matrix(features, image_pairs, inliers_idx):
    visibility_matrix = np.zeros((len(features), len(image_pairs)), dtype=int)

    for i, pair in enumerate(image_pairs):
        feat1, feat2, match_index = mapImgFeatures(pair[0], pair[1], features)

        common_inliers = [match_index[j] for j in inliers_idx if match_index[j] in match_index]
        
        for j in common_inliers:
            visibility_matrix[j, i] = 1

    return visibility_matrix


def sparse_matrix(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 7 + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    
    i = np.arange(camera_indices.size)
    
    
    for s in range(7):
        A[2 * i, camera_indices * 7 + s] = 1
        A[2 * i + 1, camera_indices * 7 + s] = 1
    
    for s in range(3):
        A[2 * i, n_cameras * 7 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 7 + point_indices * 3 + s] = 1
        
    return A