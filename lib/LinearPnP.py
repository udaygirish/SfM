# Given 2D-3D correspondences and the camera intrinsic matrix, 
# estimate camera pose using Linear PnP algorithm.
import os 
import sys
sys.path.append("../")
import numpy as np
import cv2

def linearPnP(K, pts3d, pts2d):
    """
    K: Camera Intrinsic Matrix
    ps3d: 3D points in world coordinates
    pts2d: 2D points in image coordinates
    """
    if pts3d.shape[0] < 6:
        raise ValueError("At least 6 3D-2D correspondences are required")
    
    
    # Convert 2D points to homogeneous coordinates
    if pts2d.shape[1] == 2:
        pts2d = np.hstack((pts2d, np.ones((pts2d.shape[0], 1))))
    else:
        pts2d = pts2d
    
    # Convert 3D points to homogeneous coordinates
    if pts3d.shape[1] == 3:
        pts3d = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    else:
        pts3d = pts3d
    
    # Normalize 2D points
    pts2d = np.linalg.inv(K) @ pts2d.T
    pts2d = (pts2d/pts2d[2, :]).T
    
    
    A = []
    I = np.eye(3)
    for i in range(pts3d.shape[0]):
        X, Y, Z , _ = pts3d[i]
        u, v = pts2d[i][0], pts2d[i][1]
        
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
        
    A = np.array(A)
    
    U, sigma, V = np.linalg.svd(A)
    
    # P = V[-1].reshape(3, 4)
    
    # R = P[:, :3]
    # C = -np.linalg.inv(R) @ P[:, 3]
    V = V.T
    P = V[:,-1]
    P = P.reshape(3, 4)
    R = P[:, :3]
    T = P[:, 3]
    UR, SR, VR = np.linalg.svd(R)
    
    R = UR @ VR
    R_det = np.linalg.det(R)
    scale = SR[0]
    
    #T = np.linalg.inv(K) @ P[:, 3]/scale 
    # print("Scale: ", scale)
    # print(SR)
    # C = - np.linalg.inv(R).dot(P[:, 3])
    
    if R_det < 0:
        R = -R
        T = -T
        
    C = - R.T @ T
    
    return R, C

# def ProjectionMatrix(R,C,K):
#     C = np.reshape(C, (3, 1))        
#     I = np.identity(3)
#     P = np.dot(K, np.dot(R, np.hstack((I, -C))))
#     return P

# def homo(pts):
#     return np.hstack((pts, np.ones((pts.shape[0], 1))))
    
# def reprojectionErrorPnP(x3D, pts, K, R, C):
#     P = ProjectionMatrix(R,C,K)
    
#     Error = []
#     for X, pt in zip(x3D, pts):

#         p_1T, p_2T, p_3T = P# rows of P
#         p_1T, p_2T, p_3T = p_1T.reshape(1,-1), p_2T.reshape(1,-1), p_3T.reshape(1,-1)
#         X = homo(X.reshape(1,-1)).reshape(-1,1) # make X it a column of homogenous vector
#         ## reprojection error for reference camera points 
#         u, v = pt[0], pt[1]
#         u_proj = np.divide(p_1T.dot(X) , p_3T.dot(X))
#         v_proj =  np.divide(p_2T.dot(X) , p_3T.dot(X))

#         E = np.square(v - v_proj) + np.square(u - u_proj)

#         Error.append(E)

#     mean_error = np.mean(np.array(Error).squeeze())
#     return mean_error

# def homo(pts):
#     return np.hstack((pts, np.ones((pts.shape[0], 1))))

# def linearPnP(K, X_set, x_set):
#     N = X_set.shape[0]
    
#     X_4 = homo(X_set)
#     x_3 = homo(x_set)
    
#     # normalize x
#     K_inv = np.linalg.inv(K)
#     x_n = K_inv.dot(x_3.T).T
    
#     for i in range(N):
#         X = X_4[i].reshape((1, 4))
#         zeros = np.zeros((1, 4))
        
#         u, v, _ = x_n[i]
        
#         u_cross = np.array([[0, -1, v],
#                             [1,  0 , -u],
#                             [-v, u, 0]])
#         X_tilde = np.vstack((np.hstack((   X, zeros, zeros)), 
#                             np.hstack((zeros,     X, zeros)), 
#                             np.hstack((zeros, zeros,     X))))
#         a = u_cross.dot(X_tilde)
        
#         if i > 0:
#             A = np.vstack((A, a))
#         else:
#             A = a
            
#     _, _, VT = np.linalg.svd(A)
#     P = VT[-1].reshape((3, 4))
#     R = P[:, :3]
#     U_r, D, V_rT = np.linalg.svd(R) # to enforce Orthonormality
#     R = U_r.dot(V_rT)
    
#     C = P[:, 3]
#     C = - np.linalg.inv(R).dot(C)
    
#     if np.linalg.det(R) < 0:
#         R = -R
#         C = -C
        
#     return R, C