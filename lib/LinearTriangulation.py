import os 
import sys
sys.path.append("../")
import numpy as np
# Compute the linear triangulation using two camera poses and set of points

def LinearTriangulation(K, C1, R1, C2, R2, pts1, pts2):
    """
    K: Camera Intrinsic Matrix
    C1, R1: Camera Pose 1
    C2, R2: Camera Pose 2
    pts1: 2D points in Image 1
    pts2: 2D points in Image 2
    """

    I = np.identity(3)
    C1 = np.array(C1).reshape((3,1))
    C2 = np.array(C2).reshape((3,1))
    
    # Projection Matrix for Camera 1
    P1 = K @ R1 @ np.hstack((I, -C1))
    # Projection Matrix for Camera 2
    P2 = K @ R2 @ np.hstack((I, -C2))
    
    p1T = P1[0, :].reshape((1,4))
    p2T = P1[1, :].reshape((1,4))
    p3T = P1[2, :].reshape((1,4))
    
    p1T_ = P2[0, :].reshape((1,4))
    p2T_ = P2[1, :].reshape((1,4))
    p3T_ = P2[2, :].reshape((1,4))
    
    X = []
    for i in range(pts1.shape[0]):
        x = pts1[i, 0]
        y = pts1[i, 1]
        
        x_ = pts2[i, 0]
        y_ = pts2[i, 1]
        
        A = np.vstack((y*p3T - p2T, p1T - x*p3T, y_*p3T_ - p2T_, p1T_ - x_*p3T_))
        
        A = A.reshape((4,4))
        U, S, V = np.linalg.svd(A)
        V = V.T
        X_ = V[:,-1]
        X_ = X_/X_[-1]
        X_ = X_[:3]
        X.append(X_)
    
    X = np.array(X)
    
    return X
    
# def skew(x):
#     return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    
# def LinearTriangulation(K, C1, R1, C2, R2, pts1, pts2):
#     pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
#     pts2 = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    
#     I = np.identity(3)
    
#     C1 = np.array(C1).reshape((3,1))
#     C2 = np.array(C2).reshape((3,1))
    
#     T1  = -R1 @ C1
#     T2  = -R2 @ C2
    
#     P1 = K @ np.hstack((R1, T1))
#     P2 = K @ np.hstack((R2, T2))
    
#     X = []
    
#     for i in range(pts1.shape[0]):
#         A = skew(pts1[i]) @ P1
#         B = skew(pts2[i]) @ P2
        
#         AB = np.vstack((A, B))
        
#         U, S, V = np.linalg.svd(AB)
        
#         X_ = V[:,-1]
#         X_ = X_/X_[-1]
#         X_ = X_[:3]
#         X.append(X_)
        
#     X = np.vstack(X)
#     return X