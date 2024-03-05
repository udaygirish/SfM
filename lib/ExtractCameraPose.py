import os 
import sys
sys.path.append("../")
import numpy as np 

def ExtractCamPoseFromE(E):
    """
    Extracts camera poses from essential matrix E.

    Args:
        E (numpy.ndarray): The essential matrix.

    Returns:
        list: A list of camera poses, where each camera pose is a tuple of the form (C, R).
            C is the camera center and R is the rotation matrix.

    """
    cam_poses = []
    
    U, D, Vt = np.linalg.svd(E)
    
    W = np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]])
    
    
    C = [U[:, 2], -U[:, 2]]
    
    R = [U@W@Vt, U@W.T@Vt]
    
    for i, item in enumerate(R):
        if np.linalg.det(item) > 0:
            cam_poses.append((C[0], R[i]))
            cam_poses.append((C[1], R[i]))
            
        else:
            cam_poses.append((-C[0], -R[i]))
            cam_poses.append((-C[1], -R[i]))
    
    return cam_poses
            
        