import os 
import sys
sys.path.append("../")
import numpy as np

def EssentialMatrixFromFundamentalMatrix(F, K):
    """
    Computes the essential matrix from the fundamental matrix and camera intrinsic matrix.

    Parameters:
    F (numpy.ndarray): The fundamental matrix.
    K (numpy.ndarray): The camera intrinsic matrix.

    Returns:
    numpy.ndarray: The computed essential matrix.
    """
    
    E = K.T @ F @ K
    
    # Reconstructing E
    U, _, V = np.linalg.svd(E)
    
    E_opt = U @ np.diag([1, 1, 0]) @ V 
    
    return E_opt

