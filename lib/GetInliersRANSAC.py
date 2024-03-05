import os 
import sys
sys.path.append("../")
import numpy as np 
import cv2 
from lib.EstimateFundamentalMatrix import *
from helpers.cv2_helpers import computeFundamentalMat_cv2
from lib.dataloader import*
from lib.EssentialMatrixFromFundamentalMatrix import *
from helpers.utils import FileHandler
from lib.helper_funcs import *
from tqdm import tqdm

def computeFeatInliers(feats1, feats2):
    """
    Computes the feature inliers using RANSAC algorithm.

    Args:
        feats1 (numpy.ndarray): Array of feature points in image 1.
        feats2 (numpy.ndarray): Array of feature points in image 2.

    Returns:
        tuple: A tuple containing the following elements:
            - inliers (list): List of indices of inlier feature points.
            - inliers_ct (int): Number of inlier feature points.
            - inlier_score (list): List of inlier scores corresponding to each inlier feature point.
    """
    eps = 0.5
    num_its = 2000
    inliers_ct = 0
    inliers = []
    inlier_score = []

    print("Running RANSAC Iterations for Feature Inliers")
    for i in tqdm(range(num_its)):
        inliers_ct_i = 0
        inliers_i = []
        inliers_score = []
        
        idx = np.random.randint(0,feats1.shape[0],8)
        rand_eight_pts_1 = feats1[idx, :]
        rand_eight_pts_2 = feats2[idx, :]
        
        F = computeFundamentalMat(rand_eight_pts_1, rand_eight_pts_2)
        
        for j, (pt1, pt2) in enumerate(zip(feats1, feats2)):
            x1, y1 = pt1[0], pt1[1]
            x2, y2 = pt2[0], pt2[1]
            
            X1 = np.array([x1, y1, 1])
            X2 = np.array([x2, y2, 1])
            
            val = X1.T @F @ X2 #X2 @ F @ X1
            
            if(abs(val) < eps):
                inliers_ct_i +=1
                inliers_i.append(j)
                inliers_score.append(abs(val))
                
        if (inliers_ct_i > inliers_ct):
            inliers_ct = inliers_ct_i
            inliers = inliers_i
            inlier_score = inliers_score 
            
    # Return the inliers based on the inlier score in descending order
    #inliers_pairs = list(zip(inliers, inlier_score))
    
    # Sort the inliers based on the inlier score
    # sorted_inlier_pairs = sorted(inliers_pairs, key = lambda x: x[1], reverse = True)
    
    # Extract the inliers and inlier score
    #inliers = [pair[0] for pair in sorted_inlier_pairs]
    #inlier_score = [pair[1] for pair in sorted_inlier_pairs]
    
    return inliers, inliers_ct, inlier_score
        

def main():
    # Test the Function
    calib_path = '../../../P3Data/calibration.txt'
    features, total_matching_feat_list = readFeatures("../../../P3Data/")
    file_handler = FileHandler("../../../P3Data/", calib_path)
    K = file_handler.loadCameraInt()
    feats1, feats2 = mapImgFeatures(1, 2, features)

    inliers, inliers_ct, inliers_score = computeFeatInliers(feats1, feats2)
    print("Inliers Length:", len(inliers))
    print("Inliers Count:", inliers_ct)
    print("Inliers Score length", len(inliers_score))
    feats1_inliers = feats1[inliers,:]
    feats2_inliers = feats2[inliers,:]
    

    print("Length of Inlier Features 1: ", feats1_inliers.shape)
    print("Length of Inlier Features 2: ", feats2_inliers.shape)
    
    # Select 8 point correspondences 
    # Check on this is it literally necessary to select 8 points 
    # Or free to choose
    rand_eight_pts_1 = feats1_inliers[:8, :]
    rand_eight_pts_2 = feats2_inliers[:8, :]
    
    F_opt = computeFundamentalMat(feats1_inliers, feats2_inliers)
    F_cv2 = computeFundamentalMat_cv2(feats1_inliers, feats2_inliers)
    
    print("Optimized Fundamental Matrix: ", F_opt)
    print("CV2 Fundamental Matrix: ", F_cv2)
    print("Feature Inlier Count: ", len(inliers))
    
    rank_F_opt = np.linalg.matrix_rank(F_opt)
    rank_F_cv2 = np.linalg.matrix_rank(F_cv2)
    print("Optimized Fundamental Matrix Rank: ",rank_F_opt)
    print("CV2 Fundamental Matrix Rank: ",rank_F_cv2)

if __name__ == "__main__":
    main()
    