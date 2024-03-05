import os 
import sys
sys.path.append("../")
import numpy as np 
import cv2 
from lib.dataloader import *
from helpers.cv2_helpers import*
from lib.helper_funcs import mapImgFeatures, normalize_points

def computeFundamentalMat(feat1, feat2):
    '''
    img_id_i: Image Number of First Image in Consideration -- {1, 2, 3, 4}
    img_id_j: Image Number of Second Image in Consideration -- {2, 3, 4, 5}
    Image Pairs to get Matching Features: 
    Format: [img_id_i, img_id_j]
    [1,2], [1,3], [1,4], [1,5]
    [2,3], [2,4], [2,5]
    [3,4], [3,5]
    [4,5]
    
    features: 
    [[matching_features_img1], [matching_features_img2], [matching_features_img3], [matching_features_img4]]
    Extract features for each img
    '''
    
    feat1_norm, T1 = normalize_points(feat1)
    feat2_norm, T2 = normalize_points(feat2)
    
    x_i, y_i = feat1_norm[:, 0], feat1_norm[:, 1]
    
    x_j, y_j = feat2_norm[:, 0], feat2_norm[:, 1]
    
    ones = np.ones(x_i.shape[0])
    A =[x_i*x_j, y_i*x_j, x_j, x_i*y_j, y_i*y_j, y_j, x_i, y_i, ones]
    A = np.vstack(A).T
    U, sigma, V = np.linalg.svd(A)
    f = V[np.argmin(sigma),:]
    #f = V[:, -1]
    #f = V[-1, :]
    f = f.reshape((3,3))
    

    
    Uf, sigmaf, Vf = np.linalg.svd(f)
    sigmaf[2] = 0 # rank 2 constraint
    F = Uf @ np.diag(sigmaf) @ Vf  
    #F = f
    
    F = T2.T @ F @ T1
    
    # Homogenize the Fundamental Matrix
    F = F/F[-1,-1]
    
    return F    

def get_epipolar_lines(F, feat1, feat2):
    '''
    Compute Epipolar Lines
    '''
    # Convert to Homogeneous Coordinates
    feat1 = np.hstack((feat1, np.ones((feat1.shape[0], 1))))
    feat2 = np.hstack((feat2, np.ones((feat2.shape[0], 1))))
    lines1 = feat1 @ F
    lines2 = feat2 @ F.T
    return lines1, lines2

def get_epipoles(F):
    '''
    Compute Epipoles
    '''
    U, sigma, V = np.linalg.svd(F)
    e1 = V[-1, :]
    e1 = e1/e1[-1]
    
    U, sigma, V = np.linalg.svd(F.T)
    e2 = V[-1, :]
    e2 = e2/e2[-1]
    
    return e1, e2

def main():
    
    # This func is to test the above functions
    features, total_matching_feat_list = readFeatures("../../../P3Data/")
    print("Length of features: ", len(features))
    feats1, feats2 = mapImgFeatures(1, 2, features)
    F = computeFundamentalMat(feats1, feats2)
    F_CV2 = computeFundamentalMat_cv2(feats1, feats2)

    print("Fundamental Matrix: ", F)
    print("Fundamental matrix using cv2: ", F_CV2)
    
    epipoles = get_epipoles(F)
    
    print("Epipoles: ", epipoles)
    
    lines1, lines2 = get_epipolar_lines(F, feats1, feats2)
    print("Epipolar Lines 1: ", lines1)
    print("Epipolar Lines 2: ", lines2)
    
    print("Lines 1 Shape: ", lines1.shape)
    print("Lines 2 Shape: ", lines2.shape)
    

if __name__ == "__main__":
    main()
