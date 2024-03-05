import numpy as np  
import cv2 
import sys 
sys.path.append("../")
from helpers.cv2_helpers import computeFundamentalMat_cv2
from helpers.logger import setup_logger
from helpers.utils import FileHandler


def mapImgFeatures(img_id_i, img_id_j, features):
    img_data = features[img_id_i-1] # extract data for the first image
    X_Y = []
    x_y = []
    match_index= []
    
    count = 0
    for row_dict in img_data:
        matching_features = row_dict['matched_feat']
        key = img_id_j
        if key in matching_features:
            x2, y2 = matching_features[key]
            X_Y.append([x2, y2])      
            
            x1, y1 = row_dict['x_y'][0], row_dict['x_y'][1]
            x_y.append([x1, y1])
            match_index.append(count)
        count += 1

    X_Y = np.array(X_Y)
    x_y = np.array(x_y)
    
    return x_y, X_Y, match_index

def normalize_points(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts)
    TK = np.array([[1/std, 0, -mean[0]/std], [0, 1/std, -mean[1]/std], [0, 0, 1]])
    normalized_points = np.hstack((pts, np.ones((pts.shape[0], 1))))
    normalized_points = (TK @ normalized_points.T).T
    return normalized_points, TK

def normalize_points_3d(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts)
    TK = np.array([[1/std, 0, 0, -mean[0]/std], [0, 1/std, 0, -mean[1]/std], [0, 0, 1/std, -mean[2]/std], [0, 0, 0, 1]])
    normalized_points = np.hstack((pts, np.ones((pts.shape[0], 1))))
    normalized_points = (TK @ normalized_points.T).T
    return normalized_points, TK