# Given initialize camera poses and 3d points, refine by minimizing projection error
# Refine using scipy.optimize.least_squares using large scale bundle adjustment

import os 
import sys
sys.path.append("../")
import numpy as np
import cv2
from scipy.optimize import least_squares, minimize
from helpers.cv2_helpers import *
from lib.helper_funcs import *
from scipy.spatial.transform import Rotation
from lib.NonlinearPnP import *

def bundle_adjustment_no_visibility(C_set, R_set, K, X_total, feat_set):
    num_images = len(C_set)
    num_points = [len(X_total[i]) for i in range(num_images-1)]
    I = np.identity(3)
    print("Rset Inside BA", R_set)
    print("Cset Inside BA", C_set)
    def residual(params, num_images, num_points, feat_set):
        residuals = []

        for i in range(num_images-1):
            C_i = params[i * 6: i * 6 + 3]
            R_i = params[i * 6 + 3: i * 6 + 6]
            R_i = euler_angles_to_rotation_matrix(R_i)
            C_i = np.array(C_i).reshape(3, 1)   
            P = K @ R_i @ np.hstack((I, -C_i))

            for j in range(num_points[i]-1):
                X_j = params[num_images * 6 + j * 3: num_images * 6 + (j + 1) * 3]
                # print(X_j)
                # print(X_j.shape)
                try:
                    if X_j.shape[0] == 0:
                        continue
                    proj = reproject_3D_2D_P(P, X_j)
                    residuals.extend((proj - feat_set[i][j])**2)
                except:
                    continue

        return np.array(residuals)

    initial_params = []

    # print("SHAPES INSIDE BUNDLE ADJUSTMENT")
    # print(len(X_total))
    # print(len(X_total[0]))
    # print(num_images)
    # print(num_points)
    for i in range(num_images-1):
        initial_params.extend(C_set[i])
        temp_set = euler_angles_from_rotation_matrix(R_set[i])
        initial_params.extend(temp_set)
        #num_points = len(X_total[i])
        for j in range(num_points[i]-1):
            # print(i,j)
            # print(len(X_total[i]))
            initial_params.extend(X_total[i][j])
            #initial_params.extend(X_total[i])

    result = least_squares(residual, initial_params, args=(num_images, num_points, feat_set), verbose=2, max_nfev=10)

    updated_params = result.x
    print(len(updated_params))
    
    updated_C_set = [updated_params[i * 6: i * 6 + 3] for i in range(num_images-1)]
    updated_R_set = [euler_angles_to_rotation_matrix(updated_params[i * 6 + 3: i * 6 + 6]).reshape(3, 3) for i in range(num_images-1)]
    updated_X_total = [updated_params[num_images * 6 + i * 3: num_images * 6 + (i + 1) * 3] for i in range(sum(num_points)-1)]

    return updated_C_set, updated_R_set, updated_X_total
