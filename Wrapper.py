# Create Your Own Starter Code :)

# Base Library imports
import os 
import cv2 
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt 
import sys
sys.path.append("../")

matplotlib.use('TkAgg')

# Imports from the Functions and helpers
from lib.dataloader import * 
from lib.EstimateFundamentalMatrix import *
from lib.GetInliersRANSAC import *
from helpers.utils import FileHandler
from helpers.cv2_helpers import *
from lib.helper_funcs import *
from lib.ExtractCameraPose  import * 
from lib.LinearTriangulation import *
from lib.DisambiguateCameraPose import *
from lib.NonlinearTriangulation import *
from lib.LinearPnP import *
from lib.PnPRANSC import *
from lib.NonlinearPnP import *
from lib.BuildVisibilityMatrix import *
from lib.BundleAdjustment import *

from helpers.logger import setup_logger


logger = setup_logger()

# For now let us do for two images 
def test(args):
    data_path = args.matching_features_path
    calib_path = args.calib_file_path

    file_handler = FileHandler(data_path, calib_path)
    # Read the features and get the total matching feature list
    features, total_matching_feat_list = readFeatures(data_path)
    
    # Get the camera calibration matrix - Instrinsic Matrix 
    K = file_handler.loadCameraInt()
    
    # Load Images  in Color -3 Channel 
    imgs = file_handler.readImgs()
    
    print("Number of Images: ", len(imgs))
    print("Number of Features: ", len(features))    
    no_of_images = len(imgs)
    
    # Get the features for the first and second image
    feats1, feats2, _ = mapImgFeatures(1, 2, features)
    
    print("Number of Features in Image 1: ", len(feats1))
    print("Number of Features in Image 2: ", len(feats2))
    # Plot feature matches before
    feat_match_before = draw_matches_cv2(imgs[0], imgs[1], feats1, feats2)
    
    file_handler.write_output(feat_match_before, "FeatureMatches/", "Feature_Matches_Before.png")
    
    # Compute the inliers using RANSAC - Homography
    best_inliers = compute_homography_RANSAC(feats1, feats2)
    
    print("Number of Inliers from Homography: ", len(best_inliers))
    feat_match_homography = draw_matches_cv2(imgs[0], imgs[1], feats1[best_inliers], feats2[best_inliers])
    
    file_handler.write_output(feat_match_homography, "FeatureMatches/", "Feature_Matches_Homography.png")
    feats1 = feats1[best_inliers,:]
    feats2 = feats2[best_inliers,:]
    
    # Compute the inliers using RANSAC - 8 Point Algorithm
    inliers, inliers_ct, inliers_score = computeFeatInliers(feats1, feats2)
    
    print("Number of Inliers from 8 Point Algorithm: ", len(inliers))
    # Plot feature matches after
    feat_match_after = draw_matches_cv2(imgs[0], imgs[1], feats1[inliers], feats2[inliers])
    
    file_handler.write_output(feat_match_after, "FeatureMatches/", "Feature_Matches_After.png")
    
    # Get the inliers from the 8 Point Algorithm
    # inliers = inliers[:8]
    
    # Compute the Fundamental Matrix using the inliers
    F_opt = computeFundamentalMat(feats1[inliers], feats2[inliers])
    
    # Compute the Fundamental Matrix using the inliers using OpenCV
    F_cv2 = computeFundamentalMat_cv2(feats1[inliers], feats2[inliers])
    
    logger.info(f"Optimized Fundamental Matrix: {F_opt}")
    logger.info(f"CV2 Fundamental Matrix: {F_cv2}")
    
    
    # Check the rank of the fundamental matrix
    rank_F_opt = np.linalg.matrix_rank(F_opt)
    rank_F_cv2 = np.linalg.matrix_rank(F_cv2)
    
    print("Optimized Fundamental Matrix: ", F_opt)
    print("CV2 Fundamental Matrix: ", F_cv2)
    
    print("Optimized Fundamental Matrix Rank: ",rank_F_opt)
    print("CV2 Fundamental Matrix Rank: ",rank_F_cv2)
    
    # Get epipoles
    e1, e2 = get_epipoles(F_opt)
    print("Epipoles: ", e1, e2)
    
    # Get Epipolar Lines
    lines1, lines2 = get_epipolar_lines(F_opt, feats1, feats2)
    print("Epipolar Lines 1: ", lines1)
    print("Epipolar Lines 2: ", lines2)
    
    # Draw the epipolar lines
    epipole_img1, epipole_img2 = drawlines(imgs[0].copy(), imgs[1].copy(), lines1, feats1, feats2)
    
    # Draw the epipolar lines
    epipole_img1_1, epipole_img2_1 = drawlines(imgs[1].copy(), imgs[0].copy(), lines2, feats2, feats1)
    
    # Write Images 
    file_handler.write_output(epipole_img1, "Epipoles/", "Epipole_img1.png")
    file_handler.write_output(epipole_img2, "Epipoles/", "Epipole_img2.png")
    file_handler.write_output(epipole_img1_1, "Epipoles/", "Epipole_img1_1.png")
    file_handler.write_output(epipole_img2_1, "Epipoles/", "Epipole_img2_1.png")

    
    # Get the Essential Matrix
    E = EssentialMatrixFromFundamentalMatrix(F_opt, K)
    logger.info(f"Epipolar Matrix: {E}")
    
    print("Epipolar Matrix: ", E)
    
    # Check Rank and Determinant of the Essential Matrix
    rank_E = np.linalg.matrix_rank(E)
    det_E = np.linalg.det(E)
    
    print("Rank of Essential Matrix (Ideally should be 2 because of Rank Enforcement): ", rank_E)
    print("Determinant of Essential Matrix (Ideally Should be closer to zero): ", det_E)
    
    # Check of Epipoles from F and E
    e1_F, e2_F = get_epipoles(F_opt)
    e1_E, e2_E = get_epipoles(E)
    
    e1_F1 = K @ e1_E
    e2_F1 = K @ e2_E
    
    print("Epipoles from F: ", e1_F, e2_F)
    print("Epipoles from E: ", e1_F1, e2_F1)
    
    # Check both are same or not
    print("Epipole 1 Check: ", np.allclose(e1_F, e1_F1))
    print("Epipole 2 Check: ", np.allclose(e2_F, e2_F1))
    
    # Extract Camera Pose
    cam_poses = ExtractCamPoseFromE(E)
    
    
    print("Camera Poses: ", cam_poses)
    print("Number of Camera Poses: ", len(cam_poses))
    
    # Print each camera pose
    for i in range(len(cam_poses)):
        print("Camera Pose ", i, "C: ", cam_poses[i][0])
        print("Camera Pose ", i, "R: ", cam_poses[i][1])
        
    # Plot the Camera Poses in 3D using Matplotlib
    plot_camera_pose(cam_poses, type= "center", save_path='Data/IntermediateOutputImages/Cam_Poses_Plot.png') # Accepted types- center, angle, both
    
    
    # Get the 3D Points using Linear Triangulation
    X_All_Poses = []

    # Base frame 
    C0 = np.zeros(3)
    R0 = np.eye(3)
    
    for i in range(len(cam_poses)):
        C = cam_poses[i][0]
        R = cam_poses[i][1]
        
        X = LinearTriangulation(K, C0, R0, C, R, feats1, feats2)
        X_All_Poses.append(X)
        

    # Plot the 3D Points
    plot_3D_points(X_All_Poses, save_path='Data/IntermediateOutputImages/3D_Points_Plot.png')
    
    # Print 10 random 3D points
    # print("10 Random 3D Points: ", X[np.random.choice(X.shape[0], 30, replace=False)])
        

    C_set = []
    R_set = []
    
    for i in range(len(cam_poses)):
        C_set.append(cam_poses[i][0])
        R_set.append(cam_poses[i][1])
    # # Disambiguate the camera poses
    C_Correct, R_Correct, X_Correct = disambiguateCameraPose(C_set, R_set, X_All_Poses)
    
    print(C, R, len(X_Correct))
    
    # Plot the correct 3D Points
    # Plot the 3D Points
    X_correct_plot = [X_Correct]
    plot_3D_points(X_correct_plot , save_path='Data/IntermediateOutputImages/3D_Points_Plot_Correct.png')
    
    # Reproject the 3D Points to the 2D Image Plane
    reprojected_2D_points_img1 = reproject_3D_2D(K, C_Correct, R_Correct, X_Correct, feats1) 
    reprojected_2D_points_img2 = reproject_3D_2D(K, C_Correct, R_Correct, X_Correct, feats2)
    
    
    # Draw the reprojected 2D points
    reprojected_linear_img1 = draw_points_on_image(imgs[0], reprojected_2D_points_img1)
    reprojected_linear_img2 = draw_points_on_image(imgs[1], reprojected_2D_points_img2)
    
    
    print("Reprojection Error Image 1: ", calculate_reproj_error(reprojected_2D_points_img1, feats1))
    print("Reprojection Error Image 2: ", calculate_reproj_error(reprojected_2D_points_img2, feats2))
    
    # Write the reprojected 2D points
    file_handler.write_output(reprojected_linear_img1, "Reprojected2DPoints_LinearT/", "Reprojected2DPoints_img1.png")
    file_handler.write_output(reprojected_linear_img2, "Reprojected2DPoints_LinearT/", "Reprojected2DPoints_img2.png")
    
    # Non Linear Triangulation
    X_init = X_Correct
    X_NonLinear = nonLinearTriangulation(K,C0, R0, C_Correct, R_Correct, feats1, feats2, X_init)
    
    # Reproject the 3D Points to the 2D Image Plane
    reprojected_2D_points_img1_non_l = reproject_3D_2D(K, C_Correct, R_Correct, X_NonLinear, feats1) 
    reprojected_2D_points_img2_non_l = reproject_3D_2D(K, C_Correct, R_Correct, X_NonLinear, feats2)
    
    print("Reprojection Error Image 1 Non Linear: ", calculate_reproj_error(reprojected_2D_points_img1_non_l, feats1))
    print("Reprojection Error Image 2 Non Linear: ", calculate_reproj_error(reprojected_2D_points_img2_non_l, feats2))
    
    X_NonLinear_Plot = [X_NonLinear]
    
    # ToDo: 
    # 1. Correct the Non Linear triangulation method - Done optimized to some extent
    # 2. Correct Linear PnP  
    # 3. Correct RANSAC PnP
    # 4. Correct Non Linear PnP
    # 5. Correct Visibiliy Matrix
    # 6. Correct Bundle Adjustment
    # 7. Implement Loop and DO bundle adjustment using SBA
    # 8. Implement Deep learning based feature matching and compare the results
    # 9. Implement Visualization of 3D points and camera poses using Open3D
    # 10. Implement the Streamlit App for the same - Which gives 3D visualization and 2D Matlab Plot 
    # 11. Deploy it on the cloud and share the link
    
    plot_3D_points(X_NonLinear_Plot , save_path='Data/IntermediateOutputImages/3D_Points_Plot_NonLinear.png')

    # Plot both
    
    X_total_plot = [X_Correct, X_NonLinear]
    plot_3D_points(X_total_plot , save_path='Data/IntermediateOutputImages/3D_Points_Plot_Comparison.png')
    
    # R_init, C_init = np.eye(3), np.zeros(3)
    
    # best_R1, best_C1 = PnPRANSAC(K, X_NonLinear, feats1, threshold=8.0, max_iters=2000)
    
    # success, best_R11 , best_C11, _ = cv2.solvePnPRansac(X_NonLinear, feats1, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    
    # reprojected_points_pnp_ransac_1 = project_points(X_NonLinear, best_R1, best_C1, K)
    # # convert rotation vector to rotation matrix
    # best_R11, _ = cv2.Rodrigues(best_R11)
    # reprojected_points_pnp_ransac_11 = project_points(X_NonLinear, best_R11, best_C11, K)

    
    # print("Reprojection Error Image 1 - RANSAC PNP: ", calculate_reproj_error(reprojected_points_pnp_ransac_1, feats1))
    # print("Reprojection Error Image 1 - RANSAC PNP CV2: ", calculate_reproj_error(reprojected_points_pnp_ransac_11, feats1))

    
    # Linear PnP
    
    
    
    # R1 , C1 = linearPnP(K, X_NonLinear, feats1)
    
    # R2, C2 = linearPnP(K, X_NonLinear, feats2)
    
    # # print("Linear PnP Camera Pose 1: ", R1, C1)
    # # print("Linear PnP Camera Pose 2: ", R2, C2)

    
    # reprojected_points_pnp_1 = project_points(X_NonLinear, R1, C1, K)
    # reprojected_points_pnp_2 = project_points(X_NonLinear, R2, C2, K)

    # print("Reprojection Error Image 1 - Linear PNP: ", calculate_reproj_error(reprojected_points_pnp_1, feats1))
    # print("Reprojection Error Image 2 - Linear PNP: ", calculate_reproj_error(reprojected_points_pnp_2, feats2))
    
    
    # # RANSAC PnP
    # best_R1, best_C1 = PnPRANSAC(K, X_NonLinear, feats1, threshold=8.0, max_iters=2000)
    # best_R2, best_C2 = PnPRANSAC(K, X_NonLinear, feats2, threshold=8.0, max_iters=2000)
    
    # # print("RANSAC PnP Camera Pose 1: ", best_R1, best_C1)
    # # print("RANSAC PnP Camera Pose 2: ", best_R2, best_C2)
    
    # reprojected_points_pnp_ransac_1 = project_points(X_NonLinear, best_R1, best_C1, K)
    # reprojected_points_pnp_ransac_2 = project_points(X_NonLinear, best_R2, best_C2, K)

    
    # print("Reprojection Error Image 1 - RANSAC PNP: ", calculate_reproj_error(reprojected_points_pnp_ransac_1, feats1))
    # print("Reprojection Error Image 2 - RANSAC PNP: ", calculate_reproj_error(reprojected_points_pnp_ransac_2, feats2))
    
    
    # # Non Linear PnP
    # R1_NonLinear_Rot, C1_NonLinear_Rot = NonLinearPnP_Rot(K, X_NonLinear, best_R1, best_C1, feats1)
    # R2_NonLinear_Rot, C2_NonLinear_Rot = NonLinearPnP_Rot(K, X_NonLinear, best_R2, best_C2, feats2)
    
    # # print("Non Linear PnP Camera Pose 1 - Rot: ", R1_NonLinear_Rot, C1_NonLinear_Rot)
    # # print("Non Linear PnP Camera Pose 2 - Rot: ", R2_NonLinear_Rot, C2_NonLinear_Rot)
    # # Non Linear PnP
    # R1_NonLinear_Q, C1_NonLinear_Q = NonLinearPnP_Quat(K, X_NonLinear, best_R1, best_C1, feats1)
    # R2_NonLinear_Q, C2_NonLinear_Q = NonLinearPnP_Quat(K, X_NonLinear, best_R2, best_C2, feats2)
    
    # # print("Non Linear PnP Camera Pose 1 - Quat: ", R1_NonLinear_Q, C1_NonLinear_Q)
    # # print("Non Linear PnP Camera Pose 2 - Quat: ", R2_NonLinear_Q, C2_NonLinear_Q)
    
    # reprojected_points_pnp_q_1 = project_points(X_NonLinear, R1_NonLinear_Q, C1_NonLinear_Q, K)
    # reprojected_points_pnp_q_2 = project_points(X_NonLinear, R2_NonLinear_Q, C2_NonLinear_Q, K)
    
    
    # print("Reprojection Error Image 1: ", calculate_reproj_error(reprojected_points_pnp_q_1, feats1))
    # print("Reprojection Error Image 2: ", calculate_reproj_error(reprojected_points_pnp_q_2, feats2))
    
    
    # Construct Visibility Matrix
    
    
    # Bundle Adjustment


# Lets do for all images
def main(args):
    
    data_path = args.matching_features_path
    calib_path = args.calib_file_path

    file_handler = FileHandler(data_path, calib_path)
    # Read the features and get the total matching feature list
    features, total_matching_feat_list = readFeatures(data_path)
    
    # Get the camera calibration matrix - Instrinsic Matrix 
    K = file_handler.loadCameraInt()
    
    # Load Images  in Color -3 Channel 
    imgs = file_handler.readImgs()
    
    no_of_images = len(imgs)
    
    C_set = []
    R_set = []
    X_total = []
    feat_set = []
    X_total_extend = []
    
    # Pairs of images
    image_pairs = []
    for i in range(1, no_of_images+1):
        for j in range(i+1, no_of_images+1):
            image_pairs.append((i, j))
            
    print(image_pairs)
    
    print("Current Image Pair ", image_pairs[0])
    # Initial World Coordinates from Image 1  and Image 2
    feats1, feats2, match_index = mapImgFeatures(1, 2, features)
    
    best_inliers = compute_homography_RANSAC(feats1, feats2)
    
    feats1 = feats1[best_inliers,:]
    feats2 = feats2[best_inliers,:]
    
    inliers, inliers_ct, inliers_score = computeFeatInliers(feats1, feats2)
    
    F_opt = computeFundamentalMat(feats1[inliers], feats2[inliers])
    
    E = EssentialMatrixFromFundamentalMatrix(F_opt, K)
    
    cam_poses = ExtractCamPoseFromE(E)
    
    X_All_Poses = []
    
    C0 = np.zeros(3)
    R0 = np.eye(3)
    
    temp_C_set = []
    temp_R_set = []
    
    for i in range(len(cam_poses)):
        C = cam_poses[i][0]
        R = cam_poses[i][1]
        
        temp_C_set.append(C)
        temp_R_set.append(R)
        
        X = LinearTriangulation(K, C0, R0, C, R, feats1, feats2)
        X_All_Poses.append(X)
        
    C_correct, R_correct, X_correct = disambiguateCameraPose(temp_C_set, temp_R_set, X_All_Poses)
    
    reprojected_2D_points_img1_l = reproject_3D_2D(K, C_correct, R_correct, X_correct, feats1) 
    reprojected_2D_points_img2_l = reproject_3D_2D(K, C_correct, R_correct, X_correct, feats2)
    
    reproj_error_linear1 = calculate_reproj_error(reprojected_2D_points_img1_l, feats1)
    reproj_error_linear2 = calculate_reproj_error(reprojected_2D_points_img2_l, feats2)
    
    
    print("Reprojection Error Image 1 Linear: ", reproj_error_linear1)
    print("Reprojection Error Image 2 Linear: ", reproj_error_linear2)
    
    
    X_NonLinear = nonLinearTriangulation(K,C0, R0, C_correct, R_correct, feats1, feats2, X_correct)
    
    
    reprojected_2D_points_img1_non_l = reproject_3D_2D(K, C_correct, R_correct, X_NonLinear, feats1) 
    reprojected_2D_points_img2_non_l = reproject_3D_2D(K, C_correct, R_correct, X_NonLinear, feats2)
    
    reproj_error_nonlinear1 = calculate_reproj_error(reprojected_2D_points_img1_non_l, feats1)
    reproj_error_nonlinear2 = calculate_reproj_error(reprojected_2D_points_img2_non_l, feats2)
    
    print("Reprojection Error Image 1 Non Linear: ", reproj_error_nonlinear1)
    print("Reprojection Error Image 2 Non Linear: ", reproj_error_nonlinear2)
    
    logger.info(f"Reprojection Error Image 1 Linear: {reproj_error_linear1}")
    logger.info(f"Reprojection Error Image 2 Linear: {reproj_error_linear2}")
    logger.info(f"Reprojection Error Image 1 Non Linear: {reproj_error_nonlinear1}")
    logger.info(f"Reprojection Error Image 2 Non Linear: {reproj_error_nonlinear2}")
    
    C_set.append(C_correct)
    R_set.append(R_correct)
    
    # Get common inlier index based on image1  
    inliers_idx = []
    
    # Assuming order is preserved
    for i in inliers:
        inliers_idx.append(best_inliers[i])
    
    print(len(inliers_idx))
    print(len(X_NonLinear))
    # print(match_index)
    _, _, match_index = mapImgFeatures(1, 2, features)
    
    print(len(match_index))
    filtered_match_index = [match_index[i] for i in inliers_idx]
    print(len(filtered_match_index))
    X_total.append(X_NonLinear)
    X_total_extend.extend(X_NonLinear)
    
    feat_set.append(feats1)
    feat_set.append(feats2)
    # # For each pair of images
    for img_pair in image_pairs[1:4]:
        C_ref, R_ref = C_set[0], R_set[0]
        print("Current Image Pair ", img_pair)
        
        temp_world_coord = []
        temp_feats = []
        temp_feats1 = []
        feat1_ , feat2_, match_index_ = mapImgFeatures(img_pair[0], img_pair[1], features)
        
        for i in inliers_idx:
            temp_filter_mat_index = match_index[i]
            if temp_filter_mat_index in match_index_:
                t_index = match_index_.index(temp_filter_mat_index)
                temp_feats.append(feat2_[t_index])  
                temp_feats1.append(feat1_[t_index])
                temp_world_coord.append(X_NonLinear[inliers_idx.index(i)])
        
        print("Number of Features: ", len(temp_feats))
        print("Number of 3D Points: ", len(temp_world_coord))   
        
        temp_world_coord = []
        temp_feats = []
        temp_feats1 = []
        # Compute the inliers using RANSAC - Homography
        best_inliers = compute_homography_RANSAC(feat1_, feat2_)
        best_inliers = np.arange(len(feat1_))
        
        feat11_ = feat1_[best_inliers,:]
        feat21_ = feat2_[best_inliers,:]
        
        # Compute the inliers using RANSAC - 8 Point Algorithm
        inliers, inliers_ct, inliers_score = computeFeatInliers(feat11_, feat21_)
        
        temp_inlier_index= []
        for i in inliers:
            temp_inlier_index.append(best_inliers[i])
        
        for i in inliers_idx:
            temp_filter_mat_index = match_index[i]
            if temp_filter_mat_index in match_index_:
                t_index = match_index_.index(temp_filter_mat_index)
                if t_index in temp_inlier_index:
                    temp_feats.append(feat2_[t_index])  
                    temp_feats1.append(feat1_[t_index])
                    temp_world_coord.append(X_NonLinear[inliers_idx.index(i)])
        
        print("Number of Features: ", len(temp_feats))
        print("Number of 3D Points: ", len(temp_world_coord))           
        temp_feats = np.array(temp_feats)
        temp_world_coord = np.array(temp_world_coord)
        temp_feats1 = np.array(temp_feats1)
        
        
        temp_R, temp_C = PnPRANSAC(K, temp_world_coord, temp_feats, threshold=30, max_iters=5000)
        
        
        reprojected_points_pnp_ransac_1 = reproject_3D_2D(K, temp_C, temp_R, temp_world_coord, temp_feats)
        
        print(reprojected_points_pnp_ransac_1[0])
        print(temp_feats[0])
        print("Reprojection Error Image 1 - RANSAC PNP: ", calculate_reproj_error(reprojected_points_pnp_ransac_1, temp_feats))
        
        reprojection_error_temp_linear_ransac = calculate_reproj_error(reprojected_points_pnp_ransac_1, temp_feats)
        logger.info(f"Reprojection Error Image 1 - RANSAC PNP - Img ID:{img_pair[1]}: {reprojection_error_temp_linear_ransac}")
        
        temp_R_nonlinear, temp_C_nonlinear = NonLinearPnP_Rot(K, temp_R, temp_C,temp_world_coord, temp_feats)
        
        reprojected_points_pnp_q_1 = reproject_3D_2D(K, temp_C_nonlinear, temp_R_nonlinear, temp_world_coord, temp_feats)
        print("Reprojection Error Image 1 - Non Linear PNP: ", calculate_reproj_error(reprojected_points_pnp_q_1, temp_feats))
        
        reprojection_error_temp_nonlinear_pnp = calculate_reproj_error(reprojected_points_pnp_q_1, temp_feats)
        logger.info(f"Reprojection Error Image 1 - Non Linear PNP - Img ID:{img_pair[1]}: {reprojection_error_temp_nonlinear_pnp}")
        
        # Triangulate the 3D Points
        temp_X_Lin = LinearTriangulation(K, C_ref, R_ref, temp_C_nonlinear, temp_R_nonlinear, temp_feats1, temp_feats)
        

        
        # Calculate the reprojection error
        reprojected_2D_points_temp_img_lin = reproject_3D_2D(K, temp_C_nonlinear, temp_R_nonlinear, temp_X_Lin, temp_feats)
        print("Reprojection Error Image Temp Linear: ", calculate_reproj_error(reprojected_2D_points_temp_img_lin, temp_feats))
        
        reprojection_error_linear_temp_triang = calculate_reproj_error(reprojected_2D_points_temp_img_lin, temp_feats)
        logger.info(f"Reprojection Error Image Temp Linear - Img ID: {img_pair[1]}: {reprojection_error_linear_temp_triang}")
        
        temp_X_nonlinear = nonLinearTriangulation(K,C_ref, R_ref, temp_C_nonlinear, temp_R_nonlinear, temp_feats1, temp_feats, temp_X_Lin)
        
        # Calculate the reprojection error
        reprojected_2D_points_temp_img = reproject_3D_2D(K, temp_C_nonlinear, temp_R_nonlinear, temp_X_nonlinear, temp_feats)
        print("Reprojection Error Image Temp Non Linear: ", calculate_reproj_error(reprojected_2D_points_temp_img, temp_feats))
        
        reprojection_error_non_linear_temp_triang = calculate_reproj_error(reprojected_2D_points_temp_img, temp_feats)
        logger.info(f"Reprojection Error Image Temp Non Linear - Img ID: {img_pair[1]}: {reprojection_error_non_linear_temp_triang}")
        
        C_set.append(temp_C_nonlinear)
        R_set.append(temp_R_nonlinear)
        
        print(len(temp_X_nonlinear))
        print(len(temp_feats))
        # Add Boundary conditions
        new_temp_X_nonlinear = []
        new_temp_feats = []
        for i in range(len(temp_X_nonlinear)-1):
            if temp_X_nonlinear[i][2] < -5 or temp_X_nonlinear[i][2] > 30:
                continue
            else:
                new_temp_X_nonlinear.append(temp_X_nonlinear[i])
                new_temp_feats.append(temp_feats[i])
            
        temp_X_nonlinear = np.array(new_temp_X_nonlinear)
        temp_feats = np.array(new_temp_feats)
        
        X_total_extend.extend(temp_X_nonlinear)
        
        print("After Boundary Conditions")
        print(len(temp_X_nonlinear)) 
        print(len(temp_feats))
        
        X_total.append(temp_X_nonlinear)
        feat_set.append(temp_feats)
        
        # visibility_matrix = create_visibility_matrix(features, image_pairs, inliers_idx)
        #print(visibility_matrix.shape)

        print("Im in Bundle Adjustment")
        print("C_set: ", C_set)
        print("R_set: ", R_set)
        print("Shape of C_set before: ", len(C_set[0]))
        print("Shape of R_set before: ", len(R_set[0]))   
        
        print("Length of X_total_extend before: ", len(X_total_extend))
        C_set_, R_set_, X_total_extend = bundle_adjustment_no_visibility(C_set, R_set,K, X_total, feat_set)
        new_X_total_extend = []
        
        X_total_extend = new_X_total_extend
        print("===============================================")
        print("Printing R_set_")
        print(R_set_)
        print("===============================================")
        print("Printing C_set_")
        print(C_set_)
        print("===============================================")
        #Remove the last element from the list
        # C_set = C_set[:-1]
        # R_set = R_set[:-1]
        # #Append the new values
        # C_set.append(C_set_[0])
        # R_set.append(R_set_[0])
        print("Length of X_total_extend after: ", len(X_total_extend))
        print("Im out of Bundle Adjustment")
        plot_2D_points_with_Camera_poses_bundle(X_total_extend, C_set, R_set, ['r', 'g', 'b', 'y', 'c', 'm'], save_path='Data/IntermediateOutputImages/TOTAL_2D_Points_Plot_{0}_{1}.png'.format(img_pair[0], img_pair[1]),legend=False)
    
    # Plot the 3D Points
    print(len(X_total))
    print([len(X) for X in X_total])
    plot_3D_points(X_total , save_path='Data/IntermediateOutputImages/TOTAL_3D_Points_Plot_Comparison.png')
    plot_3D_points_with_Camera_poses(X_total, C_set, R_set, save_path='Data/IntermediateOutputImages/TOTAL_3D_Points_Plot_Comparison_With_Camera_Poses.png')
    #plot_2D_points_with_Camera_poses(X_total, C_set, R_set, ['r', 'g', 'b', 'y', 'c', 'm'], save_path='Data/IntermediateOutputImages/TOTAL_2D_Points_Plot_Comparison_With_Camera_Poses.png')
    
    # Make X_total as a single list
    X_total_ = []
    for X in X_total:
        X_total_.extend(X)
    X_total = X_total_
    
    X_total_extend_ = []
    for i in X_total_extend:
        print(i)
        print(i[0])
        X_total_extend_.extend(i[0])
    X_total_extend = X_total_extend_
        
    plot_2D_points_with_Camera_poses([X_total, X_total_extend], C_set, R_set, ['r', 'g', 'b', 'y', 'c', 'm'], save_path='Data/IntermediateOutputImages/TOTAL_2D_Points_Plot_Comparison_With_Camera_Poses.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFM")
    
    parser.add_argument('--matching_features_path', type=str, default='../../P3Data/',help='Path to the directory containing the matching feature files', required=False)
    parser.add_argument('--calib_file_path', type=str, default='../../P3Data/calibration.txt', help='Path to the camera calibration file', required=False)
    
    args = parser.parse_args()
    #test(args)
    main(args)