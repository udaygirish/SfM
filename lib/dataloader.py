import os 
import sys
sys.path.append("../")
import glob
import csv
import numpy as np

'''
Structure of the Output Dictionary - Can be saved as JSON
matching_dict = 
    [[{
        'no_of_feats' : 2,
        'color' : tuple(3),
        'x_y' : tuple(2),
        'matched_feat' :[tuple(3) - def (id, x_2, y_2), tuple(3) - def(id, x_3,y_3)]
    },.....], 
    [{
        'no_of_feat' : 2,
        'color' : tuple(3),
        'x_y' : tuple(2),
        'matched_feat' :[tuple(3) - def (id, x_2, y_2), tuple(3) - def(id, x_3,y_3)]
    }]
    ,[{
        'no_of_feat' : 2,
        'color' : tuple(3),
        'x_y' : tuple(2),
        'matched_feat' :[tuple(3) - def (id, x_2, y_2), tuple(3) - def(id, x_3,y_3)]
    }]
    ,[{
        'no_of_feat' : 2,
        'color' : tuple(3),
        'x_y' : tuple(2),
        'matched_feat' :[tuple(3) - def (id, x_2, y_2), tuple(3) - def(id, x_3,y_3)]
    }]
    ,[{
        'no_of_feat' : 2,
        'color' : tuple(3),
        'x_y' : tuple(2),
        'matched_feat' :[tuple(3) - def (id, x_2, y_2), tuple(3) - def(id, x_3,y_3)]
    }]
    ]

first order - list - number of files
second order - list-  all features
    
matching_features = [3930, 3230, - , - ,-]
'''

def readFeatures(path):
    """
    Reads matching features from files in the specified path.

    Args:
        path (str): The path to the directory containing the matching feature files.

    Returns:
        list: A list of numpy arrays, where each array represents the matching features for a file.
        Each array contains dictionaries representing the matching features for each row in the file.
        Each dictionary contains the following keys:
        - 'no_of_feats': The number of matching features.
        - 'color': The RGB color value.
        - 'curr_img_id': The current image ID.
        - 'x_y': The x and y coordinates of the first matching feature.
        - 'matched_feat': A dictionary of matched features, where the key is the feature index
        and the value is a tuple of x and y coordinates.
    """
    img_nums = []
    tot_matching_features = []
    file_paths = glob.glob(path+"matching*.txt")
    file_paths.sort()
    
    for path in file_paths: 
        img_num = path.rsplit(".", 1)[0][-1]
        img_nums.append(img_num)
        
    matching_features_list = []
    for file_num, path in enumerate(file_paths):
        with open(path, "r") as file:
            matching_file_list = []
            for row_idx, row in enumerate(file):
                elements = row.split()
                
                row_dict = {}
                if row_idx == 0:
                    total_feats = elements[1]
                    tot_matching_features.append(total_feats)
                else:
                    num_matches = int(elements[0])
                    row_dict['no_of_feats'] = num_matches
                    rgb_val = (int(elements[1]), int(elements[2]), int(elements[3]))
                    row_dict['color'] = rgb_val
                    row_dict['curr_img_id'] = img_nums[file_num]
                    features = []
                    features = {}
                    for idx in range(num_matches):
                        
                        if idx == 0:
                            row_dict['x_y'] = (float(elements[4]), float(elements[5]))
                        else:
                            i = (idx+1)*3
                            features[int(elements[i])] = (float(elements[i+1]), float(elements[i+2]))
                            
                    row_dict['matched_feat'] = features      
                    matching_file_list.append(row_dict)

            matching_file_list = np.array(matching_file_list)
            matching_features_list.append(matching_file_list)
    return matching_features_list , tot_matching_features


def main():
    # Test read features 
    features, total_matching_feat_list = readFeatures("../../../P3Data/")
    print("Total Number of Features: ", len(features))
    print("Matching Features for Image 2: ", features[1])

if __name__ == "__main__":
    main()




