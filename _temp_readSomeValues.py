import h5py
import numpy as np
import utils
import os
def read_h5_file(file_path, stimId, group_name):
    
    with h5py.File(file_path, 'r') as f1:
        print("Root level keys:", list(f1.keys()))
        data = f1[stimId][group_name][:]
        print(data.shape)
        #print(data[400][:5])
        # Count NaN and non-NaN values
        nan_count = np.count_nonzero(np.isnan(data))
        total_elements = data.size
        non_nan_count = total_elements - nan_count
        
        print(f"Total elements: {total_elements}")
        print(f"NaN values: {nan_count}")
        print(f"Non-NaN values: {non_nan_count}")
        print(f"Percentage of NaN values: {(nan_count/total_elements)*100:.2f}%")
        
        # Optional: print some basic statistics of non-NaN values
        print("\nStatistics of non-NaN values:")
        print(f"Min: {np.nanmin(data)}")
        print(f"Max: {np.nanmax(data)}")
        print(f"Mean: {np.nanmean(data)}")


def read_npy_keys(file_path):
    """
    Print all keys in a NumPy .npy file
    
    Args:
        file_path (str): Path to the .npy file
    """
    try:
        # Load the numpy file
        data = np.load(file_path, allow_pickle=True)
        print("\nLoaded data type:", type(data))
        
        # If data is a dictionary
        if isinstance(data, dict):
            print("\nKeys in the file:")
            for key in data.keys():
                print(f"- {key}")
            

        # If data is a numpy array containing a dictionary
        elif isinstance(data, np.ndarray):
            if data.dtype == np.dtype('O'):
                if isinstance(data.item(), dict):
                    print("\nKeys in the file:")
                    for key in data.item().keys():
                        print(f"- {key}")
                    print(len(data.item().keys()))
                else:
                    print("\nArray contains object type but not a dictionary")
                    print("Content:", data)
            else:
                print("\nThis is a regular numpy array:")
                print("Shape:", data.shape)
                print("Data type:", data.dtype)
                print("First few elements:", data.flatten()[:5])
                
    except Exception as e:
        print(f"Error loading file: {str(e)}")


if __name__ == "__main__":
    #file_path = "/home/bagga005/algo/comp_data/stimulus_features/pca/friends_movie10/language/features_test.npy"
    #print_npy_keys(file_path)
    stimId = "friends_s01e01a"
    stimId = "s01e01a"
    #file1_path = "/mnt/c/temp/" + stimId + ".h5"
    #file1_path = "/home/bagga005/algo/comp_data/stimulus_features/raw/visual/" + stimId + ".h5"
    file1_path = "/teamspace/studios/present-tomato-9u8y/algo_data/stimulus_features/raw/language/friends_s01e01a_features_language.h5"
    #file1_path = "/home/bagga005/algo/comp_data/stimulus_features/raw/language/friends_s01e01a_features_language.h5"

    #read_h5_file(file1_path, stimId, 'language_pooler_output')
    #read_h5_file(file1_path, stimId, 'language_last_hidden_state')
    # file1_path = '/teamspace/studios/this_studio/algo_data/stimulus_features/pca/friends_movie10/language/features_train_new.npy'
    #file1 = "/home/bagga005/algo/comp_data/stimulus_features/pca/friends_movie10/visual/features_test.npy"
    #file1 = "/home/bagga005/algo/comp_data/algonauts_2025.competitors/fmri/sub-03/target_sample_number/sub-03_friends-s7_fmri_samples.npy"
    #file1 = "/home/bagga005/algo/comp_data/algonauts_2025.competitors/fmri/sub-03/func/sub-03_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5"
    #file_name = "sub-03_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5"
    # file_name = "sub-01_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5"
    # file_path = os.path.join(utils.get_data_root_dir(), "algonauts_2025.competitors","fmri","sub-01","func",file_name)
    # read_subject_fmri_session_h5_write_summary(file_path, '01', 0)
    file = '/teamspace/studios/this_studio/algo_data/stimulus_features/pca/friends_movie10/visual/features_train.npy'
    read_npy_keys(file_path=file)