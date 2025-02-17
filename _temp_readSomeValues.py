import h5py
import numpy as np

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

import numpy as np

def print_npy_keys(file_path):
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
                    print(data.item()['wolf17'].shape)
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
    file1_path = "/mnt/c/temp/" + stimId + ".h5"
    #file1_path = "/home/bagga005/algo/comp_data/stimulus_features/raw/visual/" + stimId + ".h5"
    #file1_path = "/home/bagga005/algo/comp_data/stimulus_features/raw/visual/friends_s01e01a_features_visual.h5"
    #file1_path = "/home/bagga005/algo/comp_data/stimulus_features/raw/language/friends_s01e01a_features_language.h5"

    #read_h5_file(file1_path, stimId, 'visual')
    # file1_path = '/teamspace/studios/this_studio/algo_data/stimulus_features/pca/friends_movie10/language/features_train_new.npy'
    file1 = "/home/bagga005/algo/comp_data/stimulus_features/raw/visual/friends_s01e01a.h5"
    print_npy_keys(file1)