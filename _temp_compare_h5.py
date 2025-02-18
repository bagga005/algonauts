import h5py
import numpy as np
from utils import get_shortstim_name
from glob import glob
def compare_h5_datasets(file1_path, file2_path, group_name1, group_name2, dataset_name):
    """
    Compare datasets from two HDF5 files and return correlation coefficient.
    
    Parameters
    ----------
    file1_path : str
        Path to first HDF5 file
    file2_path : str
        Path to second HDF5 file
    group_name1 : str
        Name of the group containing the dataset in first file
    group_name2 : str
        Name of the group containing the dataset in second file
    dataset_name : str
        Name of the dataset to compare
    
    Returns
    -------
    tuple
        (is_equal, r_score) where:
        - is_equal is bool: True if datasets are identical
        - r_score is float: Pearson correlation coefficient between datasets
    """
    try:
        with h5py.File(file1_path, 'r') as f1, h5py.File(file2_path, 'r') as f2:
            # Load datasets
            data1 = f1[group_name1][dataset_name][:]
            data2 = f2[group_name2][dataset_name][:]
            
            print(data1.shape, data2.shape)
            # Check if shapes match
            if data1.shape != data2.shape:
                print(f"Shapes don't match: {data1.shape} vs {data2.shape}")
                return False, 0.0
            
            # Calculate R-score (correlation coefficient)
            # Flatten arrays and remove NaN values
            mask = ~(np.isnan(data1) | np.isnan(data2))
            flat1 = data1[mask].flatten()
            flat2 = data2[mask].flatten()
            r_score = np.corrcoef(flat1, flat2)[0, 1]
            print(f"R-score (correlation coefficient): {r_score:.6f}")

            # Rest of the comparison logic
            data1_rounded = np.round(data1, decimals=1)
            data2_rounded = np.round(data2, decimals=1)
            is_equal = np.array_equal(
                data1_rounded[~np.isnan(data1_rounded)], 
                data2_rounded[~np.isnan(data2_rounded)]
            )
            
            # Compare data with tolerances, ignoring NaNs
            mask = ~(np.isnan(data1) | np.isnan(data2))
            is_equal2 = np.allclose(
                data1[mask], 
                data2[mask], 
                rtol=1e-2, 
                atol=1e-5
            )
            
            print(data1[400][:5])
            print(data2[400][:5])
            if is_equal:
                print("Datasets are identical!")
            else:
                # Find where differences occur
                diff_mask = ~np.isclose(data1, data2, rtol=1e-5, atol=1e-1)
                diff_indices = np.where(diff_mask)
                
                if len(diff_indices[0]) > 0:
                    print(f"Datasets differ in {len(diff_indices[0])} positions")
                    differences = []
                    for idx in zip(*diff_indices):
                        diff_magnitude = abs(float(data1[idx]) - float(data2[idx]))
                        differences.append((idx, data1[idx], data2[idx], diff_magnitude))
                    
                    differences.sort(key=lambda x: x[3], reverse=True)
                    for idx, val1, val2, magnitude in differences[:50]:
                        print(f"Position {idx}: {val1} vs {val2} (diff: {magnitude:.6f})")
                else:
                    print("Datasets are identical in range!")
                        
            return is_equal, r_score
            
    except KeyError as e:
        print(f"Error: Group or dataset not found: {e}")
        return False, 0.0
    
    


def compare_npy_shapes(file1_path, file2_path):
    """
    Compare shapes of numpy arrays stored in dictionaries within two .npy files
    
    Args:
        file1_path (str): Path to the first .npy file
        file2_path (str): Path to the second .npy file
    """
    
    # Load both files
    data1 = np.load(file1_path, allow_pickle=True)
    data2 = np.load(file2_path, allow_pickle=True)
        
    # Convert to dictionaries if stored as numpy arrays
    if isinstance(data1, np.ndarray):
        data1 = data1.item()
    if isinstance(data2, np.ndarray):
        data2 = data2.item()
        
    # Verify both are dictionaries
    if not isinstance(data1, dict) or not isinstance(data2, dict):
        raise ValueError("Both files must contain dictionary data")
        
    print(f"\nComparing shapes between {file1_path} and {file2_path}")
    print(f"File 1 has {len(data1)} keys")
    print(f"File 2 has {len(data2)} keys")
    
    # Check each key in first file
    for key in data1.keys():
        short_key = get_shortstim_name(key)
        if short_key not in data2:
            print(f"Key '{key}' exists in file 1 but not in file 2")
            continue
            
        shape1 = data1[key].shape
        shape2 = data2[short_key].shape

        print(f"Shape of {key} in file 1: {shape1}, in file 2: {shape2}")
        
        if shape1 != shape2:
            print(f"Shape mismatch for key '{key}':")
            print(f"  File 1 shape: {shape1}")
            print(f"  File 2 shape: {shape2}")
            
def has_nan(file1_path, group_name, dataset_name):
    try:
        with h5py.File(file1_path, 'r') as f1:
            data1 = f1[group_name][dataset_name][:]
            is_nan = np.isnan(data1).any()
            print('key: ', group_name, 'data.shape: ', data1.shape, 'is nan: ', is_nan)
            return is_nan
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def has_nan_for_folder(folder_path, dataset_name):
    files = glob(f"{folder_path}/*.h5")
    files.sort()
    print(len(files), files[:3], files[-3:])
    stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
    for stim_id, stim_path in stimuli.items():
        if has_nan(stim_path, stim_id, dataset_name):
            print(f"Dataset {dataset_name} in {stim_id} has NaNs")

def compare_npy_values(file1_path, file2_path):
    """
    Compare values of numpy arrays stored in dictionaries within two .npy files
    and compute correlation coefficient (R-score) for each key.
    
    Args:
        file1_path (str): Path to the first .npy file
        file2_path (str): Path to the second .npy file
    """
    try:
        # Load both files
        data1 = np.load(file1_path, allow_pickle=True)
        data2 = np.load(file2_path, allow_pickle=True)
        
        # Convert to dictionaries if stored as numpy arrays
        if isinstance(data1, np.ndarray):
            data1 = data1.item()
        if isinstance(data2, np.ndarray):
            data2 = data2.item()
            
        # Verify both are dictionaries
        if not isinstance(data1, dict) or not isinstance(data2, dict):
            raise ValueError("Both files must contain dictionary data")
            
        print(f"\nComparing values between {file1_path} and {file2_path}")
        print(f"File 1 has {len(data1)} keys")
        print(f"File 2 has {len(data2)} keys")
        
        # Check each key in first file
        for key in data1.keys():
            if key not in data2:
                print(f"Key '{key}' exists in file 1 but not in file 2")
                continue
                
            array1 = data1[key]
            array2 = data2[key]
            
            if array1.shape != array2.shape:
                print(f"Shape mismatch for key '{key}': {array1.shape} vs {array2.shape}")
                continue
            
            # Calculate R-score (correlation coefficient)
            # Flatten arrays and remove NaN values
            mask = ~(np.isnan(array1) | np.isnan(array2))
            flat1 = array1[mask].flatten()
            flat2 = array2[mask].flatten()
            
            if len(flat1) == 0 or len(flat2) == 0:
                print(f"No valid data for comparison in key '{key}'")
                continue
                
            r_score = np.corrcoef(flat1, flat2)[0, 1]
            print(f"Key '{key}' - R-score: {r_score:.6f}")
            
            # Optional: check if arrays are exactly equal
            is_equal = np.array_equal(
                array1[~np.isnan(array1)],
                array2[~np.isnan(array2)]
            )
            if is_equal:
                print(f"  Arrays for key '{key}' are identical!")
                
    except Exception as e:
        print(f"Error comparing files: {str(e)}")


if __name__ == "__main__":
        # Example usage
    # file1 = "/home/bagga005/algo/comp_data/stimulus_features/raw/language/friends_s05e03a.h5"
    # file2 = "/mnt/c/temp/friends_s05e03a.h5"
    # group = "friends_s05e03a"
    # dataset = 'language_last_hidden_state'#"language_pooler_output"
    file1 = "/home/bagga005/algo/comp_data/stimulus_features/raw/visual/friends_s01e01a.h5"
    file2 = "/home/bagga005/algo/comp_data/stimulus_features/raw/visual/friends_s01e01a_features_visual.h5"
    folder_path = "/home/bagga005/algo/comp_data/stimulus_features/raw/visual"
    print(has_nan_for_folder(folder_path, 'visual'))
    group1 = "friends_s01e01a"
    group2 = "s01e01a"
    dataset = 'visual'#"language_pooler_output"

    # are_equal, r_score = compare_h5_datasets(file1, file2, group1, group2, dataset)
    # print('r_score', r_score)
    # print(f"Are datasets equal? {are_equal}, R-score: {r_score:.6f}")
    file1 = "/home/bagga005/algo/comp_data/stimulus_features/pca/friends_movie10/visual/features_train_new.npy"
    file2 = "/home/bagga005/algo/comp_data/stimulus_features/pca/friends_movie10/visual/features_train_orig.npy"
    #compare_npy_values(file1, file2)
    # file1_path = "/mnt/c/temp/friends_s01e01a.npy"
    # file2_path = "/home/bagga005/algo/comp_data/stimulus_features/raw/visual/friends_s01e01a.npy"
    #compare_npy_shapes(file1, file2)