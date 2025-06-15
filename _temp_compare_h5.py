import h5py
import numpy as np
import gzip
import pickle
from utils import get_shortstim_name
from glob import glob
import utils
import os
import torch

def compare_h5_datasets(file1_path, file2_path, group_name1, group_name2, dataset_name, rtol=1e-5, atol=1e-8, max_diff_to_show=10):
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
            print('got 1')
            # Load datasets
            data2 = f2[group_name2][dataset_name][:]
            print('got 2')
            data1 = f1[group_name1][dataset_name][:]
            
            print(data1.shape, data2.shape)
            # Check if shapes match
            if data1.shape != data2.shape:
                print(f"Shapes don't match: {data1.shape} vs {data2.shape}")
                return False, 0.0
            
            # Create masks for non-NaN values in both datasets
            mask1 = ~np.isnan(data1)
            mask2 = ~np.isnan(data2)
            
            # Combined mask where both datasets have non-NaN values
            valid_mask = mask1 & mask2
            
            # Count NaN values in each dataset
            nan_count1 = np.sum(~mask1)
            nan_count2 = np.sum(~mask2)
            
            print(f"NaN count in {file1_path}: {nan_count1}")
            print(f"NaN count in {file2_path}: {nan_count2}")
            
            # Check if NaNs are in the same positions
            nan_positions_match = np.array_equal(~mask1, ~mask2)
            if not nan_positions_match:
                mismatched_nan_count = np.sum(~mask1 != ~mask2)
                print(f"NaN positions don't match in {mismatched_nan_count} elements")
            
            # Compare only non-NaN values
            if np.sum(valid_mask) > 0:
                is_close = np.allclose(
                    data1[valid_mask], 
                    data2[valid_mask], 
                    rtol=rtol, 
                    atol=atol
                )
                
                # Calculate differences for non-NaN values
                diff = np.abs(data1[valid_mask] - data2[valid_mask])
                max_diff = np.max(diff) if diff.size > 0 else 0
                mean_diff = np.mean(diff) if diff.size > 0 else 0
                
                print(f"Max difference (excluding NaNs): {max_diff}")
                print(f"Mean difference (excluding NaNs): {mean_diff}")
                
                # Show positions with largest differences
                if diff.size > 0:
                    # Get indices of largest differences
                    largest_diff_indices = np.argsort(diff)[-min(max_diff_to_show, diff.size):]
                    
                    # Convert flat indices back to original array coordinates
                    flat_indices = np.where(valid_mask.flatten())[0][largest_diff_indices]
                    orig_indices = np.unravel_index(flat_indices, data1.shape)
                    
                    print("\nLargest differences:")
                    for i in range(len(largest_diff_indices)):
                        idx = tuple(coord[i] for coord in orig_indices)
                        print(f"Position {idx}: {data1[idx]} vs {data2[idx]} (diff: {data1[idx] - data2[idx]})")
                
                return is_close
            else:
                print("No valid (non-NaN) values to compare")
                return False
            
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
            #print('key: ', group_name, 'data.shape: ', data1.shape, 'is nan: ', is_nan)
            return is_nan
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def has_nan_for_folder(folder_path, dataset_name):
    files = glob(f"{folder_path}/*.h5")
    files.sort()
    print(len(files), files[:3], files[-3:])
    stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
    return_val = False
    for stim_id, stim_path in stimuli.items():
        if has_nan(stim_path, stim_id, dataset_name):
            print(f"Dataset {dataset_name} in {stim_id} has NaNs")
            return_val = True
    return return_val

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
            #print(f"Key '{key}' - R-score: {r_score:.6f}")
            
            # Optional: check if arrays are exactly equal
            is_equal = np.array_equal(
                array1[~np.isnan(array1)],
                array2[~np.isnan(array2)]
            )
            if not is_equal:
                print(f"  Arrays for key '{key}' are not identical!")
                
    except Exception as e:
        print(f"Error comparing files: {str(e)}")

def return_r_score(tensor1, tensor2, verbose=False):
    mask = ~(np.isnan(tensor1) | np.isnan(tensor2))
    nan_count = np.sum(~mask)
    if verbose:
        print('nan_count', nan_count)
    flat1 = tensor1[mask].flatten()
    flat2 = tensor2[mask].flatten()
    r_score = np.corrcoef(flat1, flat2)[0, 1]
    return r_score, nan_count

def get_tensor_from_file(file_path, verbose=False, take_first_n=True):
    with gzip.open(file_path, 'rb') as f:
        loaded_tensor = pickle.load(f)
        if loaded_tensor.dtype == torch.bfloat16:
            print(f'{file_path} is bfloat16, converting to float')
            loaded_tensor = loaded_tensor.float()
        if loaded_tensor.shape[0] > 7:
            if take_first_n:
                loaded_tensor = loaded_tensor[:7,:]
            else:
                loaded_tensor = loaded_tensor[-7:,:]
        if verbose:
            print('loaded_tensor.shape', loaded_tensor.shape)
    return loaded_tensor.numpy()

def compare_two_tensor_files(file1_path, file2_path, verbose=False, take_first_n=True):
    loaded_tensor1 = get_tensor_from_file(file1_path, verbose)
    loaded_tensor2 = get_tensor_from_file(file2_path, verbose)
    max_size = min(loaded_tensor1.shape[0], loaded_tensor2.shape[0])
    for i in range(max_size):
        print('loaded_tensor1.shape', loaded_tensor1.shape, i)
        print('loaded_tensor2.shape', loaded_tensor2.shape)
        print('loaded_tensor1[i].shape', loaded_tensor1[i].shape)
        print('loaded_tensor2[i].shape', loaded_tensor2[i].shape)
        r_score, nan_count = return_r_score(loaded_tensor1[i], loaded_tensor2[i], verbose)
        print('r_score', r_score)
    r_score, nan_count = return_r_score(loaded_tensor1, loaded_tensor2, verbose)
    print('r_score full', r_score)
    print('nan_count', nan_count)


if __name__ == "__main__":
        # Example usage
    # file1 = "/home/bagga005/algo/comp_data/stimulus_features/raw/language/friends_s05e03a.h5"
    # file2 = "/mnt/c/temp/friends_s05e03a.h5"
    # group = "friends_s05e03a"
    # dataset = 'language_last_hidden_state'#"language_pooler_output"
    file1 = "/teamspace/studios/this_studio/algo_data/stimulus_features/post/visual/friends_s01e24a.h5"
    file2 = "/teamspace/studios/this_studio/algo_data/stimulus_features/raw/visual/friends_s01e24a.h5"
    folder_path = "/home/bagga005/algo/comp_data/stimulus_features/raw/visual"
    folder_path = "/teamspace/studios/this_studio/algo_data/stimulus_features/raw/visual"
    #print(has_nan_for_folder(folder_path, 'visual'))
    group1 = "friends_s01e24a"
    group2 = "friends_s01e24a"
    dataset = "visual" #"language_pooler_output"
    #compare_h5_datasets(file1, file2, group1, group2, dataset)
    # are_equal, r_score = compare_h5_datasets(file1, file2, group1, group2, dataset)
    # print('r_score', r_score)
    # print(f"Are datasets equal? {are_equal}, R-score: {r_score:.6f}")
    file1 = "/workspace/temp/compare_dims/features_train-250.npy"
    file2 = "/workspace/temp/compare_dims/features_train-250-1.npy"
    compare_npy_values(file1, file2)
    # file1_path = "/mnt/c/temp/friends_s01e01a.npy"
    # file2_path = "/home/bagga005/algo/comp_data/stimulus_features/raw/visual/friends_s01e01a.npy"
    #compare_npy_shapes(file1, file2)
    # folder1 = "embeddings3"
    # folder2 = "embeddings4"
    # layer = "language_model_model_norm"
    # out_dir = utils.get_output_dir()
    # file_name1 = "friends_s06e01b_tr_108_language_model_model_norm.pt.gz"
    # file_name2 = "friends_s05e01a_tr_11_language_model_model_norm.pt.gz"
    # file1 = os.path.join(out_dir, folder1, layer,  file_name2)
    # file2 = os.path.join(out_dir, folder2, layer, file_name2)
    # compare_two_tensor_files(file1, file2, verbose=True)