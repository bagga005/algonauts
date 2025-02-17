import h5py
import numpy as np
from utils import get_shortstim_name
def compare_h5_datasets(file1_path, file2_path, group_name1, group_name2, dataset_name):
    """
    Compare datasets from two HDF5 files.
    
    Parameters
    ----------
    file1_path : str
        Path to first HDF5 file
    file2_path : str
        Path to second HDF5 file
    group_name : str
        Name of the group containing the dataset
    dataset_name : str
        Name of the dataset to compare
    
    Returns
    -------
    bool
        True if datasets are identical, False otherwise
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
                return False
            # Compare data after rounding to 4 decimal places, ignoring NaNs
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
                    # Get all differences and their magnitudes
                    differences = []
                    for idx in zip(*diff_indices):
                        diff_magnitude = abs(float(data1[idx]) - float(data2[idx]))
                        differences.append((idx, data1[idx], data2[idx], diff_magnitude))
                    
                    # Sort by difference magnitude in descending order and show top 50
                    differences.sort(key=lambda x: x[3], reverse=True)
                    for idx, val1, val2, magnitude in differences[:50]:
                        print(f"Position {idx}: {val1} vs {val2} (diff: {magnitude:.6f})")
                else:
                    print("Datasets are identical in range!")
                        
            return is_equal
            
    except KeyError as e:
        print(f"Error: Group or dataset not found: {e}")
        return False
    
    


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
            


if __name__ == "__main__":
        # Example usage
    # file1 = "/home/bagga005/algo/comp_data/stimulus_features/raw/language/friends_s05e03a.h5"
    # file2 = "/mnt/c/temp/friends_s05e03a.h5"
    # group = "friends_s05e03a"
    # dataset = 'language_last_hidden_state'#"language_pooler_output"

    file2 = "/home/bagga005/algo/comp_data/stimulus_features/raw/visual/friends_s01e01a_features_visual.h5"
    file1 = "/home/bagga005/algo/comp_data/stimulus_features/raw/visual/friends_s01e01a.h5"
    # group1 = "s01e01a"
    # group2 = "friends_s01e01a"
    # dataset = 'visual'#"language_pooler_output"

    # are_equal = compare_h5_datasets(file1, file2, group1, group2, dataset)
    file1 = "/home/bagga005/algo/comp_data/stimulus_features/pca/friends_movie10/visual/features_train_new.npy"
    file2 = "/home/bagga005/algo/comp_data/stimulus_features/pca/friends_movie10/visual/features_train_orig.npy"
    # file1_path = "/mnt/c/temp/friends_s01e01a.npy"
    # file2_path = "/home/bagga005/algo/comp_data/stimulus_features/raw/visual/friends_s01e01a.npy"
    compare_npy_shapes(file1, file2)