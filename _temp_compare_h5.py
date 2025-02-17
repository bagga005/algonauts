import h5py
import numpy as np

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
            data1_rounded = np.round(data1, decimals=4)
            data2_rounded = np.round(data2, decimals=4)
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
                diff_mask = ~np.isclose(data1, data2, rtol=1e-5, atol=1e-8)
                diff_indices = np.where(diff_mask)
                
                if len(diff_indices[0]) > 0:
                    print(f"Datasets differ in {len(diff_indices[0])} positions")
                    # Convert zip to list before slicing to get first 5 differences
                    for idx in list(zip(*diff_indices))[:5]:
                        print(f"Position {idx}: {data1[idx]} vs {data2[idx]}")
                        
            return is_equal
            
    except KeyError as e:
        print(f"Error: Group or dataset not found: {e}")
        return False
    
    
    # Example usage
# file1 = "/home/bagga005/algo/comp_data/stimulus_features/raw/language/friends_s05e03a.h5"
# file2 = "/mnt/c/temp/friends_s05e03a.h5"
# group = "friends_s05e03a"
# dataset = 'language_last_hidden_state'#"language_pooler_output"

file1 = "/home/bagga005/algo/comp_data/stimulus_features/raw/visual/friends_s01e01a_features_visual.h5"
file2 = "/home/bagga005/algo/comp_data/stimulus_features/raw/visual/friends_s01e01a.h5"
group1 = "s01e01a"
group2 = "friends_s01e01a"
dataset = 'visual'#"language_pooler_output"

are_equal = compare_h5_datasets(file1, file2, group1, group2, dataset)