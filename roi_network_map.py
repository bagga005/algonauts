import json
import utils


def get_breakup_by_network(fmri_val, fmri_val_pred):
    """
    Break up fMRI validation data by network regions defined in roi_network_map.json.
    
    Parameters
    ----------
    fmri_val : numpy.ndarray
        Ground truth fMRI validation data
    fmri_val_pred : numpy.ndarray
        Predicted fMRI validation data
    
    Returns
    -------
    list of tuples
        Each tuple contains (measure_name, network_fmri_val, network_fmri_val_pred)
        where the fMRI data only includes the ROIs specified in the network ranges
    """
    # Load the network mapping
    network_map_path = utils.get_roi_network_map()  # You'll need to implement this in utils.py
    with open(network_map_path, 'r') as f:
        network_map = json.load(f)
    
    network_breakups = []

    for network in network_map:
        measure = network['Measure']
        ranges = network['ranges']
        #print(measure)
        #print(ranges)
        
        # Initialize arrays to store the concatenated ROIs for this network
        network_indices = []
        
        # Collect all indices from the ranges
        for range_dict in ranges:
            start_idx = range_dict['from']
            end_idx = range_dict['to']  # inclusive
            network_indices.extend(range(start_idx, end_idx + 1))
        #print('network_indices', network_indices)
        
        # Extract the relevant columns for this network
        network_fmri_val = fmri_val[:, network_indices]
        network_fmri_val_pred = fmri_val_pred[:, network_indices]
        # print('network_fmri_val.shape', network_fmri_val.shape)
        # print('network_fmri_val', network_fmri_val[500,161])
        # Add to our results
        network_breakups.append((measure, network_fmri_val, network_fmri_val_pred))
    
    return network_breakups


