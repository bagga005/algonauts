import os
import pickle
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import torch
load_dotenv()

def get_data_root_dir():
    return os.getenv("DATA_ROOT_DIR")

def save_model(model, model_name):
    file_name = f'{model_name}.pth'
    full_path = os.path.join(get_data_root_dir(), 'models', file_name)
    torch.save(model.state_dict(), full_path)

def load_model(model_name):
    file_name = f'{model_name}.pth'
    full_path = os.path.join(get_data_root_dir(), 'models', file_name)
    return torch.load(full_path)

def save_npy(encoding_accuracy, subject, modality):
    """
    Save encoding accuracy values to both CSV and NPY files.
    
    Parameters
    ----------
    encoding_accuracy : numpy.ndarray
        Array containing accuracy values for each parcel
    subject : int
        Subject number
    modality : str
        Feature modality used
    """
    # Create eval_results directory if it doesn't exist
    root_dir = get_data_root_dir()
    eval_dir = os.path.join(root_dir, 'eval_results')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Generate filenames
    npy_filename = f'sub-{str(subject).zfill(2)}_modality-{modality}_accuracy.npy'
    
    # Save to NPY file
    npy_filepath = os.path.join(eval_dir, npy_filename)
    np.save(npy_filepath, encoding_accuracy)

def load_csv_to_array(csv_path):
    """
    Load rows 2-1001 from a CSV file into a numpy array.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file
        
    Returns
    -------
    numpy.ndarray
        Array of shape (1000,) containing the values
    """
    
    # Read CSV, skip first row (assuming it's a header), and take rows 2-1001
    # Note: Python uses 0-based indexing, so rows 2-1001 are at indices 1-1000
    df = pd.read_csv(csv_path, skiprows=1, nrows=1000, header=None)
    
    # Convert to numpy array and reshape to (1000,)
    return df.values.flatten()
    

