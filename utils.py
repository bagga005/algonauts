import os
import pickle
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
load_dotenv()

def get_stimulus_features_dir():
    return os.getenv("STIMULUS_FEATURES_DIR")

def get_raw_data_dir():
    return os.getenv("RAW_DATA_DIR")

def get_pca_dir():
    return os.getenv("PCA_DIR")

def get_data_root_dir():
    return os.getenv("DATA_ROOT_DIR")

def get_tmp_dir():
    return os.getenv("TMP_DIR")

def get_output_dir():
    return os.getenv("OUTPUT_DIR")

def save_model_pytorch(model, model_name):
    file_name = f'{model_name}.pth'
    full_path = os.path.join(get_output_dir(), 'models', file_name)
    torch.save(model.state_dict(), full_path)

def load_model_pytorch(model_name):
    file_name = f'{model_name}.pth'
    full_path = os.path.join(get_output_dir(), 'models', file_name)
    return torch.load(full_path)

def save_model_sklearn(model, model_name):
    file_name = f'{model_name}.pkl'
    full_path = os.path.join(get_output_dir(), 'models', file_name)
    pickle.dump(model, open(full_path, 'wb'))

def load_model_sklearn(model_name):
    file_name = f'{model_name}.pkl'
    full_path = os.path.join(get_output_dir(), 'models', file_name)
    return pickle.load(open(full_path, 'rb'))

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
    

def analyze_fmri_distribution(fmri_data):
    """
    Analyzes the distribution of fMRI values.
    
    Parameters:
    fmri_data: numpy array or torch tensor of fMRI values
    """
    # Convert to numpy if it's a torch tensor
    if torch.is_tensor(fmri_data):
        fmri_data = fmri_data.cpu().numpy()
    
    total_values = fmri_data.size
    positive_values = np.sum(fmri_data > 0)
    negative_values = np.sum(fmri_data < 0)
    zero_values = np.sum(fmri_data == 0)
    
    mean_value = np.mean(fmri_data)
    std_value = np.std(fmri_data)
    min_value = np.min(fmri_data)
    max_value = np.max(fmri_data)
    
    print(f"Distribution Analysis of fMRI Data:")
    print(f"Total values: {total_values:,}")
    print(f"Positive values: {positive_values:,} ({(positive_values/total_values)*100:.2f}%)")
    print(f"Negative values: {negative_values:,} ({(negative_values/total_values)*100:.2f}%)")
    print(f"Zero values: {zero_values:,} ({(zero_values/total_values)*100:.2f}%)")
    print(f"\nStatistics:")
    print(f"Mean: {mean_value:.4f}")
    print(f"Standard deviation: {std_value:.4f}")
    print(f"Min value: {min_value:.4f}")
    print(f"Max value: {max_value:.4f}")

def compute_encoding_accuracy(fmri_val, fmri_val_pred, subject, modality):
    """
    Compare the  recorded (ground truth) and predicted fMRI responses, using a
    Pearson's correlation. The comparison is perfomed independently for each
    fMRI parcel. The correlation results are then plotted on a glass brain.

    Parameters
    ----------
    fmri_val : float
        fMRI responses for the validation movies.
    fmri_val_pred : float
        Predicted fMRI responses for the validation movies
    subject : int
        Subject number used to train and validate the encoding model.
    modality : str
        Feature modality used to train and validate the encoding model.

    """

    ### Correlate recorded and predicted fMRI responses ###
    encoding_accuracy = np.zeros((fmri_val.shape[1]), dtype=np.float32)
    for p in range(len(encoding_accuracy)):
        encoding_accuracy[p] = pearsonr(fmri_val[:, p],
            fmri_val_pred[:, p])[0]
    print('encoding_accuracy.shape', encoding_accuracy.shape)
    mean_encoding_accuracy = np.round(np.mean(encoding_accuracy), 3)
    std_encoding_accuracy = np.round(np.std(encoding_accuracy), 3)
    print(f"Encoding accuracy, sub-0{subject}, modality-{modality}, mean accuracy: {mean_encoding_accuracy}, std: {std_encoding_accuracy}")

    #plot_encoding_accuracy(subject, encoding_accuracy, modality)
    # utils.save_npy(encoding_accuracy, subject, modality)
    # # Save accuracy values to CSV
    # save_encoding_accuracy(encoding_accuracy, subject, modality)
    return mean_encoding_accuracy