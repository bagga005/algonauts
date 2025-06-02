import os
import pickle
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
import json
import datetime
import pandas as pd

load_dotenv()

def get_stimulus_features_dir():
    return os.getenv("STIMULUS_FEATURES_DIR")

def get_stimulus_pre_features_dir():
    return os.getenv("STIMULUS_PRE_FEATURES_DIR")

def get_raw_data_dir():
    return os.getenv("RAW_DATA_DIR")

def get_pca_dir():
    return os.getenv("PCA_DIR")

def get_data_root_dir():
    return os.getenv("DATA_ROOT_DIR")

def get_tmp_dir():
    return os.getenv("TMP_DIR")

def get_embeddings_dir():
    return os.getenv("EMBEDDINGS_DIR")

def get_output_dir():
    return os.getenv("OUTPUT_DIR")

def get_mvl_model():
    return os.getenv("MVL_MODEL")

def get_wandb_config():
    return os.getenv("WANDB_PROJECT"), os.getenv("WANDB_MODEL_NAME"), os.getenv("WANDB_ENABLE")

def get_runpod_config():
    return os.getenv("RUNPOD_ID"), str_to_bool(os.getenv("RUNPOD_TERMINATE_ON_EXIT"))

def get_accuracy_json_file():
    return os.path.join(get_output_dir(), 'accuracy.json')

def get_network_accuracy_json_file():
    return os.path.join(get_output_dir(), 'network_accuracy.json')

def get_roi_network_map():
    """Returns the path to the ROI network mapping file"""
    return os.path.join(os.path.dirname(__file__), 'roi_network_map.json')

def save_model_pytorch(model, model_name):
    file_name = f'{model_name}.pth'
    full_path = os.path.join(get_output_dir(), 'models', file_name)
    
    torch.save(model.state_dict(), full_path)
def str_to_bool(s):
    return s.lower() in ("true", "1", "t", "yes", "y")

def isMockMode():
    strm = os.getenv("MOCK_MODE")
    if strm:
        return str_to_bool(strm)
    
def get_lora_config():
    return int(os.getenv("LORA_BATCH_SIZE")), int(os.getenv("LORA_EPOCH")), int(os.getenv("LORA_START_EPOCH"))

def get_model_checkpoint():
    return os.getenv("MODEL_CHECKPOINT")

def load_model_pytorch(model_name):
    file_name = f'{model_name}.pth'
    full_path = os.path.join(get_output_dir(), 'models', file_name)
    return torch.load(full_path, map_location=torch.device('cpu'))

def save_model_sklearn(model, model_name):
    file_name = f'{model_name}.pkl'
    full_path = os.path.join(get_output_dir(), 'models', file_name)
    pickle.dump(model, open(full_path, 'wb'))

def load_model_sklearn(model_name):
    file_name = f'{model_name}.pkl'
    full_path = os.path.join(get_output_dir(), 'models', file_name)
    return pickle.load(open(full_path, 'rb'))

def save_viewing_session_for_subject(subject, viewing_session):
    file_name = os.path.join(get_output_dir(), 'viewing_session', f'{subject}_viewing_session.pkl')
    with open(file_name, 'wb') as f:
        pickle.dump(viewing_session, f)

def load_viewing_session_for_subject(subject, require_file_exists=True):
    file_name = os.path.join(get_output_dir(), 'viewing_session', f'{subject}_viewing_session.pkl')
    if not os.path.exists(file_name):
        if require_file_exists:
            raise FileNotFoundError(f"Viewing session file not found for subject {subject}")
        else:
            return {}
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def get_shortstim_name(stimuli):
    if 'friends' in stimuli:
        return stimuli[8:]
    elif 'movie10' in stimuli:
        return stimuli[8:]
    else:
        return stimuli

def save_predictions_accuracy(fmri_val_pred, accuracy):
    ts_path = os.path.join(get_output_dir(), 'predictions', 'timeseries.npy')
    np.save(ts_path, fmri_val_pred)
    acc_path = os.path.join(get_output_dir(), 'predictions', 'accuracy.npy')
    np.save(acc_path, accuracy)

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

def get_roi_name(parcel):
    """
    Get the first ROI name that contains the given parcel number.
    
    Parameters
    ----------
    parcel : int
        The parcel number to lookup
        
    Returns
    -------
    str
        The name of the ROI containing the parcel
    """
    # Load the ROI network mapping from JSON
    with open(get_roi_network_map(), 'r') as f:
        roi_map = json.load(f)
    
    # Iterate through each ROI in the mapping
    for roi in roi_map:
        measure = roi["Measure"]
        ranges = roi["ranges"]
        
        # Check if parcel falls within any of the ranges for this ROI
        for range_dict in ranges:
            if range_dict["from"] <= parcel <= range_dict["to"]:
                return measure
    
    # If no matching ROI found
    return None

def write_encoding_accuracy_to_csv(file_name, encoding_accuracy, subject, modality):
    """
    Write encoding accuracy values to a CSV file.
    
    Parameters
    ----------
    file_name : str
        Path to the CSV file
    encoding_accuracy : numpy.ndarray
        Array containing accuracy values for each parcel
    subject : int
        Subject number
    modality : str
        Feature modality used
    """
    # Check if file exists, if so append column, otherwise create new file
    if os.path.exists(file_name):
        # Read existing file
        existing_df = pd.read_csv(file_name)
        # Add new column with subject as column name
        existing_df['sub-0'+ str(subject)] = encoding_accuracy
        # Write back to file
        existing_df.to_csv(file_name, index=False)
    else:
        # Create new DataFrame with initial data
        df = pd.DataFrame({
            'Number': range(1, len(encoding_accuracy)+1),
            'roi': [get_roi_name(i-1) for i in range(1, len(encoding_accuracy)+1)]
        })
        df['sub-0'+ str(subject)] = encoding_accuracy
        # Output to CSV
        df.to_csv(file_name, index=False)

def compute_encoding_accuracy(fmri_val, fmri_val_pred, subject, modality, print_output=True, write_to_csv=False):
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
    mean_encoding_accuracy = np.round(np.mean(encoding_accuracy), 3)
    std_encoding_accuracy = np.round(np.std(encoding_accuracy), 3)
    if print_output:    
        print(f"Encoding accuracy, sub-0{subject}, modality-{modality}, mean accuracy: {mean_encoding_accuracy}, std: {std_encoding_accuracy}")
    if write_to_csv:
        write_encoding_accuracy_to_csv('encoding_file.csv', encoding_accuracy, subject, modality)

    #plot_encoding_accuracy(subject, encoding_accuracy, modality)
    # utils.save_npy(encoding_accuracy, subject, modality)
    # # Save accuracy values to CSV
    # save_encoding_accuracy(encoding_accuracy, subject, modality)
    return mean_encoding_accuracy, encoding_accuracy

def append_network_accuracies_to_json(json_path, accuracy_tuples):
    """
    Append network accuracies to a JSON file.
    
    Parameters
    ----------
    json_path : str
        Path to the JSON file to update
    accuracy_tuples : list of tuples
        List of tuples where each tuple contains (measure_name, accuracy)
    
    Returns
    -------
    None
    """
    # Get current timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Load existing data if file exists, otherwise create empty list
    data = []
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            # If file exists but is not valid JSON, start with empty list
            data = []
    
    # Create entries for each measure and append to data
    for measure, accuracy in accuracy_tuples:
        entry = {
            "measure": measure,
            "accuracy": float(accuracy),  # Convert to float to ensure it's JSON serializable
            "timestamp": current_time
        }
        data.append(entry)
    
    # Write updated data back to file
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Appended {len(accuracy_tuples)} network accuracy entries to {json_path}")

def get_roi_name(parcel):
    """
    Get the first ROI name that contains the given parcel number.
    
    Parameters
    ----------
    parcel : int
        The parcel number to lookup
        
    Returns
    -------
    str
        The name of the ROI containing the parcel
    """
    # Load the ROI network mapping from JSON
    with open(get_roi_network_map(), 'r') as f:
        roi_map = json.load(f)
    
    # Iterate through each ROI in the mapping
    for roi in roi_map:
        measure = roi["Measure"]
        ranges = roi["ranges"]
        
        # Check if parcel falls within any of the ranges for this ROI
        for range_dict in ranges:
            if range_dict["from"] <= parcel <= range_dict["to"]:
                return measure
    
    # If no matching ROI found
    return None

def print_input_tokens_with_offsets(prompt, offsets, input_ids, pre_start=None, pre_end=None, post_start=None, post_end=None, last_chat_end=None):
    """
    Prints token number, corresponding text span, input ID, and marker for each token.
    Args:
        prompt (str): The original input string.
        offsets (list or tensor): Shape (T, 2). List of [start, end] for each token.
        input_ids (list or tensor): Shape (T,). List of token IDs.
        pre_start (int, optional): Token number to mark as 'pre_start'. Defaults to None.
        pre_end (int, optional): Token number to mark as 'pre_end'. Defaults to None.
        post_start (int, optional): Token number to mark as 'post_start'. Defaults to None.
        post_end (int, optional): Token number to mark as 'post_end'. Defaults to None.
        last_chat_end (int, optional): Token number to mark as 'last_chat_end'. Defaults to None.
    """
    # If offsets is a PyTorch tensor, convert to list
    if hasattr(offsets, 'tolist'):
        offsets = offsets.tolist()
    
    # If input_ids is a PyTorch tensor, convert to list
    if hasattr(input_ids, 'tolist'):
        input_ids = input_ids.tolist()
    
    # If batch dimension exists, use the first example
    if len(offsets) == 1 and isinstance(offsets[0], list):
        offsets = offsets[0]
    
    if len(input_ids) == 1 and isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    # Create mapping of token numbers to parameter names
    marker_map = {}
    if pre_start is not None:
        marker_map[pre_start] = 'pre_start'
    if pre_end is not None:
        marker_map[pre_end] = 'pre_end'
    if post_start is not None:
        marker_map[post_start] = 'post_start'
    if post_end is not None:
        marker_map[post_end] = 'post_end'
    if last_chat_end is not None:
        marker_map[last_chat_end] = 'last_chat_end'

    print("Token # | Token Text           | Input ID | Marker")
    print("--------+----------------------+----------+---------------")
    for i, ((start, end), token_id) in enumerate(zip(offsets, input_ids)):
        # Some special tokens may have (0, 0)
        if start == end:
            token_text = "[special token]"
        else:
            token_text = prompt[start:end]
        
        # Skip rows where token text is <IMG_CONTEXT>
        if token_text == "<IMG_CONTEXT>":
            continue
        
        # Get marker if token number matches any parameter
        marker = marker_map.get(i, "")
        
        print(f"{i:7} | {repr(token_text):20} | {token_id:8} | {marker}")

def compare_tensors(tensor1, tensor2):
    mismatch_count = 0
    for i in range(tensor1.shape[0]):
        for j in range(tensor1.shape[1]):
            if tensor1[i, j] != tensor2[i, j]:
                mismatch_count += 1
                break
    return mismatch_count