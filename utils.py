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

def get_embeddings_combined_dir():
    return os.getenv("EMBEDDINGS_COMBINED_DIR")

def get_output_dir():
    return os.getenv("OUTPUT_DIR")

def get_mvl_model():
    return os.getenv("MVL_MODEL")

def get_mvl_extraction_format():
    if get_mvl_pix_last():
        return 1
    else:
        return 0

def get_mvl_pix_last():
    return str_to_bool(os.getenv("MVL_PIX_LAST"))

def get_mvl_skip_pix():
    return str_to_bool(os.getenv("MVL_SKIP_PIX"))

def get_mvl_simple_extraction():
    return str_to_bool(os.getenv("MVL_SIMPLE_EXTRACTION"))

def get_mvl_batch_size():
    return int(os.getenv("MVL_BATCH_SIZE"))

def get_stimuli_prefix():
    return os.getenv("STIMULI_PREFIX")

def get_hf_token():
    return os.getenv("HF_TOKEN")

def get_min_length_for_summary():
    return int(os.getenv("MIN_LENGTH_FOR_SUMMARY"))

def get_wandb_config():
    return os.getenv("WANDB_PROJECT"), os.getenv("WANDB_MODEL_NAME"), os.getenv("WANDB_ENABLE")

def get_runpod_config():
    return os.getenv("RUNPOD_ID"), str_to_bool(os.getenv("RUNPOD_TERMINATE_ON_EXIT"))

def get_accuracy_json_file():
    return os.path.join(get_output_dir(), 'accuracy.json')

def get_network_accuracy_json_file():
    return os.path.join(get_output_dir(), 'network_accuracy.json')

def get_subject_network_accuracy_file_for_experiement(experiment_name, results_output_directory):
    if results_output_directory:
        if not os.path.exists(results_output_directory):
            os.makedirs(results_output_directory, exist_ok=True)
        filepath = os.path.join(results_output_directory, experiment_name + '_all_subjects_accuracy.csv')
    else:
        filepath = os.path.join(get_output_dir(), 'all_subjects_accuracy.csv')
    return filepath

def get_embeddings_format():
    embeddings_format = os.getenv("EMBEDDINGS_FORMAT")
    if not embeddings_format:
        return '1'
    return embeddings_format

def get_experiment_name():
    exp_name = os.getenv("EXPERIMENT_NAME")
    if not exp_name:
        return 'EXPERIMENT_NAME'
    return exp_name

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
    
def get_run_settings_file():
    dir_path = os.getenv("RUN_SETTINGS_DIR")
    if dir_path and dir_path.strip() != '':
        settings_file = os.getenv("RUN_SETTING")
        if settings_file and settings_file.strip() != '':
            return os.path.join(dir_path, settings_file+ ".json")
        else:
            raise Exception("RUN_SETTINGS_FILE is not set")
    else:
        raise Exception("RUN_SETTINGS_DIR is not set")


def get_lora_config():
    return int(os.getenv("LORA_BATCH_SIZE")), int(os.getenv("LORA_EPOCH")), int(os.getenv("LORA_START_EPOCH"))

def get_lora_prediction_subject():
    return int(os.getenv("LORA_PRED_SUBJECT"))

def get_model_checkpoint():
    mck = os.getenv("MODEL_CHECKPOINT")
    if mck is None or mck == '':
        return None
    return mck

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

def get_predictions_file_path(subject, movie_name):
    movie = 'movie'
    if movie_name:
        movie = movie_name
    exp_name = get_experiment_name()
    predictions_dir = os.path.join(get_output_dir(), 'predictions', exp_name)
    predictions_file = os.path.join(predictions_dir, f'sub-{subject}_predictions_{movie}.npy')
    return predictions_file

def get_accuracy_file_path(subject, movie_name):
    movie = 'movie'
    if movie_name:
        movie = movie_name
    exp_name = get_experiment_name()
    accuracy_dir = os.path.join(get_output_dir(), 'predictions', exp_name)
    accuracy_file = os.path.join(accuracy_dir, f'sub-{subject}_accuracy_{movie}.npy')
    return accuracy_file

def save_predictions_accuracy(subject, movie_name, fmri_val_pred, accuracy):
    if fmri_val_pred is not None:
        ts_path = get_predictions_file_path(subject, movie_name)
        ts_dir = os.path.dirname(ts_path)
        os.makedirs(ts_dir, exist_ok=True)
        np.save(ts_path, fmri_val_pred)
    if accuracy is not None:
        acc_path = get_accuracy_file_path(subject, movie_name)
        acc_dir = os.path.dirname(acc_path)
        os.makedirs(acc_dir, exist_ok=True)
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

def get_last_x_words(long_text, x):
    words = long_text.split()
    return " ".join(words[-x:])

def normalize_and_clean_word(word):
        """Normalize word by converting to lowercase and keeping only alphanumeric characters"""
        new_word = ''.join(c.lower() for c in word if c.isalnum())
        # if new_word == 'cmon':
            #new_word = 'come'
        return new_word
def set_hf_home_path():
    if os.getenv("HF_HOME") is not None:
        os.environ['HF_HOME'] = os.getenv("HF_HOME")

def save_embedding_metadata(transcript_id, metadata):
    meta_file = get_embeddding_meta_file_name(transcript_id)

    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)

def get_embeddding_meta_file_name(transcript_id):
    embeddings_dir = os.path.join(get_output_dir(), get_embeddings_dir(), 'metadata')
    os.makedirs(embeddings_dir, exist_ok=True)
    embeddings_prefix = f"{transcript_id}"
    meta_file = os.path.join(embeddings_dir, f"{embeddings_prefix}_metadata.json")
    return meta_file

def is_transcript_already_processed(transcript_id):
    meta_file = get_embeddding_meta_file_name(transcript_id)
    return os.path.exists(meta_file)

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

def print_input_tokens_with_offsets(prompt, offsets, input_ids, pre_start=None, pre_end=None, post_start=None, post_end=None):
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

    # Create mapping of token numbers to parameter names (allow multiple per token)
    from collections import defaultdict
    marker_map = defaultdict(list)
    if pre_start is not None:
        marker_map[pre_start].append('pre_start')
    if pre_end is not None:
        marker_map[pre_end].append('pre_end')
    if post_start is not None:
        marker_map[post_start].append('post_start')
    if post_end is not None:
        marker_map[post_end].append('post_end')


    log_to_file("Token # | Token Text           | Input ID | Marker")
    log_to_file("--------+----------------------+----------+---------------")
    for i, ((start, end), token_id) in enumerate(zip(offsets, input_ids)):
        # Some special tokens may have (0, 0)
        if start == end:
            token_text = "[special token]"
        else:
            token_text = prompt[start:end]
        
        # Skip rows where token text is <IMG_CONTEXT>
        if token_text == "<IMG_CONTEXT>":
            continue
        
        # Get markers if token number matches any parameter (comma-separated)
        markers = marker_map.get(i, [])
        marker = ", ".join(markers) if markers else ""
        
        log_to_file(f"{i:7} | {repr(token_text):20} | {token_id:8} | {marker}")

def compare_tensors(tensor1, tensor2):
    mismatch_count = 0
    for i in range(tensor1.shape[0]):
        for j in range(tensor1.shape[1]):
            if tensor1[i, j] != tensor2[i, j]:
                mismatch_count += 1
                break
    return mismatch_count

def log_to_file(*args):
    """
    Append a message to a hardcoded log file.
    
    Args:
        *args: Multiple arguments that will be converted to strings and joined
    """
    log_file_path = "debug_log.txt"  # Hardcoded file path
    
    try:
        # Convert all arguments to strings and join them with spaces
        message = ' '.join(str(arg) for arg in args)
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    except Exception as e:
        print(f"Error writing to log file: {e}")

def consolidate_results(pca_dim, network):
    """
    Consolidate evaluation results from different strategies into a single CSV file.
    
    Parameters
    ----------
    pca_dim : int
        PCA dimension used for the experiments
    network : str
        Network name to extract results for (e.g., "Visual", "Somatomotor", etc.)
        
    Returns
    -------
    str or None
        Path to the consolidated CSV file if successful, None if no results found
    """
    import pandas as pd
    import os
    from datetime import datetime
    
    # Get the combined embeddings directory
    combined_embeddings_dir = get_embeddings_combined_dir()
    output_dir = get_output_dir()
    combined_dir_path = os.path.join(output_dir, combined_embeddings_dir)
    
    if not os.path.exists(combined_dir_path):
        print(f"Combined embeddings directory not found: {combined_dir_path}")
        return None
    
    # List to store consolidated data
    consolidated_data = []
    
    # Iterate through strategy directories
    for strategy_name in os.listdir(combined_dir_path):
        strategy_path = os.path.join(combined_dir_path, strategy_name)
        
        # Skip if not a directory
        if not os.path.isdir(strategy_path):
            continue
            
        # Check if evaluation results file exists
        eval_dir = os.path.join(strategy_path, 'evals')
        results_file_path = get_subject_network_accuracy_file_for_experiement(
            strategy_name + '-' + str(pca_dim), eval_dir
        )
        
        if os.path.exists(results_file_path):
            try:
                # Read the results CSV
                df = pd.read_csv(results_file_path)
                
                # Find the row for the specified network
                network_row = df[df['network'] == network]
                
                if not network_row.empty:
                    # Extract values
                    row_data = network_row.iloc[0]
                    avg = row_data['average']
                    subject_1 = row_data['subject 1']
                    subject_2 = row_data['subject 2'] 
                    subject_3 = row_data['subject 3']
                    subject_5 = row_data['subject 5']
                    
                    # Get file creation/modification time
                    file_stat = os.stat(results_file_path)
                    creation_time = datetime.fromtimestamp(file_stat.st_mtime)
                    date_time = creation_time.strftime("%Y-%m-%d %H:%M")
                    
                    # Add to consolidated data
                    consolidated_data.append({
                        'strategy': strategy_name,
                        'avg': avg,
                        'subject 1': subject_1,
                        'subject 2': subject_2,
                        'subject 3': subject_3,
                        'subject 5': subject_5,
                        'date_time': date_time
                    })
                    
                    print(f"Added strategy: {strategy_name}")
                else:
                    print(f"Network '{network}' not found in {results_file_path}")
                    
            except Exception as e:
                print(f"Error processing {results_file_path}: {e}")
                continue
        else:
            print(f"Results file not found for strategy {strategy_name}: {results_file_path}")
    
    # Create consolidated DataFrame and save
    if consolidated_data:
        consolidated_df = pd.DataFrame(consolidated_data)
        
        # Sort by average accuracy (descending)
        consolidated_df = consolidated_df.sort_values('avg', ascending=False)
        
        # Save consolidated CSV
        output_filename = f"consolidated-{network}-{pca_dim}.csv"
        output_path = os.path.join(combined_dir_path, output_filename)
        consolidated_df.to_csv(output_path, index=False)
        
        print(f"\nConsolidated results saved to: {output_path}")
        print(f"Included {len(consolidated_data)} strategies for network '{network}' with PCA dimension {pca_dim}")
        
        # Print summary statistics
        print(f"\nTop 5 strategies by average accuracy:")
        top_5 = consolidated_df.head(5)
        for idx, row in top_5.iterrows():
            print(f"  {row['strategy']}: {row['avg']:.4f}")
        
        return output_path
    else:
        print(f"No results found for network '{network}' with pca_dim {pca_dim}")
        return None