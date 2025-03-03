import h5py
import numpy as np
import utils
import os

def get_session_order_number_for_key(session_task_dict, session_num):
    max_val = 1
    for key in session_task_dict.keys():
        ses_num, count = session_task_dict[key] 
        if int(ses_num) == session_num:
            max_val += 1
    return max_val


def read_subject_fmri_session_h5_write_summary(file_path, subject, create_new, base_value=0):
    """
    Print all first-level group names in an HDF5 file
    
    Args:
        file_path (str): Path to the .h5 file
    """
    with h5py.File(file_path, 'r') as f:
        session_task_dict = utils.load_viewing_session_for_subject(subject, not create_new)
        print("First level groups in the file:")
        max_videos_per_session = {}
        temp_session_task_dict = {}
        order_num_per_video = {}
        for key in f.keys():
            print(f"- {key}")
            key_val, value = parse_session_task(key)
            session_order_num = get_session_order_number_for_key(temp_session_task_dict, int(value) + base_value)
            order_num_per_video[key_val] = session_order_num
            temp_session_task_dict[key_val] = (int(value) + base_value, session_order_num)
            if value in max_videos_per_session:
                max_videos_per_session[value] = max(max_videos_per_session[value], session_order_num)
            else:
                max_videos_per_session[value] = session_order_num
        
        print(max_videos_per_session)
        
        for key in f.keys():
            print(f"- {key}")
            key_val, value = parse_session_task(key)
            session_order_num = order_num_per_video[key_val]
            max_session_num = max_videos_per_session[value]
            session_task_dict[key_val] = (int(value) + base_value, session_order_num, max_session_num)
            print(f"Key: {key_val}, session_id: {str(int(value) + base_value)}, in_session_order_num: {session_order_num}, max_session_num: {max_session_num}")
        utils.save_viewing_session_for_subject(subject, session_task_dict)

def make_session_summary_file_all_subjects():
    subjects_str = ['sub-01', 'sub-02', 'sub-03', 'sub-05']
    for sub in subjects_str:
        fr_file = sub + '_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
        mov_file = sub + '_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
        file_path = os.path.join(utils.get_data_root_dir(), "algonauts_2025.competitors","fmri",sub,"func",fr_file)
        read_subject_fmri_session_h5_write_summary(file_path, sub, True, 0)
        file_path = os.path.join(utils.get_data_root_dir(), "algonauts_2025.competitors","fmri",sub,"func",mov_file)
        read_subject_fmri_session_h5_write_summary(file_path, sub, False,85)

def parse_session_task(input_string):
    """
    Parse a session-task string to extract key and value.
    
    Args:
        input_string (str): String in format 'ses-XXX_task-YYY'
    
    Returns:
        tuple: (key, value) where value is between first - and _ and key is after second -
    
    Example:
        >>> parse_session_task('ses-001_task-s01e02a')
        ('s01e02a', '001')
    """
    try:
        # Split by underscore first
        parts = input_string.split('_')
        
        # Get the value (between first - and _)
        value = parts[0].split('-')[1]
        
        # Get the key (after second -)
        key = parts[1].split('-')[1]

        run = 0
        if len(parts) == 3:
            run = parts[2].split('-')[1]
            key = key + '_' + run
        
        return key, value
        
    except Exception as e:
        print(f"Error parsing string: {str(e)}")
        return None, None
    
if __name__ == "__main__":
    make_session_summary_file_all_subjects()