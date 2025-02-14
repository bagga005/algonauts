import utils
import os
from glob import glob
from tqdm import tqdm
import h5py
import torch
from algonaut_funcs import extract_visual_features, get_vision_model, extract_audio_features, get_language_model, define_frames_transform, extract_language_features

def extract_raw_visual_features():
    root_data_dir = utils.get_data_root_dir()
    out_data_dir = utils.get_output_dir()
# As an exemple, extract visual features for season 1, episode 1 of Friends
    #episode_path = root_data_dir + "algonauts_2025.competitors/stimuli/movies/friends/s1/friends_s01e01a.mkv"
    # Collecting the paths to all the movie stimuli
    files = glob(f"{root_data_dir}algonauts_2025.competitors/stimuli/movies/**/**/*.mkv")
    files.sort()

    stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
    print(len(stimuli), list(stimuli)[:3], list(stimuli)[-3:])

    # Duration of each movie chunk, aligned with the fMRI TR of 1.49 seconds
    tr = 1.49

    # Saving directories
    save_dir_temp = utils.get_tmp_dir()
    #save_dir_features = out_data_dir +  "stimulus_features/raw/visual/"
    feature_extractor, model_layer, device = get_vision_model()
    transform = define_frames_transform()
    exclude_list =['friends_s03e05b', 'friends_s03e06a']
    # iterate across all the stimuli movie files
    iterator = tqdm(enumerate(stimuli.items()), total=len(list(stimuli)))
    for i, (stim_id, stim_path) in iterator:
        print(f"Extracting visual features for {stim_id}", stim_path)
        fn = os.path.join(out_data_dir, "stimulus_features", "raw", "visual", f"{stim_id}.h5")
        if os.path.exists(fn) or stim_id in exclude_list: continue; 
        # Execute visual feature extraction
        visual_features = extract_visual_features(stim_path, tr, feature_extractor,
        model_layer, transform, device, save_dir_temp, fn, stim_id)

def extract_raw_language_features():
    root_data_dir = utils.get_data_root_dir()
    out_data_dir = utils.get_output_dir()
# As an exemple, extract visual features for season 1, episode 1 of Friends
    #episode_path = root_data_dir + "algonauts_2025.competitors/stimuli/movies/friends/s1/friends_s01e01a.mkv"
    # Collecting the paths to all the movie stimuli
    files = glob(f"{root_data_dir}algonauts_2025.competitors/stimuli/transcripts/**/**/*.tsv")
    files.sort()

    stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
    print(len(stimuli), list(stimuli)[:3], list(stimuli)[-3:])

    # Duration of each movie chunk, aligned with the fMRI TR of 1.49 seconds
    tr = 1.49
    # Other parameters
    num_used_tokens = 510
    kept_tokens_last_hidden_state = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Saving directories
    save_dir_temp = utils.get_tmp_dir()
    #save_dir_features = out_data_dir +  "stimulus_features/raw/visual/"
    # Load the model and tokenizer
    model, tokenizer = get_language_model(device)
    exclude_list =[]
    # iterate across all the stimuli movie files
    iterator = tqdm(enumerate(stimuli.items()), total=len(list(stimuli)))
    for i, (stim_id, stim_path) in iterator:
        print(f"Extracting language features for {stim_id}", stim_path)
        fn = os.path.join(out_data_dir, "stimulus_features", "raw", "language", f"{stim_id}.h5")
        if os.path.exists(fn) or stim_id in exclude_list: continue; 
        # Execute language feature extraction
        extract_language_features(stim_path, model, tokenizer, num_used_tokens,
        kept_tokens_last_hidden_state, device,  fn, stim_id)
        


def extract_raw_audio_features():
    root_data_dir = utils.get_data_root_dir()
    out_data_dir = utils.get_output_dir()
# As an exemple, extract visual features for season 1, episode 1 of Friends
    #episode_path = root_data_dir + "algonauts_2025.competitors/stimuli/movies/friends/s1/friends_s01e01a.mkv"
    # Collecting the paths to all the movie stimuli
    files = glob(f"{root_data_dir}algonauts_2025.competitors/stimuli/movies/**/**/*.mkv")
    files.sort()

    stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
    print(len(stimuli), list(stimuli)[:3], list(stimuli)[-3:])

    # Duration of each movie chunk, aligned with the fMRI TR of 1.49 seconds
    tr = 1.49

    # Saving directories
    save_dir_temp = utils.get_tmp_dir()

    #save_dir_features = out_data_dir +  "stimulus_features/raw/visual/"
    sr = 22050
    exclude_list =['friends_s03e05b', 'friends_s03e06a']
    # iterate across all the stimuli movie files
    iterator = tqdm(enumerate(stimuli.items()), total=len(list(stimuli)))
    for i, (stim_id, stim_path) in iterator:
        print(f"Extracting audio features for {stim_id}", stim_path)
        fn = os.path.join(out_data_dir, "stimulus_features", "raw", "audio", f"{stim_id}.h5")
        if os.path.exists(fn) or stim_id in exclude_list: continue; 
        # Execute visual feature extraction
        extract_audio_features(stim_path, tr, sr, save_dir_temp, fn, stim_id)

if __name__ == "__main__":
    #extract_raw_visual_features()
    #extract_raw_audio_features()
    extract_raw_language_features()