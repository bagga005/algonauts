import utils
from glob import glob
from tqdm import tqdm

from algonaut_funcs import extract_visual_features, get_vision_model, define_frames_transform
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
    save_dir_features = out_data_dir +  "stimulus_features/raw/visual/"
    feature_extractor, model_layer, device = get_vision_model()
    transform = define_frames_transform()
    # iterate across all the stimuli movie files
    iterator = tqdm(enumerate(stimuli.items()), total=len(list(stimuli)))
    for i, (stim_id, stim_path) in iterator:
        print(f"Extracting visual features for {stim_id}", stim_path)
        fn = f"{out_data_dir}/stim_id.h5"
        # Execute visual feature extraction
        visual_features = extract_visual_features(stim_path, tr, feature_extractor,
        model_layer, transform, device, save_dir_temp, fn, stim_id)

if __name__ == "__main__":
    extract_raw_visual_features()