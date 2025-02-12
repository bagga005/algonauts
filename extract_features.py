import utils
from algonaut_funcs import extract_visual_features, get_vision_model, define_frames_transform
def extract_raw_visual_features():
    root_data_dir = utils.get_data_root_dir()
# As an exemple, extract visual features for season 1, episode 1 of Friends
    episode_path = root_data_dir + "/algonauts_2025.competitors/stimuli/movies/friends/s1/friends_s01e01a.mkv"

    # Duration of each movie chunk, aligned with the fMRI TR of 1.49 seconds
    tr = 1.49

    # Saving directories
    save_dir_temp = utils.get_tmp_folder()
    save_dir_features = root_data_dir +  "/stimulus_features/raw/visual/"
    
    feature_extractor, model_layer, device = get_vision_model()
    transform = define_frames_transform()
    # Execute visual feature extraction
    visual_features = extract_visual_features(episode_path, tr, feature_extractor,
        model_layer, transform, device, save_dir_temp, save_dir_features)

if __name__ == "__main__":
    extract_raw_visual_features()