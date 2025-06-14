from train import load_stimulus_features_friends_s7, align_features_and_fmri_samples_friends_s7
import utils
import os
from glob import glob

def get_num_chunks(episode_id):
    season = 's1'
    if 's02' in episode_id:
        season = 's2'
    elif 's03' in episode_id:
        season = 's3'
    elif 's04' in episode_id:
        season = 's4'
    elif 's05' in episode_id:
        season = 's5'
    elif 's06' in episode_id:
        season = 's6'
    elif 's07' in episode_id:
        season = 's7'
    season_folder = os.path.join(utils.get_output_dir(), 'video_chunks', season)
    files = glob(f"{season_folder}/friends_{episode_id}_*.mp4")
    return len(files), season_folder

root_data_dir = utils.get_data_root_dir()
features_friends_s7 = load_stimulus_features_friends_s7(root_data_dir)

aligned_features_friends_s7 =align_features_and_fmri_samples_friends_s7(features_friends_s7, root_data_dir)



for sub, features in aligned_features_friends_s7.items():
    print(sub)
    for epi, feat_epi in features.items():
        print(epi, feat_epi.shape)
        print(get_num_chunks(epi))
    break

# Empty submission predictions dictionary
# submission_predictions = {}

# # Loop through each subject
# desc = "Predicting fMRI responses of each subject"
# for sub, features in tqdm(aligned_features_friends_s7.items(), desc=desc):

#     # Initialize the nested dictionary for each subject's predictions
#     submission_predictions[sub] = {}

#     # Loop through each Friends season 7 episode
#     for epi, feat_epi in features.items():

#         # Predict fMRI responses for the aligned features of this episode, and
#         # convert the predictions to float32
#         fmri_pred = baseline_models[sub].predict(feat_epi).astype(np.float32)

#         # Store formatted predictions in the nested dictionary
#         submission_predictions[sub][epi] = fmri_pred