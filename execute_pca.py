#from algonaut_funcs import load_features_visual, preprocess_features, perform_pca
#from extract_features import do_pca_npy, reduce_dims_npy
from utils_video import extract_video_chucks
#import utils
import os
def run_pca(modality, n_components):
    # Choose modality and PCs
    

    features_dir = utils.get_output_dir()
    features_dir = os.path.join(features_dir, 'stimulus_features', 'raw', modality)
    print(features_dir)
    # Load the stimulus features
    features = load_features_visual(features_dir, 'friends_s01e02a')
    # Preprocess the stimulus features
    prepr_features = preprocess_features(features)

    # Perform PCA
    features_pca = perform_pca(prepr_features, n_components, modality)
    print('features_pca.shape', features_pca.shape)

if __name__ == "__main__":
    # n_components = 250
    # # run_pca('visual', n_components)
    # stim_folder = utils.get_stimulus_features_dir()
    # inpath = os.path.join(stim_folder, 'pca', 'friends_movie10', 'audio', 'features_train-whisper2000.npy')
    # #inpath = "/home/bagga005/algo/comp_data/stimulus_features/pca/friends_movie10/language/features_train.npy"
    # outfile = os.path.join(stim_folder, 'pca', 'friends_movie10', 'audio', 'features_train-250-1.npy')
    # reduce_dims_npy(inpath, outfile,   n_components = 250)
    # outfile = os.path.join(stim_folder, 'pca', 'friends_movie10', 'audio', 'features_train-500-1.npy')
    # reduce_dims_npy(inpath, outfile,  n_components = 500)
    # outfile = os.path.join(stim_folder, 'pca', 'friends_movie10', 'audio', 'features_train-1000-1.npy')
    # reduce_dims_npy(inpath, outfile,  n_components = 1000)
    
    # modality = 'audio'
    # do_zscore = False
    # outfile = os.path.join(stim_folder, 'pca', 'friends_movie10', 'audio', 'features_train-250.npy')
    # do_pca_npy(inpath, outfile, modality, do_zscore, n_components = 250)
    # outfile = os.path.join(stim_folder, 'pca', 'friends_movie10', 'audio', 'features_train-500.npy')
    # do_pca_npy(inpath, outfile, modality, do_zscore, n_components = 500)
    # outfile = os.path.join(stim_folder, 'pca', 'friends_movie10', 'audio', 'features_train-1000.npy')
    # do_pca_npy(inpath, outfile, modality, do_zscore, n_components = 1000)
    extract_video_chucks()