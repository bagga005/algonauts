from algonaut_funcs import load_features_visual, preprocess_features, perform_pca
import utils
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
    n_components = 250
    run_pca('visual', n_components)