from extract_features import do_pca_npy, reduce_dims_npy

if __name__ == "__main__":

    #stim_folder = utils.get_stimulus_features_dir()
    inpath = "/workspace/temp/compare_dims/features_train-250.npy"
    #inpath = "/home/bagga005/algo/comp_data/stimulus_features/pca/friends_movie10/language/features_train.npy"
    outfile = "/workspace/temp/compare_dims/features_train-250-1.npy"
    reduce_dims_npy(inpath, outfile,   n_components = 250)
    # outfile = os.path.join(stim_folder, 'pca', 'friends_movie10', 'audio', 'features_train-500-1.npy')
    # reduce_dims_npy(inpath, outfile,  n_components = 500)
    # outfile = os.path.join(stim_folder, 'pca', 'friends_movie10', 'audio', 'features_train-1000-1.npy')
    # reduce_dims_npy(inpath, outfile,  n_components = 1000)