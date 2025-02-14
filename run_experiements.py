import train
import h5py
def run_trainings():

    # root_data_dir = utils.get_data_root_dir()
    # areas_of_interest_path = os.path.join(root_data_dir, 'sub-01_modality-all_accuracy.npy')
    # npn = np.load(areas_of_interest_path)
    # print(npn)
    excluded_samples_start = 5  #@param {type:"slider", min:0, max:20, step:1}
    excluded_samples_end = 5  #@param {type:"slider", min:0, max:20, step:1}
    hrf_delay = 3  #@param {type:"slider", min:0, max:10, step:1}
    stimulus_window = 5  #@param {type:"slider", min:1, max:20, step:1}
    subject = 3
    movies_train = ["friends-s01", "friends-s02", "friends-s03", "friends-s06", "friends-s04", "friends-s05", "movie10-bourne", "movie10-figures", "movie10-life"] # @param {allow-input: true}
    #movies_train = ["movie10-wolf"] # @param {allow-input: true}
    movies_train_val = ["friends-s02"]
    movies_val = ["movie10-wolf"] # @param {allow-input: true}c
    training_handler = 'sklearn'
    experiment_comments = 'ridge'
    specific_modalities = ["all"]
    recurrence = 0 #not needed as feature extraction includes option to include features from previous time steps
    #movies_train = ["friends-s01"] # @param {allow-input: true
    
    # features = get_features(modality)
    # #print('features.keys()', features.keys())
    
    fmri = train.get_fmri(subject)
    #measure_yony_accuracy(subject, specific_modalities[0], fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train)
    # areas_of_interest_path = os.path.join(root_data_dir, 'eval_results', 'areas-of-interest.csv')
    # arr = utils.load_csv_to_array(areas_of_interest_path)
    # print('arr', arr[:100])
    # print('arr.shape', arr.shape)
    # utils.save_npy(arr, subject, modality)
    #features_train, fmri_train = align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train)
    #run_validation(subject, modality, features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val)
    # print('modality', modality)
    # print('features_train.shape', features_train.shape)
    # print('fmri_train.shape', fmri_train.shape)
    #train_for_all_subjects(excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train)
    print('starting for handler:', training_handler, ' with comments: ',experiment_comments)
    print('train_movies', movies_train)
    print('movies_train_val', movies_train_val)
    print('moviels_val', movies_val)
    # train.train_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, movies_train_val, training_handler, specific_modalities, recurrence)
    # train.validate_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, training_handler, specific_modalities, recurrence)
    # train.validate_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train_val, training_handler, specific_modalities, recurrence)
    # train.validate_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val, training_handler, specific_modalities, recurrence)
    movies_train = ["friends-s01"]
    features = train.get_features("all")
    #print('features', features['visual'].keys())
    #print('features', features['visual']['s01e01a'].shape)
    train.align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train)
    #validate_for_all_subjects(excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val)

    # Print the shape of the training fMRI responses and stimulus features: note
    # that the two have the same sample size!
    # print("Training fMRI responses shape:")
    # print(fmri_train.shape)
    # print('(Train samples × Parcels)')
    # print("\nTraining stimulus features shape:")
    # print(features_train.shape)
    # print('(Train samples × Features)')
    
    # # Train the encoding model
    # print("Training the encoding model... ", model_name)
    # model, training_time = train_encoding(features_train, fmri_train)
    # print(f"Training completed in {training_time:.2f} seconds")
    # utils.save_model(model, model_name)

    #model = utils.load_model(f'sub-01_modality-all')

def readh5(path):
    with h5py.File(path, 'r') as data:
        for episode in data.keys():
            print(episode)
            print(data[episode]['audio'].shape)


if __name__ == "__main__":
    #run_trainings()
    #file = '/home/bagga005/algo/comp_data/stimulus_features/raw/visual/friends_s01e01a.h5'
    file = '/home/bagga005/algo/comp_data/stimulus_features/raw/audio/friends_s01e01a.h5'
    readh5(file)
    #readh5('/home/bagga005/algo/comp_data/stimulus_features/raw/visual/friends_s01e01a_features_visual.h5')