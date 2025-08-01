import train
from datetime import datetime
import traceback
from utils import load_viewing_session_for_subject, get_accuracy_json_file, isMockMode, get_runpod_config, get_output_dir, \
    set_hf_home_path, get_run_settings_file, get_stimulus_windows, get_hrf_delay
import os
import subprocess
import sys
import json

def load_training_settings():

    # Default settings
    default_settings = {
        "excluded_samples_start": 5,
        "excluded_samples_end": 5,
        "subject": 1,
        "include_viewing_sessions": False,
        "movies_train": ["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05"],
        "movies_val": ["friends-s06"],
        "training_handler": "sklearn",
        "experiment_comments": "experiment from settings file",
        "specific_modalities": ["language"],
        "run_training_1_subject": False,
        "run_training_all_subjects": False,
        "run_validation_1_subject": False,
        "run_validation_all_subjects": False
    }
    settings_file_path = get_run_settings_file()
    # Load settings from file if it exists
    if os.path.exists(settings_file_path):
        try:
            with open(settings_file_path, 'r') as f:
                file_settings = json.load(f)
            
            # Update defaults with file settings
            default_settings.update(file_settings)
            print(f"Settings loaded from: {settings_file_path}")
            
        except (json.JSONDecodeError, IOError) as e:
            raise Exception(f"Error loading settings file {settings_file_path}: {e}")
            
    else:
        raise Exception(f"Settings file {settings_file_path} not found")
    
    return default_settings

def run_trainings(experiment_name=None, results_output_directory=None):

    # root_data_dir = utils.get_data_root_dir()
    # areas_of_interest_path = os.path.join(root_data_dir, 'sub-01_modality-all_accuracy.npy')
    # npn = np.load(areas_of_interest_path)
    # print(npn)
    settings = load_training_settings()
    excluded_samples_start = settings["excluded_samples_start"]
    excluded_samples_end = settings["excluded_samples_end"]
    subject = settings["subject"]
    include_viewing_sessions = settings["include_viewing_sessions"]
    movies_train = settings["movies_train"]
    movies_val = settings["movies_val"]
    training_handler = settings["training_handler"]
    experiment_comments = settings["experiment_comments"]
    specific_modalities = settings["specific_modalities"]
    run_training_1_subject = settings["run_training_1_subject"]
    run_training_all_subjects = settings["run_training_all_subjects"]
    run_validation_1_subject = settings["run_validation_1_subject"]
    run_validation_all_subjects = settings["run_validation_all_subjects"]
    
    assert not(run_training_1_subject and run_training_all_subjects), "run_training_1_subject and run_training_all_subjects cannot be True at the same time"
    assert not(run_validation_1_subject and run_validation_all_subjects), "run_validation_1_subject and run_validation_all_subjects cannot be True at the same time"
    assert training_handler in ["sklearn", "loravision"], "training_handler must be either sklearn or loravision"

    config = {
        'trained_model_name': None, #'lora-7-checkpoint-params',#'lora-best-distributed',
        'subject': subject,
    }

    #loading fmri
    if subject == -1:
        fmri = train.get_fmri_for_all_subjects()
    else:
        fmri = train.get_fmri(subject)
        
    # print('fmri', fmri.keys())
    # print('fmri', fmri['fmri1'].keys())
    # print('fmri[s01e01a].shape', fmri['s01e01a'].shape)
    # fmri2 = train.get_fmri(2)
    # fmri3 = train.get_fmri(3)
    # fmri5 = train.get_fmri(5)
    # fmri1, fmri2, fmri3, fmri5 = train.align_fmri_for_all_subjects(fmri1, fmri2, fmri3, fmri5)
    # print('key length', len(fmri1.keys()))
    # print('key length', len(fmri2.keys()))
    # print('key length', len(fmri3.keys()))
    # print('key length', len(fmri5.keys()))
    # for key in fmri5.keys():
    #     if key != 's05e20a' and key != 's06e03a' and (fmri1[key].shape != fmri2[key].shape or fmri1[key].shape != fmri3[key].shape or fmri1[key].shape != fmri5[key].shape):
    #         print(' failed fmri1', key, fmri1[key].shape)
    #     # else:
    #     #     print('passed', key, fmri1[key].shape)
    
    # print('fmri2', fmri2['s01e01a'].shape)
    
    # print('fmri3', fmri3['s01e01a'].shape)
    
    # print('fmri5', fmri5['s01e01a'].shape)
    #measure_yony_accuracy(subject, specific_modalities[0], fmri, excluded_samples_start, excluded_samples_end, movies_train)
    # areas_of_interest_path = os.path.join(root_data_dir, 'eval_results', 'areas-of-interest.csv')
    # arr = utils.load_csv_to_array(areas_of_interest_path)
    # print('arr', arr[:100])
    # print('arr.shape', arr.shape)
    # utils.save_npy(arr, subject, modality)
    #features_train, fmri_train = align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end, movies_train)
    #run_validation(subject, modality, features, fmri, excluded_samples_start, excluded_samples_end, movies_val)
    # print('modality', modality)
    # print('features_train.shape', features_train.shape)
    # print('fmri_train.shape', fmri_train.shape)
    #train_for_all_subjects(excluded_samples_start, excluded_samples_end, movies_train)
    movies_train_val = None
    if isMockMode():
        print('******************* MOCK MODE *******************')
    print('\nstarting for handler:', training_handler, 'model name:', config['trained_model_name'], 'with comments: ',experiment_comments)
    print('train_movies', movies_train)
    print('movies_train_val', movies_train_val)
    print('moviels_val', movies_val)
    if run_training_1_subject:
        train.train_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, movies_train, movies_train_val, training_handler,  include_viewing_sessions, config, specific_modalities)
    #subject = 3
    #train.train_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, movies_train, movies_train_val, training_handler,  include_viewing_sessions, config, specific_modalities)
    if run_training_all_subjects:
        for subject in [1, 2, 3, 5]:
            fmri = train.get_fmri(subject)
            train.train_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, movies_train, movies_train_val, training_handler,  include_viewing_sessions, config, specific_modalities)
        # train.train_for_all_subjects(excluded_samples_start, excluded_samples_end, movies_train, movies_train_val, training_handler, include_viewing_sessions, \
        #                          config, specific_modalities)
        
    if run_validation_1_subject:    
        for movie in movies_val:
            train.validate_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, [movie], training_handler, \
                include_viewing_sessions, config, specific_modalities, plot_encoding_fig=False, break_up_by_network=True, write_accuracy_to_csv=False)

    if run_validation_all_subjects:
        for movie in movies_val:
            train.validate_for_all_subjects(excluded_samples_start, excluded_samples_end, [movie], training_handler, include_viewing_sessions, \
                config, specific_modalities, plot_encoding_fig=False, break_up_by_network=True, \
                write_accuracy_to_csv=False, save_combined_accuracy=True, experiment_name=experiment_name, results_output_directory=results_output_directory)
       
    #movies_train = ["friends-s01"]
    #features = train.get_features("all")
    #print('features', features['visual'].keys())
    #print('features', features['visual']['s01e01a'].shape)
    #train.align_features_and_fmri_samples(features, fmri, excluded_samples_start, excluded_samples_end, movies_train)
    #validate_for_all_subjects(excluded_samples_start, excluded_samples_end, movies_val)

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

def run_for_stimulus_window():
    excluded_samples_start = 5  #@param {type:"slider", min:0, max:20, step:1}
    excluded_samples_end = 5  #@param {type:"slider", min:0, max:20, step:1}
    hrf_delay = 3  #@param {type:"slider", min:0, max:10, step:1}
    stimulus_window = 5  #@param {type:"slider", min:1, max:20, step:1}
    include_viewing_sessions = False
    movies_train = ["friends-s05"]#["friends-s01","friends-s02", "friends-s06", "friends-s04", "friends-s05", "movie10-bourne",  "movie10-wolf"] # @param {allow-input: true}
    movies_val = ["friends-s03"] # @param {allow-input: true}c
    training_handler = 'pytorch'
    experiment_comments = 'test for stimulus window'
    specific_modalities = ["visual"]
    print('starting for handler:', training_handler, ' with comments: ',experiment_comments)
    print('train_movies', movies_train)
    print('moviels_val', movies_val)
    #subject = 3
    #fmri = train.get_fmri(subject)
    for stimulus_window in range(20, 21):
    #train.train_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, movies_train_val, training_handler,  include_viewing_sessions,specific_modalities)
        train.train_for_all_subjects(excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, movies_val, training_handler, include_viewing_sessions, specific_modalities, skip_if_accuracy_exists=True)
        train.validate_for_all_subjects(excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val, training_handler, include_viewing_sessions, specific_modalities, write_accuracy=True)
    #train.validate_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train, training_handler, include_viewing_sessions, specific_modalities,  recurrence)
    #train.validate_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_train_val, training_handler, include_viewing_sessions, specific_modalities, recurrence)
    #train.validate_for_all_modalities(subject, fmri, excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window, movies_val, training_handler, include_viewing_sessions, specific_modalities)
    


def readh5(path):
    with h5py.File(path, 'r') as data:
        for episode in data.keys():
            print(episode)
            print(data[episode]['language_pooler_output'].shape)
            print(data[episode]['language_last_hidden_state'].shape)
            #print(data[episode]['audio'].shape)

def cleanup_env():
    try:
        # Get the current date and time
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        outfile = os.path.join(get_output_dir(), 'exithistory.txt')
        with open(outfile, 'a') as file:
            file.write(f"{formatted_time} done\n")
    except Exception as e:
        print(e)
    runpod_id, runpod_terminate_on_exit = get_runpod_config()
    if runpod_id is not None and runpod_terminate_on_exit:
        print(f'Terminating runpod {runpod_id}')
        result = subprocess.run(["runpodctl", "remove", "pod", runpod_id])
        if result.returncode == 0:
            print("Pod successfully removed")
        else:
            print(f"Error removing pod: {result.stderr}")

if __name__ == "__main__":
    #encoding_accuracy = np.zeros((1000,1), dtype=np.float32)
    #train.plot_encoding_accuracy(3, encoding_accuracy, 'audio')
    experiment_name = None
    results_output_directory = None
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
    if len(sys.argv) > 2:
        results_output_directory = sys.argv[2]
        
    try:
        set_hf_home_path()
        run_trainings(experiment_name, results_output_directory)
    except Exception as e:
        traceback.print_exc()
    finally:
        cleanup_env()
    #run_for_stimulus_window()
    #accuracy_file = '/mnt/c/temp/accuracy_delay.json'
    #train.plot_accuracy_by_dimension(accuracy_file, modality='all', save_path='plot.jpg')
    #encoding_accuracy = np.zeros((1000), dtype=np.float32)
    #train.plot_encoding_accuracy(subject=1, encoding_accuracy=encoding_accuracy, modality='all')
    #file = '/home/bagga005/algo/comp_data/stimulus_features/raw/visual/friends_s01e01a.h5'
    #file = '/home/bagga005/algo/comp_data/stimulus_features/raw/language/friends_s01e01a.h5'
    #readh5(file)
    #readh5('/home/bagga005/algo/comp_data/stimulus_features/raw/visual/friends_s01e01a_features_visual.h5')
