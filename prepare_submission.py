from train import load_stimulus_features_friends_s7, align_features_and_fmri_samples_friends_s7, \
    get_features, align_features_and_fmri_samples, get_fmri
import utils
import zipfile
import os
from glob import glob
import numpy as np
from algonaut_funcs import prepare_s7_fmri_for_alignment

# #dummy np
# dummy_np = np.zeros((1000, 1000))
# utils.save_predictions_accuracy(1, dummy_np, None)
# exit()

FORMAT_CODA = 1
FORMAT_FLAT = 2
FORMAT_WITH_MODALITY =3

def get_dict_key_for_subject(sub):
    return f'sub-0{sub}'

def get_boundary_from_fmri_for_movie_for_subject(subject, movie_name):
    if movie_name == "friends-s07":
        boundary = prepare_s7_fmri_for_alignment(subject)
        return boundary
    else:
        if movie_name[:7] == 'friends':
            id = movie_name[8:]
        elif movie_name[:7] == 'movie10':
            id = movie_name[8:]
        fmri = get_fmri(subject)
        movie_splits = [key for key in fmri if id in key[:len(id)]]
        boundary = []
        for split in movie_splits:
            boundary.append((split, fmri[split].shape[0]))
        print(boundary)
        return boundary

def append_to_dict(dict, subject, stimuli_id, data, format, modality):
    if format == FORMAT_CODA:
        dict[get_dict_key_for_subject(subject)][stimuli_id] = data
    elif format == FORMAT_FLAT:
        dict[stimuli_id] = data
    elif format == FORMAT_WITH_MODALITY:
        dict[modality][stimuli_id] = data

def init_dict(dict, subjects, format, modality):
    if format == FORMAT_CODA:
        dict = {}
        for sub in subjects:
            dict[get_dict_key_for_subject(sub)] = {}
    elif format == FORMAT_FLAT:
        dict = {}
    elif format == FORMAT_WITH_MODALITY:
        dict[modality] = {}
    return dict

def prepare_output_files(subjects, exp_name, file_name, format=FORMAT_CODA, modality='language', \
    movie_name='friends-s07', zip_file=False):
    submission_predictions = init_dict({}, subjects, format, modality)
    pads = np.zeros((5,1000))

    for sub in subjects:
        boundary = get_boundary_from_fmri_for_movie_for_subject(sub, movie_name)
        predictions_file = utils.get_predictions_file_path(sub, movie_name)
        sub_predictions = np.load(predictions_file, allow_pickle=True)
        predictions_dict = {}
        from_idx = 0
        total_size =0
        num_stimuli =0
        for stim_id, size in boundary:
            num_stimuli +=1
            total_size += size
            effective_size = size-10
            slice = sub_predictions[from_idx:from_idx+effective_size,:]
            slice = np.concatenate((pads, slice, pads), axis=0)
            append_to_dict(submission_predictions, sub, stim_id, slice.astype(np.float32), format, modality)
            #if from_idx < 3000: print(from_idx, from_idx + size)
            from_idx = from_idx + effective_size
            assert slice.shape[0] == size, f"size mismatch while slicing {stim_id} {slice.shape[0]} {size}"
        assert total_size == (sub_predictions.shape[0] + num_stimuli*2*pads.shape[0]), f"total_size {total_size} != sub_predictions.shape[0] {sub_predictions.shape[0] + num_stimuli*2*pads.shape[0]}"
    print(submission_predictions.keys())
    # print(submission_predictions['sub-01'].keys())
    # print(submission_predictions['sub-01']['s07e01a'].shape)

    predictions_dir = os.path.join(utils.get_output_dir(), 'predictions', exp_name)
    file_name_w_ext = f"{file_name}.npy"
    output_file = os.path.join(predictions_dir, 'output', file_name_w_ext)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, submission_predictions)
    print(f"Formatted predictions saved to: {output_file}")

    # Zip the saved file for submission
    if zip_file or format == FORMAT_CODA:
        zip_file = os.path.join(predictions_dir, 'output', f"{file_name}.zip")
        with zipfile.ZipFile(zip_file, 'w') as zipf:
            zipf.write(output_file, os.path.basename(output_file))
        print(f"Submission file successfully zipped as: {zip_file}")

def get_output_file_name(subjects, exp_name, format):
    subj_prefix = "sub-all_"
    if len(subjects) == 1:
        subj_prefix = f"sub-{subjects[0]}_"
    main = "predictions"
    if format == FORMAT_FLAT:
        main = "prediction_as_features"
    elif format == FORMAT_WITH_MODALITY:
        main = "features"
    file_name = f"{subj_prefix}{main}_{exp_name}"
    return file_name

def run_for_coda():
    exp_name = utils.get_experiment_name()
    subjects = [1,2,3,5]
    movie_name = "friends-s07"
    format = FORMAT_CODA
    file_name = get_output_file_name(subjects, exp_name, format)
    prepare_output_files(subjects, exp_name, file_name, format, movie_name=movie_name)

def run_for_flat_output():
    
    subjects = [1,2,3,5]
    movies = ["friends-s02", "friends-s07"]
    exp_name = utils.get_experiment_name()
    format = FORMAT_FLAT
    for sub in subjects:
        file_name = get_output_file_name([sub], exp_name, format)
        for movie in movies:
            prepare_output_files([sub], exp_name, file_name, format, movie_name=movie)

def run_for_predictions_as_features():
    pass

if __name__ == "__main__":
    #run_for_coda()
    run_for_flat_output()