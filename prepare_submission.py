from train import get_boundary_from_fmri_for_movie_for_subject
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

def append_to_dict(dict, subject, stimuli_id, data, format, modality):
    if format == FORMAT_CODA:
        dict[get_dict_key_for_subject(subject)][stimuli_id] = data
    elif format == FORMAT_FLAT:
        dict[stimuli_id] = data
    elif format == FORMAT_WITH_MODALITY:
        dict[modality][stimuli_id] = data

def get_output_file_path(file_name, extension, exp_name):
    predictions_dir = os.path.join(utils.get_output_dir(), 'predictions', exp_name)
    file_name_w_ext = f"{file_name}.{extension}"
    output_file = os.path.join(predictions_dir, 'output', file_name_w_ext)
    return output_file

def init_dict(dict, subjects, format, modality, file_name, exp_name):
    if format == FORMAT_CODA:
        dict = {}
        for sub in subjects:
            dict[get_dict_key_for_subject(sub)] = {}
    elif format == FORMAT_FLAT:
        #if file exists, load dict from file and return it
        output_file = get_output_file_path(file_name, 'npy', exp_name)
        if os.path.exists(output_file):
            dict = np.load(output_file, allow_pickle=True).item()
        else:
            dict = {}
        
    elif format == FORMAT_WITH_MODALITY:
        #if file exists, load dict from file and return it
        output_file = get_output_file_path(file_name, 'npy', exp_name)
        if os.path.exists(output_file):
            dict = np.load(output_file, allow_pickle=True).item()
        else:
            dict[modality] = {}    
    return dict

def prepare_output_files(subjects, exp_name, file_name, format=FORMAT_CODA, modality=None, \
    movie_name='friends-s07', zip_file=False):
    submission_predictions = init_dict({}, subjects, format, modality, file_name, exp_name)
    pads = np.zeros((5,1000))
    add_pads = False
    compare_buffer = 10
    if format == FORMAT_CODA:
        add_pads = True
        compare_buffer = 0

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
            if add_pads:
                slice = np.concatenate((pads, slice, pads), axis=0)
            append_to_dict(submission_predictions, sub, stim_id, slice.astype(np.float32), format, modality)
            #if from_idx < 3000: print(from_idx, from_idx + size)
            from_idx = from_idx + effective_size
            assert (slice.shape[0] + compare_buffer)== size, f"size mismatch while slicing {stim_id} {slice.shape[0]} {size}"
        assert total_size == (sub_predictions.shape[0] + num_stimuli*2*pads.shape[0]), f"total_size {total_size} != sub_predictions.shape[0] {sub_predictions.shape[0] + num_stimuli*2*pads.shape[0]}"
    # print(submission_predictions.keys())
    # print(submission_predictions['language'].keys())
    # print(submission_predictions['sub-01']['s07e01a'].shape)

    output_file = get_output_file_path(file_name, 'npy', exp_name)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, submission_predictions)
    print(f"Formatted predictions saved to: {output_file}")

    # Zip the saved file for submission
    if zip_file or format == FORMAT_CODA:
        zip_file = get_output_file_path(file_name, 'zip', exp_name)
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

def run_for_flat_output(modality=None):
    
    subjects = [1] # [1,2,3,5]
    movies = ["friends-s01","friends-s02", "friends-s03", "friends-s04", "friends-s05", "friends-s06"]
    exp_name = utils.get_experiment_name()
    format = FORMAT_FLAT
    if modality is not None:
        format = FORMAT_WITH_MODALITY
    
    for sub in subjects:
        file_name = get_output_file_name([sub], exp_name, format)
        output_file = get_output_file_path(file_name, 'npy', exp_name)
        #if file exists, delete it
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"##### Deleted {output_file}")
        for movie in movies:
            prepare_output_files([sub], exp_name, file_name, format, modality=modality, movie_name=movie)


if __name__ == "__main__":
    #run_for_coda()
    run_for_flat_output()