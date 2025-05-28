# Example usage and test
from glob import glob
import utils
import csv
import ast
import transcripts_enhancer
import os

def load_transcript_tsv(file_path):
    """
    Loads a TSV file containing transcript data with specific fields.
    
    Args:
        file_path (str): Path to the TSV file
        
    Returns:
        list: Array of objects with text_per_tr, words_per_tr, onsets_per_tr, durations_per_tr fields
        
    Raises:
        AssertionError: If validation rules are violated
    """
    
    transcript_data = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        
        # Check that all required fields are in the header
        required_fields = ['text_per_tr', 'words_per_tr', 'onsets_per_tr', 'durations_per_tr']
        for field in required_fields:
            assert field in reader.fieldnames, f"Required field '{field}' not found in TSV header"
        
        for row_num, row in enumerate(reader, start=2):  # Start at 2 since header is line 1
            # Assert that each row has all 4 fields
            for field in required_fields:
                assert field in row, f"Row {row_num}: Missing required field '{field}'"
            
            try:
                # Parse words_per_tr using ast.literal_eval to handle mixed quotes
                words_str = row['words_per_tr'].strip()
                if words_str:
                    words_per_tr = ast.literal_eval(words_str)
                else:
                    words_per_tr = []
                
                # Parse onsets_per_tr - these should be in JSON format
                onsets_str = row['onsets_per_tr'].strip()
                if onsets_str:
                    onsets_list = ast.literal_eval(onsets_str)
                    onsets_per_tr = [float(x) for x in onsets_list]
                else:
                    onsets_per_tr = []
                
                # Parse durations_per_tr - these should be in JSON format  
                durations_str = row['durations_per_tr'].strip()
                if durations_str:
                    durations_list = ast.literal_eval(durations_str)
                    durations_per_tr = [float(x) for x in durations_list]
                else:
                    durations_per_tr = []
                
                transcript_row = {
                    'text_per_tr': row['text_per_tr'],  # Can be empty
                    'words_per_tr': words_per_tr,
                    'onsets_per_tr': onsets_per_tr,
                    'durations_per_tr': durations_per_tr
                }
                
                transcript_data.append(transcript_row)
                
            except (ValueError, SyntaxError) as e:
                print(f"Parsing error at row {row_num}: {e}")
                print(f"Problematic data:")
                print(f"  words_per_tr: '{row['words_per_tr']}'")
                print(f"  onsets_per_tr: '{row['onsets_per_tr']}'")
                print(f"  durations_per_tr: '{row['durations_per_tr']}'")
                raise
    
    return transcript_data

def save_transcript_with_dialogue_tsv(file_path, transcript_data):
    """
    Saves transcript data with dialog information to a TSV file.
    
    Args:
        file_path (str): Path where the TSV file will be saved
        transcript_data (list): Array of objects with 5 fields:
                               text_per_tr, words_per_tr, onsets_per_tr, durations_per_tr, dialogue_per_tr
    """
    
    fieldnames = ['text_per_tr', 'words_per_tr', 'onsets_per_tr', 'durations_per_tr', 'dialogue_per_tr']
    
    with open(file_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')
        
        # Write header
        writer.writeheader()
        
        # Write data rows
        for row_data in transcript_data:
            # Convert arrays to Python literal strings (maintains mixed quote format)
            tsv_row = {
                'text_per_tr': row_data['text_per_tr'],
                'words_per_tr': repr(row_data['words_per_tr']),
                'onsets_per_tr': repr(row_data['onsets_per_tr']),
                'durations_per_tr': repr(row_data['durations_per_tr']),
                'dialogue_per_tr': repr(row_data['dialogue_per_tr'])
            }
            writer.writerow(tsv_row)

def run_for_all_episodes():
    root_data_dir = utils.get_data_root_dir()
    
    #list of full text transcripts
    file_in_filter = ''
    exclude_list = []#['friends_s03e05b', 'friends_s03e06a']
    files = glob(f"{root_data_dir}/stimuli/transcripts/friends/s*/full/*.txt")
    if file_in_filter:
        files = [f for f in files if file_in_filter in f]
    files.sort()

    stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
    print(len(stimuli), list(stimuli)[:3], list(stimuli)[-3:])

    videos_iterator = enumerate(stimuli.items())
    for i, (stim_id, stim_path) in videos_iterator:
        print(f"Processing {stim_id}", stim_path)
        #load transcript files
        t_files = glob(f"{root_data_dir}/stimuli/transcripts/friends/s*/{stim_id}*.tsv")
        t_files.sort()
        f_stimuli = {f.split("/")[-1].split(".")[0]: f for f in t_files}
        print(len(f_stimuli), list(f_stimuli)[:3], list(f_stimuli)[-3:])
        #outfile_path
        season_id = 's' + stim_id[10]

        trans_iterator = enumerate(f_stimuli.items())
        trans_info_list = []
        transcript_data = None
        isfirst = True
        for j, (trans_id, trans_path) in trans_iterator:
            print(f"Handling {trans_id}", trans_path)
            tr_lines = load_transcript_tsv(trans_path)
            num_lines = len(tr_lines)
            if isfirst:
                transcript_data = tr_lines
                isfirst = False
            else:
                transcript_data.extend(tr_lines)
            trans_info = {"len": num_lines, "stimd_id": stim_id, "trans_id": trans_id, "trans_path": trans_path}
            trans_info_list.append(trans_info)
        total_tr_len = 0
        for trans_info in trans_info_list:
            total_tr_len += trans_info["len"]
        print(f"Total transcript length: {total_tr_len}")

        #enhance the transcripts
        #enhanced_transcript_data = transcript_data
        assert total_tr_len == len(transcript_data), f"Total transcript length {len(transcript_data)} does not match the number of transcript data {total_tr_len}"
        enhanced_transcript_data, _ = transcripts_enhancer.enhance_transcripts(transcript_data, stim_path, run_phase_2=True)
        assert len(enhanced_transcript_data) == total_tr_len, f"Enhanced transcript length {len(enhanced_transcript_data)} does not match the number of transcript data {total_tr_len}"

        #save the enhanced transcripts
        counter = 0
        for trans_info in trans_info_list: 
            outfile_path = os.path.join(utils.get_data_root_dir(),'stimuli', 'transcripts', 'friends',season_id,'enhanced',f'{trans_info["trans_id"]}.tsv')
            print(f"Saving to {outfile_path}")
            file_enhanced_transcript_data = enhanced_transcript_data[counter:counter+trans_info["len"]]
            assert len(file_enhanced_transcript_data) == trans_info["len"], f"File enhanced transcript length {len(file_enhanced_transcript_data)} does not match the number of transcript data {trans_info['len']}"
            save_transcript_with_dialogue_tsv(outfile_path, file_enhanced_transcript_data)
            counter += trans_info["len"]

def test_with_1_episode():
    root_data_dir = utils.get_data_root_dir()
    file_path = os.path.join(root_data_dir, 'stimuli', 'transcripts', 'friends', 's1', 'backup', 'friends_s01e01a-combined.tsv')
    transcript_data = load_transcript_tsv(file_path)
    stim_path = os.path.join(root_data_dir, 'stimuli', 'transcripts', 'friends', 's1', 'full', 'friends_s01e01.txt')
    enhanced_transcript_data, _ = transcripts_enhancer.enhance_transcripts(transcript_data, stim_path, run_phase_2=False)
    #print(transcript_data)


if __name__ == "__main__":
    #run_for_all_episodes()
    test_with_1_episode()