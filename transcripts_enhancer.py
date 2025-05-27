import json
import re
import os
import csv
import ast
import utils
import difflib
from rapidfuzz import process
from rapidfuzz.fuzz import ratio
def get_scene_dialogue(file_path):
    """
    Parses a text file containing scenes and dialogues and returns a JSON object.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        dict: JSON object containing scenes with dialogues
        
    Raises:
        AssertionError: If validation rules are violated
    """
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Remove empty lines, lines with 'Commercial Break', and strip whitespace
    valid_lines = [line.strip() for line in lines 
                   if line.strip() and 'Commercial Break' not in line and 'Closing Credits' not in line and 'OPENING TITLES' not in line and 'END' not in line]
    
    assert valid_lines, "No dialogues found"
    
    # Assert that the first valid line is a scene
    assert valid_lines[0].startswith('['), "First valid line cannot be a scene"
    
    scenes = []
    current_scene = None
    current_dialogue = None
    dialogue_id = 0
    scene_id = 0
    dialogue_line_count = 0
    
    for line in valid_lines:
        # Check if this is a scene line
        if line.startswith('['):
            # Assert that scene has closing bracket
            assert line.endswith(']'), f"Scene line missing closing bracket: {line}"
            
            # Save previous scene if exists
            if current_scene is not None:
                # Save any pending dialogue
                if current_dialogue is not None:
                    current_scene["dialogues"].append(current_dialogue)
                scenes.append(current_scene)
            
            # Start new scene
            scene_desc = line[1:-1]  # Remove brackets
            # Strip 'Scene:' from the beginning if present
            if scene_desc.startswith('Scene:'):
                scene_desc = scene_desc[6:].strip()
            current_scene = {
                "id": scene_id,
                "desc": scene_desc,
                "dialogues": []
            }
            scene_id += 1
            current_dialogue = None
            dialogue_line_count = 0
            
        else:
            # This should be a dialogue line
            assert current_scene is not None, "Dialogue found before any scene"
            
            # Check if this line starts a new dialogue (contains speaker:)
            speaker_match = re.match(r'^([^:]+):\s*(.*)$', line)
            
            if speaker_match:
                # Save previous dialogue if exists
                if current_dialogue is not None:
                    # Assert dialogue is not more than 3 lines
                    assert dialogue_line_count <= 3, f"Dialogue exceeds 3 lines: {current_dialogue}"
                    current_scene["dialogues"].append(current_dialogue)
                
                # Start new dialogue
                speaker = speaker_match.group(1).strip()
                text = speaker_match.group(2).strip()
                
                current_dialogue = {
                    "id": dialogue_id,
                    "text": text,
                    "speaker": speaker
                }
                dialogue_id += 1
                dialogue_line_count = 1
                
            else:
                # This is a continuation of the current dialogue
                if current_dialogue is not None:
                    dialogue_line_count += 1
                    # Assert dialogue is not more than 3 lines
                    assert dialogue_line_count <= 3, f"Dialogue exceeds 3 lines for speaker: {current_dialogue['speaker']}"
                    
                    # Add to existing dialogue text
                    if current_dialogue["text"]:
                        current_dialogue["text"] += " " + line
                    else:
                        current_dialogue["text"] = line
    
    # Save the last scene and dialogue
    if current_dialogue is not None:
        assert dialogue_line_count <= 3, f"Dialogue exceeds 3 lines: {current_dialogue}"
        current_scene["dialogues"].append(current_dialogue)
    
    if current_scene is not None:
        scenes.append(current_scene)
    
    return {"scenes": scenes}


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


def add_dialogues_to_transcript(transcript_data_orig, dialogues, flex_len):
    """
    Populates dialogue_per_tr field for each entry in transcript_data_orig by matching
    dialogue text with words in the transcript.
    
    Args:
        transcript_data_orig (list): Array of transcript objects from load_transcript_tsv
        dialogues (dict): Dialogue data from get_scene_dialogue with scenes and dialogues
        flex_len (int): Additional words to consider when matching dialogues
        
    Returns:
        list: Updated transcript_data with dialogue_per_tr field populated
        
    Raises:
        AssertionError: If validation rules are violated
        ValueError: If matching fails for dialogue start/end words
    """
    
    def normalize_and_clean_word(word):
        """Normalize word by converting to lowercase and keeping only alphanumeric characters"""
        new_word = ''.join(c.lower() for c in word if c.isalnum())
        # if new_word == 'cmon':
            #new_word = 'come'
        return new_word
    
    # Initialize dialogue_per_tr for all rows
    for row in transcript_data_orig:
        row['dialogue_per_tr'] = [-1] * len(row['words_per_tr'])
        # Assert that words_per_tr and dialogue_per_tr have same length
        assert len(row['words_per_tr']) == len(row['dialogue_per_tr']), \
            f"Length mismatch: words_per_tr={len(row['words_per_tr'])}, dialogue_per_tr={len(row['dialogue_per_tr'])}"
    
    # Flatten all dialogues from all scenes
    all_dialogues = []
    skipped_dialogues = []
    for scene in dialogues['scenes']:
        all_dialogues.extend(scene['dialogues'])
    
    # Track current position in transcript
    current_row = 0
    current_word_pos = 0
    
    total_dialogues = len(all_dialogues)
    last_matched_dialogue_id = None
    fuzzy_replace_length = -1
    
    for dialogue in all_dialogues:
        dialogue_id = dialogue['id']
        dialogue_text = dialogue['text']
        # Filter out text within parentheses
        dialogue_text = re.sub(r'\([^)]*\)', '', dialogue_text).strip()
        dialogue_words = dialogue_text.split()
        
        if not dialogue_words:
            continue  # Skip empty dialogues
        
        # Normalize dialogue words
        normalized_dialogue_words = [normalize_and_clean_word(word) for word in dialogue_words]
        # Filter out empty normalized words
        normalized_dialogue_words = [word for word in normalized_dialogue_words if word]
        
        if not normalized_dialogue_words:
            continue  # Skip if no valid words after normalization
        
        # Find words to search in (current position + flex_len)
        search_words = []
        search_positions = []  # Track (row, word_pos) for each search word
        
        # Collect words from current position onwards
        temp_row = current_row
        temp_word_pos = current_word_pos
        words_collected = 0
        target_words = len(normalized_dialogue_words) + flex_len
        
        # Track first and last word positions for debugging
        first_search_word_pos = None
        last_search_word_pos = None
        
        while words_collected < target_words and temp_row < len(transcript_data_orig):
            row_words = transcript_data_orig[temp_row]['words_per_tr']
            
            while temp_word_pos < len(row_words) and words_collected < target_words:
                search_words.append(row_words[temp_word_pos])
                search_positions.append((temp_row, temp_word_pos))
                
                # Track first word position
                if first_search_word_pos is None:
                    first_search_word_pos = (temp_row, temp_word_pos)
                
                # Always update last word position
                last_search_word_pos = (temp_row, temp_word_pos)
                
                temp_word_pos += 1
                words_collected += 1
            
            if temp_word_pos >= len(row_words):
                temp_row += 1
                temp_word_pos = 0
        
        # Print search word range information
        print(f"Dialogue {dialogue_id} of len {len(normalized_dialogue_words)}: '{dialogue_text}'")
        print(f"Search words of len {len(search_words)}: {search_words}")
        print(f"Search words span from row {first_search_word_pos[0]}, index {first_search_word_pos[1]} to row {last_search_word_pos[0]}, index {last_search_word_pos[1]}")
        print(f"Total search words collected: {len(search_words)}")
        if first_search_word_pos[0] != last_search_word_pos[0]:
            print(f"  -> Search words span across {last_search_word_pos[0] - first_search_word_pos[0] + 1} rows")
        else:
            print(f"  -> Search words are within single row {first_search_word_pos[0]}")
        
        if len(search_words) < len(normalized_dialogue_words):
            raise ValueError(f"Not enough words remaining to match dialogue {dialogue_id}: '{dialogue_text}'")
        
        # Normalize search words for matching
        normalized_search_words = [normalize_and_clean_word(word) for word in search_words]
        
        # Use SequenceMatcher to find best match
        matcher = difflib.SequenceMatcher(None, normalized_dialogue_words, normalized_search_words)
        first_word_search_pos = None
        second_word_search_pos = None
        third_word_search_pos = None
        last_word_search_pos = None
        second_last_word_search_pos = None
        third_last_word_search_pos = None
        last_dialogue_index = len(normalized_dialogue_words) - 1
        total_equal_words =0
        max_equal_words = 0
        start_max_equal_words_search_pos = None
        start_max_equal_words_index = None
        for opcode in matcher.get_opcodes():
            tag, i1, i2, j1, j2 = opcode
            if i1 ==0:
                if tag == 'equal':
                    first_word_search_pos = j1
                if tag == 'replace':
                    print(f'{tag} {normalized_dialogue_words[0]}: {" ".join(normalized_search_words[j1:j2])}')
                    score = ratio(normalized_dialogue_words[0], " ".join(normalized_search_words[j1:j2]))
                    print(f'score: {score}')
                    if score >= 50:
                        first_word_search_pos = j1
            if last_dialogue_index == i2 -1: #>= i1 and last_dialogue_index <= i2: 
                if tag == 'equal':
                    last_word_search_pos = j2-1
                if tag == 'replace':
                    print(f'{tag} {normalized_dialogue_words[i1:i2]}: {normalized_search_words[j1:j2]}')
                    if j2 - j1 <= 2:
                        score = ratio(normalized_dialogue_words[last_dialogue_index], " ".join(normalized_search_words[j1:j2]))
                        print(f'score: {score}')
                        if score >= 50:
                            last_word_search_pos = j2-1
            if tag == 'equal':
                in_seq = i2 - i1
                total_equal_words += in_seq
                if in_seq > max_equal_words:
                    max_equal_words = in_seq
                    start_max_equal_words_search_pos = j1
                    start_max_equal_words_index = i1

        ratio_equal_words = total_equal_words/len(normalized_dialogue_words)
        if(dialogue_id > 100):
            break
        

        #first_word_search_pos setting based on 2nd and 3rd words
        if first_word_search_pos is None:
            if ratio_equal_words > 0.7 and len(normalized_dialogue_words) > 3:
                #check if 2nd and 3rd words match
                second_word_search_pos = None
                third_word_search_pos = None
                for opcode in matcher.get_opcodes():
                    tag, i1, i2, j1, j2 = opcode
                    if tag == 'equal' and i1 == 1:
                        second_word_search_pos = j1
                    if tag == 'equal' and i1 <= 2 and i2 > 2:
                        gap = 2 - i1
                        third_word_search_pos = j1 + gap
                print(f'second_word_search_pos: {second_word_search_pos}, third_word_search_pos: {third_word_search_pos}')
                if second_word_search_pos is not None and third_word_search_pos is not None and third_word_search_pos == second_word_search_pos + 1:
                    first_word_search_pos = second_word_search_pos

        #set first word based on max seq match
        if first_word_search_pos is None and ratio_equal_words > 0.7 and len(normalized_dialogue_words) > 4:
            if max_equal_words > 3:
                print(f'max_equal_words: {max_equal_words}, start_max_equal_words_search_pos: {start_max_equal_words_search_pos}')
                first_word_search_pos = start_max_equal_words_search_pos

        # if end is not found, assume it is around ratio of matched words
        if first_word_search_pos is not None and last_word_search_pos is None:
            print('*******Used exception')
            if len(normalized_dialogue_words) > 2:
                last_word_search_pos = int(len(normalized_dialogue_words)*ratio_equal_words) - 2
            if len(normalized_dialogue_words) <= 2:
                last_word_search_pos = first_word_search_pos

        #if selected length is too long
        if first_word_search_pos is not None and last_word_search_pos is not None and last_word_search_pos - first_word_search_pos > len(normalized_dialogue_words):
            print(f'len of selection is too long: {last_word_search_pos - first_word_search_pos} > {len(normalized_dialogue_words)}')
            last_word_search_pos = int(len(normalized_dialogue_words)*ratio_equal_words) 

        # matches = matcher.get_matching_blocks()
        
        # # Compute first_word_search_pos and last_word_search_pos based on matches
        
        
        # # Find the first matching block that contains the first dialogue word (index 0)
        # for match in matches:
        #     print('match', match)
        #     dialogue_start, search_start, length = match
        #     if dialogue_start == 0 and length > 0:  # First word is at the start of this match
        #         first_word_search_pos = search_start
        #         break
        
        # #if first word was not found, then can we find the next 2 word
        
        # # Find the last matching block that contains the last dialogue word
        
        # for match in matches:
        #     dialogue_start, search_start, length = match
        #     dialogue_end = dialogue_start + length - 1
        #     if dialogue_end == last_dialogue_index and length > 0:  # Last word is at the end of this match
        #         last_word_search_pos = search_start + length - 1
        #         break
        
        # Check if both first and last words were found
        if first_word_search_pos is None:
            print(f"Could not find first word '{dialogue_words[0]}' (normalized: '{normalized_dialogue_words[0]}') of dialogue {dialogue_id}: '{dialogue_text}'")
            print(f"Search words used for matching: {search_words}")
            print(f"Normalized search words: {normalized_search_words}")
            print(f"Normalized dialogue words: {normalized_dialogue_words}")
            #raise ValueError(f"Could not find first word '{dialogue_words[0]}' of dialogue {dialogue_id}: '{dialogue_text}'")
        
        if last_word_search_pos is None:
            print(f"Could not find last word '{dialogue_words[-1]}' (normalized: '{normalized_dialogue_words[-1]}') of dialogue {dialogue_id}: '{dialogue_text}'")
            print(f"Search words used for matching: {search_words}")
            print(f"Normalized search words: {normalized_search_words}")
            print(f"Normalized dialogue words: {normalized_dialogue_words}")
            #raise ValueError(f"Could not find last word '{dialogue_words[-1]}' of dialogue {dialogue_id}: '{dialogue_text}'")
        
        if first_word_search_pos is None or last_word_search_pos is None:
            print('****skipping dialogue', dialogue_id)
            skipped_dialogues.append(dialogue)
        else:
            # Ensure first word comes before last word
            if first_word_search_pos > last_word_search_pos:
                raise ValueError(f"First word position ({first_word_search_pos}) comes after last word position ({last_word_search_pos}) for dialogue {dialogue_id}: '{dialogue_text}'")
            
            best_match = (first_word_search_pos, last_word_search_pos)
            
            # Print the actual row and index positions for the matched words
            search_start, search_end = best_match
            start_row, start_index = search_positions[search_start]
            end_row, end_index = search_positions[search_end]
            
            print(f"Best match found:")
            print(f"  First word '{dialogue_words[0]} {normalized_search_words[first_word_search_pos]} ' at row {start_row}, index {start_index}")
            print(f"  Last word '{dialogue_words[-1]} {normalized_search_words[last_word_search_pos]} ' at row {end_row}, index {end_index}")
            print(f"dialogue    : {dialogue_text}")
            search_string = " ".join(normalized_search_words[first_word_search_pos:last_word_search_pos])
            print(f"search words: {search_string}")
            if start_row != end_row:
                print(f"  -> Dialogue spans across {end_row - start_row + 1} rows")
            else:
                print(f"  -> Dialogue is within single row {start_row}")
            
            # Update dialogue_per_tr based on the match
            # Mark the first word of the dialogue with dialogue_id
            first_word_row, first_word_pos = search_positions[search_start]
            transcript_data_orig[first_word_row]['dialogue_per_tr'][first_word_pos] = dialogue_id
            
            # Update current position to after the last matched word
            current_row, current_word_pos = search_positions[search_end]
            current_word_pos += 1  # Move to next word
            
            # If we've reached the end of current row, move to next row
            if current_word_pos >= len(transcript_data_orig[current_row]['words_per_tr']):
                current_row += 1
                current_word_pos = 0
            
            last_matched_dialogue_id = dialogue_id
    
    # # Check if all transcript data has been processed
    # if current_row < len(transcript_data_orig):
    #     # Check if there are still words left in the current or subsequent rows
    #     remaining_words = 0
    #     for row_idx in range(current_row, len(transcript_data_orig)):
    #         if row_idx == current_row:
    #             remaining_words += len(transcript_data_orig[row_idx]['words_per_tr']) - current_word_pos
    #         else:
    #             remaining_words += len(transcript_data_orig[row_idx]['words_per_tr'])
        
    #     if remaining_words > 0:
    #         raise ValueError(f"Transcript data not fully processed. {remaining_words} words remaining after processing all dialogues.")
    
    # # Final assertion check for all rows
    # for row in transcript_data_orig:
    #     assert len(row['words_per_tr']) == len(row['dialogue_per_tr']), \
    #         f"Final length mismatch: words_per_tr={len(row['words_per_tr'])}, dialogue_per_tr={len(row['dialogue_per_tr'])}"
    
    print(f"Total number of dialogues processed: {total_dialogues}")
    print(f"Last matched dialogue ID: {last_matched_dialogue_id}")
    print(f"Total number of skipped dialogues: {len(skipped_dialogues)}")
    print(f"Skipped dialogues:")
    for dialogue in skipped_dialogues:
        print(f"  {dialogue['id']}: {dialogue['text']}")
    return transcript_data_orig


# Example usage and test
if __name__ == "__main__":

    # Write test file
    file_path = os.path.join(utils.get_data_root_dir(),'stimuli', 'transcripts', 'friends','s1','full','friends_s01e01.txt')
    
    # Test load dialogues file function
    try:
        result = get_scene_dialogue(file_path)
        #print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")

    file_path = os.path.join(utils.get_data_root_dir(),'stimuli', 'transcripts', 'friends','s1','friends_s01e01a.tsv')

    transcript_data_orig = load_transcript_tsv(file_path)
    # print(len(transcript_data_orig))
    # print(transcript_data_orig[16])
    # for row in transcript_data_orig:
    #     row['dialogue_per_tr'] = [4]
    
    transcript_data_enhanced = add_dialogues_to_transcript(transcript_data_orig, result, 20)
    file_path = os.path.join(utils.get_data_root_dir(),'stimuli', 'transcripts', 'friends','s1','enhanced','friends_s01e01a.tsv')
    save_transcript_with_dialogue_tsv(file_path, transcript_data_orig)
    # # Test save dialogues file function
    # file_path = os.path.join(utils.get_data_root_dir(),'stimuli', 'transcripts', 'friends','s1','friends_s01e01a.tsv')
    # save_transcript_with_dialogue_tsv(file_path, transcript_data_orig)


