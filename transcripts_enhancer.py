import json
import re
import os
import csv
import ast
import utils
import difflib
from rapidfuzz import process, fuzz
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
    assert valid_lines[0].startswith('['), "First valid line is not a scene"
    
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
                    assert dialogue_line_count <= 3, f"Dialogue exceeds 3 lines for speaker: {current_dialogue['speaker']} {current_dialogue['text']}"
                    
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




def word_substitutions_to_make(segment):
    segment = re.sub(r'whaddya', 'what do you', segment.lower())
    segment = re.sub(r'goodnight', 'good night', segment.lower())
    segment = re.sub(r'waitwait', 'wait wait', segment.lower())
    return segment
def get_dialogue_list(dialogues):
    dialogue_list = []
    dialogue_temp = []
    for scene in dialogues['scenes']:
        dialogue_temp.extend(scene['dialogues'])
    for dialogue in dialogue_temp:
        norm_text = re.sub(r'\([^)]*\)', '', dialogue['text']).strip()
        norm_text = re.sub(r'\[[^\]]*\]', '', norm_text).strip()
        norm_text = re.sub(r'\.{4,}', ' ', norm_text)
        norm_text = word_substitutions_to_make(norm_text)
        dialogue_words = norm_text.split()
        normalized_dialogue_words = [normalize_and_clean_word(word) for word in dialogue_words]
        # Filter out empty normalized words
        dialogue['normalized_text']  = [word for word in normalized_dialogue_words if word]
        dialogue_list.append(dialogue)
    return dialogue_list

def normalize_and_clean_word(word):
        """Normalize word by converting to lowercase and keeping only alphanumeric characters"""
        new_word = ''.join(c.lower() for c in word if c.isalnum())
        # if new_word == 'cmon':
            #new_word = 'come'
        return new_word

MAX_DIALOGUE_ID = -5
def get_row_word_index_for_dialogue(transcript_data, dialogue_id):
    if dialogue_id == MAX_DIALOGUE_ID:
        return len(transcript_data)-1, len(transcript_data[len(transcript_data)-1]['words_per_tr'])
    for row_idx, row in enumerate(transcript_data):
        #print(row_idx, row['dialogue_per_tr'])
        for word_idx, word in enumerate(row['dialogue_per_tr']):
            if word == dialogue_id:
                return row_idx, word_idx
    return None, None

def get_empty_trs_in_range(transcript_data, from_row, to_row):
    empty_trs = []
    for row_idx in range(from_row, to_row):
        if len(transcript_data[row_idx]['words_per_tr']) == 0:
            empty_trs.append(row_idx)
    return empty_trs

def get_ordered_row_word_index_for_dialogue(transcriptdata, dialogues, dialogue_id, isNext=True):
    if isNext:
        found_ordered = False
        num_check = 1
        row_idx, word_idx = None, None
        #print(f'dialogue_id: {dialogue_id}, num_check: {num_check}, len(dialogues): {len(dialogues)}')
        while not found_ordered and num_check < 4 and dialogue_id+num_check < len(dialogues):
            row_idx, word_idx = get_row_word_index_for_dialogue(transcriptdata, dialogue_id+num_check)
            if row_idx is not None:
                found_ordered = True
            else:
                num_check += 1

        return row_idx, word_idx, dialogue_id+num_check
    else:
        found_ordered = False
        num_check = 1
        row_idx, word_idx = None, None
        while not found_ordered and num_check < 4 and dialogue_id-num_check > 0:
            row_idx, word_idx = get_row_word_index_for_dialogue(transcriptdata, dialogue_id-num_check)
            if row_idx is not None:
                found_ordered = True
            else:
                num_check += 1
        return row_idx, word_idx, dialogue_id-num_check
def fill_empty_tr_with_dialogue(transcript_data, dialogue_id, row_idx):
    transcript_data[row_idx]['dialogue_per_tr'] = [dialogue_id]
    transcript_data[row_idx]['words_per_tr'] = ['PLACEHOLDER']
    transcript_data[row_idx]['onsets_per_tr'] = []
    transcript_data[row_idx]['durations_per_tr'] = []
    transcript_data[row_idx]['text_per_tr'] = 'PLACEHOLDER'

def set_dialogue_in_transcript(transcript_data,  dialogue_id, row_idx, word_idx, move_to_next_if_taken=True):
    able_to_set = True
    print('set_dialogue_in_transcript',transcript_data[row_idx]['dialogue_per_tr'])
    if (transcript_data[row_idx]['dialogue_per_tr'][word_idx] == -1):
        transcript_data[row_idx]['dialogue_per_tr'][word_idx] = dialogue_id
    elif move_to_next_if_taken:
        set_row_idx, set_word_idx = row_idx, word_idx
        len_row = len(transcript_data[row_idx]['words_per_tr'])
        if word_idx < len_row - 1:
            set_word_idx = word_idx + 1
        elif row_idx < len(transcript_data) - 1:
            set_row_idx = row_idx + 1
            set_word_idx = 0
        else:
            able_to_set = False
            return transcript_data, able_to_set
        
        if len(transcript_data[set_row_idx]['dialogue_per_tr']) < 1:
            fill_empty_tr_with_dialogue(transcript_data, dialogue_id, set_row_idx)
            #create new entry
        elif transcript_data[set_row_idx]['dialogue_per_tr'][set_word_idx] != -1:
            able_to_set = False
            return transcript_data, able_to_set
        else:
            transcript_data[set_row_idx]['dialogue_per_tr'][set_word_idx] = dialogue_id
    return transcript_data, able_to_set
#from(inclusive) and to(non inclusive)
def try_squeeze_in_dialogue(transcript_data, dialogues, dialogue_id):
    fixed_dialogue = False
    dialogues_list = get_dialogue_list(dialogues)
    normalized_dialogue_words = dialogues_list[dialogue_id]['normalized_text']
    prev_diaglogue_length = 0
    if dialogue_id > 0:
        from_row, from_word_pos, prev_dialogue_id = get_ordered_row_word_index_for_dialogue(transcript_data, dialogues_list, dialogue_id, isNext=False)
        if from_row is None:
            print(f'Could not find previous dialogue {dialogue_id}')
            return transcript_data, fixed_dialogue
        prev_dialogue_length = len(dialogues_list[prev_dialogue_id]['text'].split())
    else:
        from_row, from_word_pos = (0, 0)
    to_row, to_word_pos, next_dialogue_id = get_ordered_row_word_index_for_dialogue(transcript_data, dialogues_list, dialogue_id, isNext=True)
    if to_row is None:
        print(f'Could not find next dialogue {dialogue_id}')
        return transcript_data, fixed_dialogue

    temp_row = from_row
    temp_word_pos = from_word_pos
    search_words = []
    search_positions = []
    first_search_word_pos = None
    last_search_word_pos = None
    temp_word_pos = from_word_pos
    words_collected = 0
    while temp_row <= to_row:
        row_words = transcript_data[temp_row]['words_per_tr']
        max_row_words = len(row_words)
        if temp_row == to_row:
            max_row_words = to_word_pos
        while temp_word_pos < max_row_words:
            search_words.append(normalize_and_clean_word(row_words[temp_word_pos]))
            search_positions.append((temp_row, temp_word_pos))
            
            # Track first word position
            if first_search_word_pos is None:
                first_search_word_pos = (temp_row, temp_word_pos)
            
            # Always update last word position
            last_search_word_pos = (temp_row, temp_word_pos)
            
            temp_word_pos += 1
            words_collected += 1
            
        temp_row += 1
        temp_word_pos = 0
    #strategy 1: If 1-2 empty row, then squeeze in the dialogue
    empty_trs = get_empty_trs_in_range(transcript_data, from_row, to_row)
    if len(empty_trs) < 5 and len(empty_trs) > 0:
        fixed_dialogue = True
        fill_empty_tr_with_dialogue(transcript_data, dialogue_id, empty_trs[0])
    print('search_words', search_words)
    #strategy 2: Compute matching ration
    num_equal =0
    first_word_pos = None
    
    for word in normalized_dialogue_words:
        word_search_counter =0
        for word_s in search_words:
            if word == word_s:
                num_equal += 1
                if first_word_pos is None:
                    first_word_pos = search_positions[word_search_counter]
            word_search_counter += 1
    print(f'first_word_pos: {first_word_pos}')
    ratio_equal_words = num_equal/len(normalized_dialogue_words)
    print(f'ratio_equal_words: {ratio_equal_words}')
    if ratio_equal_words >= 0.5:
        transcript_data, fixed_dialogue = set_dialogue_in_transcript(transcript_data, dialogue_id, first_word_pos[0], first_word_pos[1])

    
        

    print('empty_trs', empty_trs)
    gap_available = words_collected - prev_dialogue_length
    print('search_words', search_words)
    print('normalized_dialogue_words', normalized_dialogue_words)
    # print(f'gap_available: {gap_available}')
    return transcript_data, fixed_dialogue

def best_variable_fuzzy_match(short_words, long_words, min_window=None, max_window=None):
    """
    Finds the best fuzzy match of short_words (joined as a string) inside long_words
    over all possible window sizes.
    Returns start and end word indices (inclusive), matched words, and score.
    """
    short_str = " ".join(short_words)
    if min_window is None:
        min_window = len(short_words) // 2
    n = len(long_words)
    if max_window is None:
        max_window = min(len(short_words)*2, len(long_words))
    best_score = 0
    best_start = 0
    best_end = 0
    best_matched_words = []

    for window_size in range(min_window, max_window + 1):
        for i in range(n - window_size + 1):
            window_words = long_words[i:i+window_size]
            window_str = " ".join(window_words)
            score = fuzz.ratio(short_str, window_str)
            if score > best_score:
                best_score = score
                best_start = i
                best_end = i + window_size - 1  # inclusive
                best_matched_words = window_words

    return {
        'match_score': best_score,
        'start_word': best_start,
        'end_word': best_end,
        'matched_words': best_matched_words,
        'matched_text': ' '.join(best_matched_words)
    }


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
    
    
    
    # Initialize dialogue_per_tr for all rows
    for row in transcript_data_orig:
        row['dialogue_per_tr'] = [-1] * len(row['words_per_tr'])
        # Assert that words_per_tr and dialogue_per_tr have same length
        assert len(row['words_per_tr']) == len(row['dialogue_per_tr']), \
            f"Length mismatch: words_per_tr={len(row['words_per_tr'])}, dialogue_per_tr={len(row['dialogue_per_tr'])}"
    
    # Flatten all dialogues from all scenes
    all_dialogues = get_dialogue_list(dialogues)
    skipped_dialogues = []
    
    
    # Track current position in transcript
    current_row = 0
    current_word_pos = 0

    words_added_total = 0
    skipped_words_total = 0
    num_good_fuzz_score = 0
    
    total_dialogues = len(all_dialogues)
    last_matched_dialogue_id = None

    active_flex_len = flex_len
    consecutive_skipped_dialogues = 0
    
    for dialogue in all_dialogues:
        dialogue_id = dialogue['id']
        dialogue_text = dialogue['text']
        normalized_dialogue_words = dialogue['normalized_text']
        
        if not normalized_dialogue_words:
            continue  # Skip if no valid words after normalization
        
        # Find words to search in (current position + flex_len)
        search_words = []
        search_positions = []  # Track (row, word_pos) for each search word
        
        # Collect words from current position onwards
        temp_row = current_row
        temp_word_pos = current_word_pos
        words_collected = 0
        best_match_fuzz = None

        temp_flex_len = flex_len
        skip_multiplier = 5
        #if large sentece make flex longer
        if len(normalized_dialogue_words) > 4:
            temp_flex_len = 20

        if len(normalized_dialogue_words) > 5 and consecutive_skipped_dialogues > 0:
            temp_flex_len = 50

        if consecutive_skipped_dialogues < 2:
            skip_multiplier = 5
        elif consecutive_skipped_dialogues < 5:
            skip_multiplier = 10
        elif consecutive_skipped_dialogues < 10:
            skip_multiplier = 20
        else:
            skip_multiplier = 30

        active_flex_len = temp_flex_len + skip_multiplier*consecutive_skipped_dialogues
        target_words = len(normalized_dialogue_words) + active_flex_len
        print(f'target_words len: {target_words}')
        
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

        
        if(len(search_words) < 10 and len(search_words) < len(normalized_dialogue_words)):
            print(f'Not enough words remaining ({len(search_words)} < {len(normalized_dialogue_words)}) to match dialogue {dialogue_id}: "{dialogue_text}"')
            break;
            

        # Print search word range information
        print(f"Dialogue {dialogue_id} of len {len(normalized_dialogue_words)}: '{dialogue_text}'")
        print(f"Search words of len {len(search_words)}: {search_words}")
        
        # if len(search_words) < len(normalized_dialogue_words):
        #     raise ValueError(f"Not enough words remaining to match dialogue {dialogue_id}: '{dialogue_text}'")
        
        # Normalize search words for matching
        normalized_search_words = [normalize_and_clean_word(word) for word in search_words]
        use_fuzzy_matching = False
        #use fuzzy matching to find best match
        if len(normalized_dialogue_words) > 1:
            best_match_fuzz = best_variable_fuzzy_match(normalized_dialogue_words, normalized_search_words)
            print(f'best_match: {best_match_fuzz}')
            if (len(normalized_dialogue_words) > 4 and best_match_fuzz['match_score'] > 60) or \
            (best_match_fuzz['match_score'] >= 70):
                print(f'using fuzzy matching: {best_match_fuzz["match_score"]}')
                use_fuzzy_matching = True
        
        # Use SequenceMatcher to find best match
        matcher = difflib.SequenceMatcher(None, normalized_dialogue_words, normalized_search_words)
        first_word_search_pos = None
        second_word_search_pos = None
        third_word_search_pos = None
        last_word_search_pos = None
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
        print(f'ratio_equal_words: {ratio_equal_words}, max_equal_words: {max_equal_words}, first_word_search_pos: {first_word_search_pos}, last_word_search_pos: {last_word_search_pos}')
        

        #first_word_search_pos setting based on 2nd and 3rd words
        if first_word_search_pos is None:
            if ratio_equal_words >= 0.6 and len(normalized_dialogue_words) > 2:
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
                    if not use_fuzzy_matching:
                        print('found a match with second and third words and fuzzy did not find it')
                    first_word_search_pos = second_word_search_pos

        #set first word based on max seq match
        if first_word_search_pos is None and ((ratio_equal_words >= 0.6 and len(normalized_dialogue_words) > 4) or max_equal_words > 5):
            if max_equal_words > 3:
                if not use_fuzzy_matching:
                    print('found a match with max equal words and fuzzy did not find it')
                print(f'max_equal_words: {max_equal_words}, start_max_equal_words_index: {start_max_equal_words_index}, start_max_equal_words_search_pos: {start_max_equal_words_search_pos}')
                first_word_search_pos = max(start_max_equal_words_search_pos - start_max_equal_words_index, 0)

        # if end is not found, assume it is around ratio of matched words
        if first_word_search_pos is not None and last_word_search_pos is None and (ratio_equal_words >= 0.6 or max_equal_words > 5):
            print(f'*******Used exception {first_word_search_pos} {ratio_equal_words}')
            if not use_fuzzy_matching:
                print('found a match with exception of equal words and fuzzy did not find it')
            if len(normalized_dialogue_words) > 2:
                last_word_search_pos = first_word_search_pos + int(len(normalized_dialogue_words)*ratio_equal_words) - 2
            if len(normalized_dialogue_words) <= 2:
                last_word_search_pos = first_word_search_pos

        #if selected length is too long
        if first_word_search_pos is not None and last_word_search_pos is not None and last_word_search_pos - first_word_search_pos > len(normalized_dialogue_words):
            #if much bigger, then something went wrong
            if last_word_search_pos - first_word_search_pos > len(normalized_dialogue_words)*1.2 and max_equal_words < 0.5*len(normalized_dialogue_words):
                print(f'len of selection is too long: {last_word_search_pos - first_word_search_pos} > {len(normalized_dialogue_words)} and max_equal_words < {0.5*len(normalized_dialogue_words)}')
                first_word_search_pos = None
                last_word_search_pos = None
            else:
                #if little bigger, set last word
                print(f'len of selection is too long: {last_word_search_pos - first_word_search_pos} > {len(normalized_dialogue_words)}')
                last_word_search_pos = first_word_search_pos + int(len(normalized_dialogue_words)*ratio_equal_words) 
        
        if use_fuzzy_matching:
            if first_word_search_pos is not None and last_word_search_pos is not None and first_word_search_pos != best_match_fuzz['start_word'] and last_word_search_pos != best_match_fuzz['end_word']:
                print(f'mismatch of start end {first_word_search_pos} {last_word_search_pos} {best_match_fuzz["start_word"]} {best_match_fuzz["end_word"]}')
                search_string = " ".join(normalized_search_words[first_word_search_pos:last_word_search_pos+1])
                print('phase1 words:',search_string)
            if first_word_search_pos is None or last_word_search_pos is None:
                print(f'fuzzy is smart')
            first_word_search_pos = best_match_fuzz['start_word']
            last_word_search_pos = best_match_fuzz['end_word']
            search_string = " ".join(normalized_search_words[first_word_search_pos:last_word_search_pos+1])
            print('fuzzy words:',search_string)
            print('dialogue:',dialogue_text)
        
        # # Check if both first and last words were found
        # if first_word_search_pos is None:
        #     print(f"Could not find first word '{normalized_dialogue_words[0]}' of dialogue {dialogue_id}: '{dialogue_text}'")
        #     print(f"Search words used for matching: {search_words}")
        #     print(f"Normalized search words: {normalized_search_words}")
        #     print(f"Normalized dialogue words: {normalized_dialogue_words}")
        #     #raise ValueError(f"Could not find first word '{dialogue_words[0]}' of dialogue {dialogue_id}: '{dialogue_text}'")
        
        # if last_word_search_pos is None:
        #     print(f"Could not find last word '{normalized_dialogue_words[-1]}'  of dialogue {dialogue_id}: '{dialogue_text}'")
        #     print(f"Search words used for matching: {search_words}")
        #     print(f"Normalized search words: {normalized_search_words}")
        #     print(f"Normalized dialogue words: {normalized_dialogue_words}")
        #     #raise ValueError(f"Could not find last word '{dialogue_words[-1]}' of dialogue {dialogue_id}: '{dialogue_text}'")
        
        if first_word_search_pos is None or last_word_search_pos is None:
            print('****skipping dialogue', dialogue_id)
            skipped_dialogues.append(dialogue)
            consecutive_skipped_dialogues += 1
            if best_match_fuzz is not None and best_match_fuzz['match_score'] > 60:
                num_good_fuzz_score += 1
                print(f'********************************************************************************best_match_fuzz:')
                print(best_match_fuzz['match_score'])
                print('best match:',best_match_fuzz['matched_text'])
                print('dialogue:',dialogue_text)
        else:
            if best_match_fuzz is not None and best_match_fuzz['match_score'] < 60:
                print(f'********************************************************************************bad_match_fuzz:')
                print(best_match_fuzz['match_score'])
                print('best match:',best_match_fuzz['matched_text'])
                search_string = " ".join(normalized_search_words[first_word_search_pos:last_word_search_pos+1])
                print('search words:',search_string)
                print('dialogue:',dialogue_text)
            # Ensure first word comes before last word
            if first_word_search_pos > last_word_search_pos:
                raise ValueError(f"First word position ({first_word_search_pos}) comes after last word position ({last_word_search_pos}) for dialogue {dialogue_id}: '{dialogue_text}'")
            
            best_match = (first_word_search_pos, last_word_search_pos)
            consecutive_skipped_dialogues = 0
            # Print the actual row and index positions for the matched words/152/15
            search_start, search_end = best_match
            start_row, start_index = search_positions[search_start]
            end_row, end_index = search_positions[search_end]
            
            #compute words wasted and used
            words_added_total += last_word_search_pos - first_word_search_pos 
            skipped_words_total += first_word_search_pos 

            print(f"Best match found:")
            print(f"  First word '{normalized_dialogue_words[0]} {normalized_search_words[first_word_search_pos]} ' at row {start_row}, index {start_index}")
            print(f"  Last word '{normalized_dialogue_words[-1]} {normalized_search_words[last_word_search_pos]} ' at row {end_row}, index {end_index}")
            print(f"dialogue    : {dialogue_text}")
            search_string = " ".join(normalized_search_words[first_word_search_pos:last_word_search_pos+1])
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
        
        # if dialogue_id > 200:
        #     break
    
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
    
    # Final assertion check for all rows
    for row in transcript_data_orig:
        assert len(row['words_per_tr']) == len(row['dialogue_per_tr']), \
            f"Final length mismatch: words_per_tr={len(row['words_per_tr'])}, dialogue_per_tr={len(row['dialogue_per_tr'])}"
    
    print(f"Total number of dialogues processed: {total_dialogues}")
    print(f"Last matched dialogue ID: {last_matched_dialogue_id}")
    print(f"Total number of skipped dialogues: {len(skipped_dialogues)}")
    print(f"Total number of words added: {words_added_total}")
    print(f"Total number of words skipped: {skipped_words_total}")
    print(f"Total number of good fuzz scores: {num_good_fuzz_score}")
    return transcript_data_orig, skipped_dialogues

def enhance_transcripts(transcript_data, dialogues_file, run_phase_2=True):
    dialogues = get_scene_dialogue(dialogues_file)
    still_skipped_dialogues = []
    transcript_data_enhanced, skipped_dialogues = add_dialogues_to_transcript(transcript_data, dialogues, 20)
    print(f"Skipped dialogues after phase 1:")
    for dialogue in skipped_dialogues:
        print(f"  {dialogue['id']}: {dialogue['text']}")

    if run_phase_2:
        print(f"Phase 2:")
        counter = 0
        for dialogue in skipped_dialogues:
            counter += 1
            # if dialogue['id'] > 228:
            #     break
            transcript_data_enhanced, fixed_dialogue = try_squeeze_in_dialogue(transcript_data_enhanced, dialogues, dialogue['id'])
            if not fixed_dialogue:
                still_skipped_dialogues.append(dialogue)
                print(f"  {dialogue['id']}: {dialogue['text']} (not fixed)")
            else:
                print(f"  {dialogue['id']}: {dialogue['text']} (fixed)")
        print(f"Still skipped dialogues: {len(still_skipped_dialogues)}")

    return transcript_data_enhanced, still_skipped_dialogues