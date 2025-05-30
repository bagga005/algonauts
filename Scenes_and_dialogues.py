import re


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
    
    def is_scene_line(line):
        """Check if a line starts with [scene (case-insensitive, ignoring spaces)"""
        if not line.startswith('['):
            return False
        
        # Extract content after the opening bracket
        content_after_bracket = line[1:]
        
        # Remove leading spaces and convert to lowercase
        content_normalized = content_after_bracket.lstrip().lower()
        
        # Check if it starts with 'scene'
        return content_normalized.startswith('scene')
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Remove empty lines, lines with 'Commercial Break', and strip whitespace
    valid_lines = [line.strip() for line in lines 
                   if line.strip() and 'Commercial Break' not in line and 'Closing Credits' not in line and 'OPENING TITLES' not in line and 'END' not in line and 'Opening Credits' not in line]
    
    assert valid_lines, "No dialogues found"
    
    # Assert that the first valid line is a scene
    assert is_scene_line(valid_lines[0]), "First valid line is not a scene"
    
    scenes = []
    current_scene = None
    current_dialogue = None
    dialogue_id = 1
    scene_id = 1
    dialogue_line_count = 0
    in_scene_index = 1  # Track dialogue order within each scene
    
    for line in valid_lines:
        # Check if this is a scene line
        if is_scene_line(line):
            # Assert that scene has closing bracket
            assert line.endswith(']'), f"Scene line missing closing bracket: {line}"
            
            # Save previous scene if exists
            if current_scene is not None:
                # Save any pending dialogue
                if current_dialogue is not None:
                    current_scene["dialogues"].append(current_dialogue)
                
                # Add total_dialogues_scene to each dialogue in the scene
                total_dialogues_in_scene = len(current_scene["dialogues"])
                for dialogue in current_scene["dialogues"]:
                    dialogue["total_dialogues_scene"] = total_dialogues_in_scene
                
                scenes.append(current_scene)
            
            # Start new scene
            scene_desc = line[1:-1]  # Remove brackets
            # Strip 'Scene:' from the beginning if present (case-insensitive)
            scene_desc_lower = scene_desc.lower().lstrip()
            if scene_desc_lower.startswith('scene:'):
                # Find the position of ':' and remove everything up to and including it
                colon_pos = scene_desc.lower().find(':')
                scene_desc = scene_desc[colon_pos + 1:].strip()
            elif scene_desc_lower.startswith('scene '):
                # Remove 'scene ' from the beginning
                scene_desc = scene_desc[6:].strip()
            elif scene_desc_lower == 'scene':
                # If it's just 'scene', make it empty
                scene_desc = ''
            
            current_scene = {
                "id": scene_id,
                "desc": scene_desc,
                "dialogues": []
            }
            scene_id += 1
            current_dialogue = None
            dialogue_line_count = 0
            in_scene_index = 1  # Reset dialogue index for new scene
            
        else:
            # This should be a dialogue line
            assert current_scene is not None, "Dialogue found before any scene"
            
            # Check if this line starts a new dialogue (contains speaker:)
            speaker_match = re.match(r'^([^:]+):\s*(.*)$', line)
            
            if speaker_match:
                # Save previous dialogue if exists
                if current_dialogue is not None:
                    # Assert dialogue is not more than 3 lines
                    assert dialogue_line_count <= 5, f"Dialogue exceeds 5 lines: {current_dialogue}"
                    current_scene["dialogues"].append(current_dialogue)
                
                # Start new dialogue
                speaker = speaker_match.group(1).strip()
                speaker = speaker.title()  # Convert to title case (first letter of each word capitalized)
                text = speaker_match.group(2).strip()
                
                current_dialogue = {
                    "id": dialogue_id,
                    "text": text,
                    "speaker": speaker,
                    "in_scene_index": in_scene_index
                }
                dialogue_id += 1
                dialogue_line_count = 1
                in_scene_index += 1  # Increment for next dialogue in the scene
                
            else:
                # This is a continuation of the current dialogue
                if current_dialogue is not None:
                    dialogue_line_count += 1
                    # Assert dialogue is not more than 3 lines
                    assert dialogue_line_count <= 5, f"Dialogue exceeds 5 lines for speaker: {current_dialogue['speaker']} {current_dialogue['text']}"
                    
                    # Add to existing dialogue text
                    if current_dialogue["text"]:
                        current_dialogue["text"] += " " + line
                    else:
                        current_dialogue["text"] = line
    
    # Save the last scene and dialogue
    if current_dialogue is not None:
        assert dialogue_line_count <= 5, f"Dialogue exceeds 5 lines: {current_dialogue}"
        current_scene["dialogues"].append(current_dialogue)
    
    if current_scene is not None:
        # Add total_dialogues_scene to each dialogue in the last scene
        total_dialogues_in_scene = len(current_scene["dialogues"])
        for dialogue in current_scene["dialogues"]:
            dialogue["total_dialogues_scene"] = total_dialogues_in_scene
        
        scenes.append(current_scene)
    
    return {"scenes": scenes}

def normalize_and_clean_word(word):
        """Normalize word by converting to lowercase and keeping only alphanumeric characters"""
        new_word = ''.join(c.lower() for c in word if c.isalnum())
        # if new_word == 'cmon':
            #new_word = 'come'
        return new_word

def word_substitutions_to_make(segment):
    segment = re.sub(r'whaddya', 'what do you', segment.lower())
    segment = re.sub(r'goodnight', 'good night', segment.lower())
    segment = re.sub(r'waitwait', 'wait wait', segment.lower())
    return segment



def get_dialogue_list(dialogues):
    dialogue_list = []
    for scene in dialogues['scenes']:
        scene_id = scene['id']
        for dialogue in scene['dialogues']:
            norm_text = re.sub(r'\([^)]*\)', '', dialogue['text']).strip()
            norm_text = re.sub(r'\[[^\]]*\]', '', norm_text).strip()
            norm_text = re.sub(r'\.{3,}', ' ', norm_text)
            norm_text = word_substitutions_to_make(norm_text)
            dialogue_words = norm_text.split()
            normalized_dialogue_words = [normalize_and_clean_word(word) for word in dialogue_words]
            # Filter out empty normalized words
            dialogue['normalized_text']  = [word for word in normalized_dialogue_words if word]
            dialogue['length_normalized_text'] = len(dialogue['normalized_text'])
            dialogue['scene_id'] = scene_id
            dialogue['matched_text_index_start'] = -1
            dialogue['matched_text_index_end'] = -1
            dialogue['matched_row_index_start'] = -1
            dialogue['matched_row_index_end'] = -1
            dialogue['matched_word_index_start'] = -1
            dialogue['matched_word_index_end'] = -1
            dialogue['round1_matched'] = False
            dialogue['round2_matched'] = False
            dialogue['round3_matched'] = False
            dialogue['round1_score'] = -1
            dialogue['round2_score'] = -1
            dialogue['round3_score'] = -1
            dialogue_list.append(dialogue)
    return dialogue_list


def match_dialogues_to_transcript_data(transcript_data, dialogue_list):
    """
    Populate dialogue_list with match information from an already populated dialogue_per_tr field.
    
    Args:
        transcript_data (list): Array of transcript objects with populated dialogue_per_tr field
        dialogue_list (list): List of dialogues from get_dialogue_list output
        
    Returns:
        list: Updated dialogue_list with populated match fields
    """
    
    # Create text to position mapping
    text_to_position_map = []  # Maps text index to (row_idx, word_idx)
    position_to_text_map = {}  # Maps (row_idx, word_idx) to text index
    
    text_index = 0
    for row_idx, row in enumerate(transcript_data):
        for word_idx, word in enumerate(row['words_per_tr']):
            text_to_position_map.append((row_idx, word_idx))
            position_to_text_map[(row_idx, word_idx)] = text_index
            text_index += 1
    
    def find_dialogue_start_position(dialogue_id):
        """Find the position where dialogue_id appears in dialogue_per_tr"""
        for row_idx, row in enumerate(transcript_data):
            for word_idx, d_id in enumerate(row['dialogue_per_tr']):
                if d_id == dialogue_id:
                    return row_idx, word_idx
        return None, None
    
    def find_dialogue_end_position(dialogue_id):
        """Find the position where -dialogue_id appears in dialogue_per_tr"""
        for row_idx, row in enumerate(transcript_data):
            for word_idx, d_id in enumerate(row['dialogue_per_tr']):
                if d_id == -dialogue_id:
                    return row_idx, word_idx
        return None, None
    
    def get_next_matched_dialogue_start(current_dialogue_id):
        """Find the start position of the next matched dialogue (by ID)"""
        next_start_text_index = None
        for dialogue in dialogue_list:
            if (dialogue['id'] > current_dialogue_id and 
                dialogue.get('matched_text_index_start', -1) != -1):
                start_idx = dialogue.get('matched_text_index_start', -1)
                if next_start_text_index is None or start_idx < next_start_text_index:
                    next_start_text_index = start_idx
        return next_start_text_index
    
    def approximate_end_position(dialogue, start_row, start_word):
        """Approximate end position based on dialogue length and next dialogue"""
        dialogue_id = dialogue['id']
        normalized_length = dialogue.get('length_normalized_text', 0)
        
        # Calculate end based on normalized length
        start_text_index = position_to_text_map.get((start_row, start_word))
        if start_text_index is None:
            return start_row, start_word  # Fallback to start position
        
        # Option a: End based on normalized length
        length_based_end_text_index = start_text_index + normalized_length - 1
        
        # Option b: End 1 spot before next matched dialogue
        next_dialogue_start_text_index = get_next_matched_dialogue_start(dialogue_id)
        if next_dialogue_start_text_index is not None:
            next_based_end_text_index = next_dialogue_start_text_index - 1
        else:
            next_based_end_text_index = len(text_to_position_map) - 1  # End of transcript
        
        # Take the lesser of the two
        final_end_text_index = min(length_based_end_text_index, next_based_end_text_index)
        
        # Ensure we don't go beyond transcript bounds
        final_end_text_index = max(start_text_index, min(final_end_text_index, len(text_to_position_map) - 1))
        
        # Convert back to row, word position
        end_row, end_word = text_to_position_map[final_end_text_index]
        
        return end_row, end_word
    
    matched_count = 0
    approximated_ends = 0
    
    print(f"Processing {len(dialogue_list)} dialogues to populate match information")
    
    for dialogue in dialogue_list:
        dialogue_id = dialogue['id']
        
        # Initialize all match fields to -1
        dialogue['matched_text_index_start'] = -1
        dialogue['matched_text_index_end'] = -1
        dialogue['matched_row_index_start'] = -1
        dialogue['matched_row_index_end'] = -1
        dialogue['matched_word_index_start'] = -1
        dialogue['matched_word_index_end'] = -1
        
        # Find start position
        start_row, start_word = find_dialogue_start_position(dialogue_id)
        
        if start_row is not None and start_word is not None:
            # Found start position
            start_text_index = position_to_text_map.get((start_row, start_word))
            
            dialogue['matched_text_index_start'] = start_text_index
            dialogue['matched_row_index_start'] = start_row
            dialogue['matched_word_index_start'] = start_word
            
            # Find end position
            end_row, end_word = find_dialogue_end_position(dialogue_id)
            
            if end_row is not None and end_word is not None:
                # Found explicit end position
                end_text_index = position_to_text_map.get((end_row, end_word))
                
                dialogue['matched_text_index_end'] = end_text_index
                dialogue['matched_row_index_end'] = end_row
                dialogue['matched_word_index_end'] = end_word
                
                print(f"  ✓ Dialogue {dialogue_id}: start ({start_row}, {start_word}), end ({end_row}, {end_word})")
            else:
                # Approximate end position
                end_row, end_word = approximate_end_position(dialogue, start_row, start_word)
                end_text_index = position_to_text_map.get((end_row, end_word))
                
                dialogue['matched_text_index_end'] = end_text_index
                dialogue['matched_row_index_end'] = end_row
                dialogue['matched_word_index_end'] = end_word
                
                approximated_ends += 1
                print(f"  ≈ Dialogue {dialogue_id}: start ({start_row}, {start_word}), end approximated ({end_row}, {end_word})")
            
            matched_count += 1
        else:
            print(f"  ✗ Dialogue {dialogue_id}: not found in dialogue_per_tr")
    
    #for all matched dialogues, set run_rate
    for dialogue in dialogue_list:
        if dialogue['matched_text_index_start'] != -1:
            mapped_length = dialogue['matched_text_index_end'] - dialogue['matched_text_index_start'] + 1
            if mapped_length > 0:
                dialogue['run_rate'] = dialogue['length_normalized_text'] / mapped_length 
            else:
                dialogue['run_rate'] = 1
        else:
            dialogue['run_rate'] = 0
    
    print(f"\nMatch Summary:")
    print(f"  Total dialogues: {len(dialogue_list)}")
    print(f"  Successfully matched: {matched_count}")
    print(f"  Approximated ends: {approximated_ends}")
    print(f"  Match rate: {matched_count/len(dialogue_list)*100:.1f}%" if dialogue_list else "  Match rate: 0%")
    
    return dialogue_list, text_to_position_map, position_to_text_map
