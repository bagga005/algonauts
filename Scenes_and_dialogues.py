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
    
    #print(f"Processing {len(dialogue_list)} dialogues to populate match information")
    
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
                
                #print(f"  ✓ Dialogue {dialogue_id}: start ({start_row}, {start_word}), end ({end_row}, {end_word})")
            else:
                # Approximate end position
                end_row, end_word = approximate_end_position(dialogue, start_row, start_word)
                end_text_index = position_to_text_map.get((end_row, end_word))
                
                dialogue['matched_text_index_end'] = end_text_index
                dialogue['matched_row_index_end'] = end_row
                dialogue['matched_word_index_end'] = end_word
                
                approximated_ends += 1
                #print(f"  ≈ Dialogue {dialogue_id}: start ({start_row}, {start_word}), end approximated ({end_row}, {end_word})")
            
            matched_count += 1
        
            #print(f"  ✗ Dialogue {dialogue_id}: not found in dialogue_per_tr")
    
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
    
    #print(f"\nMatch Summary:")
    #print(f"  Total dialogues: {len(dialogue_list)}")
    #print(f"  Successfully matched: {matched_count}")
    #print(f"  Approximated ends: {approximated_ends}")
    #print(f"  Match rate: {matched_count/len(dialogue_list)*100:.1f}%" if dialogue_list else "  Match rate: 0%")
    
    return dialogue_list, text_to_position_map, position_to_text_map

def get_dialogue_display_text(dialogue, withSpeaker=True, start_index=0, length=-1, add_prefix_continuation=False, add_suffix_continuation=False):
    # Split text into words
    #
    if length != 0:
        words = dialogue['normalized_text']
        if length == -1:
            length = len(words) 
            end_index = len(words) - 1
        else:
            end_index = start_index + length -1
        useFuzzy = False

        useFancyText = True
        fancy_words = dialogue['text'].split()
        length_of_fancy_words = len(fancy_words)
        length_of_normal_words = len(words)
        if length_of_normal_words == length_of_fancy_words:
            #print("Case 1: No need to use fuzzy matching")
            useFancyText = True
            full_text_start_index = start_index
            full_text_end_index = end_index
        else:
            #print("Case 2: Using fuzzy matching")
            useFuzzy = True
            start_word = words[start_index]
            end_word = words[end_index]
            full_text_start_index, full_text_end_index = normalized_to_full_text(start_word, end_word, dialogue['text'], start_index, length)
            #print(f"after search full_text_start_index: {full_text_start_index}, full_text_end_index: {full_text_end_index}")
            if full_text_start_index == -1 or full_text_end_index == -1 or full_text_start_index > full_text_end_index or \
                (full_text_end_index - full_text_start_index) < (length - 1):
                useFancyText = False
            else:
                useFancyText = True
                
        #useFancyText = False
        if useFancyText:
            if start_index == 0:
                full_text_start_index = 0
            if end_index == len(words) - 1:
                full_text_end_index = len(fancy_words) - 1
            fancy_text = " ".join(fancy_words[full_text_start_index:full_text_end_index+1])
        else:
            fancy_text  = " ".join(words[start_index:end_index+1])
        basic_text = " ".join(words[start_index:end_index+1])
    else:
        fancy_text, basic_text = "", ""
    
    #add prefix and suffix continuation if needed
    if add_prefix_continuation:
        basic_text = f"... {basic_text}"
        fancy_text = f"... {fancy_text}"
    if add_suffix_continuation:
        basic_text = f"{basic_text} ..."
        fancy_text = f"{fancy_text} ..."

    #add speaker if needed
    if dialogue["speaker"] is not None and withSpeaker:
        if basic_text:
            basic_text = f"{dialogue['speaker']}: {basic_text}"
        if fancy_text:
            fancy_text = f"{dialogue['speaker']}: {fancy_text}"

    response = {
        "fancy": fancy_text,
        "normal": basic_text
    }
    #print(f"response get_dialogue_display_text: {response}")
    return response

def get_scene_for_dialogue(dialogue, scene_and_dialogues):
    for scene in scene_and_dialogues['scenes']:
        if dialogue['scene_id'] == scene['id']:
            return scene
    return None

def get_scene_display_text(scene):
    prefeix ='| Scene'
    middle = scene['desc']
    if middle:
        middle = ': ' + middle + ' '
    else:
        middle = ' '
    suffix = '|'
    return prefeix + middle + suffix


def normalized_to_full_text(start_word, end_word, text, preferred_start_index=0, preferred_length=1):
    """
    Find the indices of words in text that best match start_word and end_word using fuzzy matching.
    Ignores words inside () and [] brackets.
    
    Args:
        start_word (str): The word to find at the start
        end_word (str): The word to find at the end
        text (str): The text to search in
        preferred_start_index (int): Give slight preference to start words after this index
        preferred_length (int): Give preference to end words after preferred_start_index + preferred_length
        
    Returns:
        tuple: (start_index, end_index) where indices refer to word positions in the original text
               Returns (-1, -1) if no matches found
    """
    import difflib
    #print(f"start_word: {start_word}, end_word: {end_word}")
    
    def calculate_similarity(word1, word2):
        """Calculate similarity between two words using sequence matching"""
        if not word1 or not word2:
            return 0.0
        return difflib.SequenceMatcher(None, word1.lower(), word2.lower()).ratio()
    
    # Remove content inside brackets and parentheses
    import re
    cleaned_text = re.sub(r'\([^)]*\)', '', text)
    cleaned_text = re.sub(r'\[[^\]]*\]', '', cleaned_text)
    
    # Split original text into words for index mapping
    original_words = text.split()
    
    # Split cleaned text into words and normalize them
    cleaned_words = cleaned_text.split()
    normalized_words = []
    original_indices = []  # Track original word positions
    
    # Create mapping between cleaned words and original positions
    original_word_idx = 0
    for cleaned_word in cleaned_words:
        # Find this cleaned word in the original words
        while original_word_idx < len(original_words):
            # Remove brackets from original word for comparison
            orig_word_clean = re.sub(r'\([^)]*\)', '', original_words[original_word_idx])
            orig_word_clean = re.sub(r'\[[^\]]*\]', '', orig_word_clean)
            
            if orig_word_clean.strip() == cleaned_word:
                normalized_word = normalize_and_clean_word(cleaned_word)
                if normalized_word:  # Only include non-empty normalized words
                    normalized_words.append(normalized_word)
                    original_indices.append(original_word_idx)
                original_word_idx += 1
                break
            original_word_idx += 1
    
    if not normalized_words:
        return (-1, -1)
    
    # Normalize input words
    norm_start_word = normalize_and_clean_word(start_word)
    norm_end_word = normalize_and_clean_word(end_word)
    
    if not norm_start_word or not norm_end_word:
        return (-1, -1)
    
    # Find best match for start word with preference for words after preferred_start_index
    start_similarities = []
    for i, word in enumerate(normalized_words):
        similarity = calculate_similarity(norm_start_word, word)
        #print(f"similarity: {similarity}", norm_start_word, word)
        # Give slight bonus to words that come after preferred_start_index
        if original_indices[i] >= preferred_start_index:
            similarity += 0.05  # Small bonus for being at or after preferred start
        
        start_similarities.append((similarity, i))
    
    # Sort by similarity (descending) and get best match
    start_similarities.sort(key=lambda x: x[0], reverse=True)
    #print(f"start_similarities: {start_similarities}")
    best_start_similarity, start_index = start_similarities[0]
    
    # If no reasonable match for start word, return failure
    if best_start_similarity < 0.3:  # Minimum similarity threshold
        return (-1, -1)
    
    # Calculate preferred end position
    preferred_end_index = preferred_start_index + preferred_length
    
    # Find best match for end word with preference for words after calculated position
    end_similarities = []
    for i, word in enumerate(normalized_words):
        similarity = calculate_similarity(norm_end_word, word)
        
        # Give bonus to words that come after the found start word
        if i > start_index:
            similarity += 0.1  # Bonus for being after actual start word
        
        # Give additional bonus to words that come after preferred end position
        if original_indices[i] >= preferred_end_index:
            similarity += 0.05  # Additional bonus for being at or after preferred end
        
        end_similarities.append((similarity, i))
    
    # Sort by similarity (descending) and get best match
    end_similarities.sort(key=lambda x: x[0], reverse=True)
    best_end_similarity, end_index = end_similarities[0]
    
    # If no reasonable match for end word, return failure
    if best_end_similarity < 0.3:  # Minimum similarity threshold
        return (-1, -1)
    
    # Ensure end comes at or after start
    if end_index < start_index:
        # Find the best end word that comes after start
        valid_end_matches = [(sim, idx) for sim, idx in end_similarities if idx >= start_index]
        if valid_end_matches:
            best_end_similarity, end_index = max(valid_end_matches, key=lambda x: x[0])
            if best_end_similarity < 0.3:
                return (-1, -1)
        else:
            # No valid end word after start, use start index as end
            end_index = start_index
    
    # Return original text indices instead of cleaned text indices
    return (original_indices[start_index], original_indices[end_index])

def get_scene_and_dialogues_display_text(scene_and_dialogues, dialogue_list, scene_id, starting_diaglogue_id=-1, max_words=1000):
    def get_dialogue_by_id(dialogue_id):
        for dialogue in dialogue_list:
            if dialogue['id'] == dialogue_id:
                return dialogue
        return None
    def get_scene_by_id(scene_id):
            for scene in scene_and_dialogues['scenes']:
                if scene['id'] == scene_id:
                    return scene
            return None
    response = {
        "fancy_scene_text": "",
        "normal_scene_text": ""
    }

    scene = get_scene_by_id(scene_id)
    if scene is None:
        raise ValueError(f"Scene {scene_id} not found")
    
    scene_text = get_scene_display_text(scene)
    words_left = max_words - len(scene_text.split())
    if words_left < 0:
        return response
    
    ending_index_of_dialogue_in_scene = -1
    need_scene_contituation_suffix = False
    if starting_diaglogue_id != -1:
        starting_dialogue = get_dialogue_by_id(starting_diaglogue_id)
        if starting_dialogue is None or starting_dialogue['scene_id'] != scene_id:
            raise ValueError(f"Starting dialogue {starting_diaglogue_id} is not in scene {scene_id} or is not a valid dialogue id")
        for i in range(len(scene['dialogues'])-1, -1, -1):
            if scene['dialogues'][i]['id'] == starting_diaglogue_id:
                ending_index_of_dialogue_in_scene = i
                break
    else:
        scene_dialogues = scene['dialogues']
        scene_dialogues.sort(key=lambda x: x['id'])
        ending_index_of_dialogue_in_scene = len(scene_dialogues) - 1
    
    if ending_index_of_dialogue_in_scene > 0:
        for i in range(ending_index_of_dialogue_in_scene, -1, -1):
            dialogue = scene['dialogues'][i]
            display_dialogue_text = get_dialogue_display_text(dialogue, withSpeaker=True)
            if display_dialogue_text['fancy']:
                if words_left > len(display_dialogue_text['fancy'].split()):
                    if response['fancy_scene_text']:
                        response['fancy_scene_text'] = display_dialogue_text['fancy'] + "\n" + response['fancy_scene_text']
                    else:
                        response['fancy_scene_text'] = display_dialogue_text['fancy']
                    words_left = words_left - len(display_dialogue_text['fancy'].split())
                    if display_dialogue_text['normal'] and words_left > len(display_dialogue_text['normal'].split()):
                        if response['normal_scene_text']:
                            response['normal_scene_text'] = display_dialogue_text['normal'] + "\n" + response['normal_scene_text']
                        else:
                            response['normal_scene_text'] = display_dialogue_text['normal']
                else:
                    need_scene_contituation_suffix = True
                    break
    if need_scene_contituation_suffix:
        scene_text = scene_text + "\n" + "..."
   
    if response['fancy_scene_text']:
        response['fancy_scene_text'] = scene_text + "\n" + response['fancy_scene_text']
    else:
        response['fancy_scene_text'] = scene_text
    if response['normal_scene_text']:
        response['normal_scene_text'] = scene_text + "\n" + response['normal_scene_text']
    else:
        response['normal_scene_text'] = scene_text
    
    #print(f"words_left: {words_left}")
    return response



def get_closest_dialogue_for_row(row_idx, dialogue_list):
    """
    Find the dialogue that best matches a given row index.
    
    First tries to find a dialogue whose range covers the row.
    If none found, returns the dialogue whose range ends closest to (but before) the row.
    
    Args:
        row_idx (int): The row index to find a dialogue for
        dialogue_list (list): List of dialogues with match information
        
    Returns:
        dict or None: The best matching dialogue, or None if no suitable dialogue found
    """
    
    # Filter out dialogues that don't have valid match information
    valid_dialogues = [d for d in dialogue_list 
                      if d.get('matched_row_index_start', -1) != -1 
                      and d.get('matched_row_index_end', -1) != -1]
    
    if not valid_dialogues:
        return None
    
    # Sort dialogues by id in ascending order
    sorted_dialogues = sorted(valid_dialogues, key=lambda x: x['id'])
    
    # First pass: find first dialogue whose range covers the row (inclusive)
    for dialogue in sorted_dialogues:
        start_row = dialogue['matched_row_index_start']
        end_row = dialogue['matched_row_index_end']
        
        if start_row <= row_idx <= end_row:
            return dialogue
    
    # Second pass: find dialogue whose range ends closest to the row but before the row
    closest_dialogue = None
    closest_end_row = -1
    
    for dialogue in sorted_dialogues:
        end_row = dialogue['matched_row_index_end']
        
        # Check if this dialogue ends before the target row and is closer than previous best
        if end_row < row_idx and end_row > closest_end_row:
            closest_dialogue = dialogue
            closest_end_row = end_row
    
    return closest_dialogue




