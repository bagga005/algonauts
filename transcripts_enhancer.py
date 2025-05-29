import re
from rapidfuzz import process, fuzz
from rapidfuzz.fuzz import ratio
import copy

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
                   if line.strip() and 'Commercial Break' not in line and 'Closing Credits' not in line and 'OPENING TITLES' not in line and 'END' not in line and 'Opening Credits' not in line]
    
    assert valid_lines, "No dialogues found"
    
    # Assert that the first valid line is a scene
    assert valid_lines[0].startswith('['), "First valid line is not a scene"
    
    scenes = []
    current_scene = None
    current_dialogue = None
    dialogue_id = 1
    scene_id = 1
    dialogue_line_count = 0
    in_scene_index = 1  # Track dialogue order within each scene
    
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
                
                # Add total_dialogues_scene to each dialogue in the scene
                total_dialogues_in_scene = len(current_scene["dialogues"])
                for dialogue in current_scene["dialogues"]:
                    dialogue["total_dialogues_scene"] = total_dialogues_in_scene
                
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
                    assert dialogue_line_count <= 3, f"Dialogue exceeds 3 lines: {current_dialogue}"
                    current_scene["dialogues"].append(current_dialogue)
                
                # Start new dialogue
                speaker = speaker_match.group(1).strip()
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
        # Add total_dialogues_scene to each dialogue in the last scene
        total_dialogues_in_scene = len(current_scene["dialogues"])
        for dialogue in current_scene["dialogues"]:
            dialogue["total_dialogues_scene"] = total_dialogues_in_scene
        
        scenes.append(current_scene)
    
    return {"scenes": scenes}




def word_substitutions_to_make(segment):
    segment = re.sub(r'whaddya', 'what do you', segment.lower())
    segment = re.sub(r'goodnight', 'good night', segment.lower())
    segment = re.sub(r'waitwait', 'wait wait', segment.lower())
    return segment

def check_dialogue_list_validity(dialogue_list, transcript_data, max_dialogue_distance_for_overlap=1):
    """
    Checks the validity of dialogue ordering based on their IDs and matched text indices,
    and checks for boundary overlaps between dialogues.
    
    Args:
        dialogue_list (list): List of dialogues from get_dialogue_list output
        transcript_data (list): Array of transcript objects from load_transcript_tsv
        max_dialogue_distance_for_overlap (int): Maximum dialogue ID distance to allow overlaps without considering it a mistake (default: 1)
        
    Returns:
        dict: Dictionary containing validation results with the following structure:
        {
            'summary': {
                'total_dialogues': int,
                'matched_dialogues': int,
                'match_rate': float,
                'round_matches': {1: int, 2: int, 3: int, 4: int, 5: int, 6: int, 7: int}
            },
            'word_statistics': {
                'dialogue_words': {
                    'total': int,
                    'unmatched': int,
                    'matched': int,
                    'match_rate': float
                },
                'transcript_words': {
                    'total': int,
                    'unmatched': int,
                    'matched': int,
                    'match_rate': float
                }
            },
            'ordering_mistakes': {
                'total': int,
                'round1': int, 'round2': int, 'round3': int, 'round4': int,
                'round5': int, 'round6': int, 'round7': int
            },
            'boundary_mistakes': {
                'total': int,
                'round1': int, 'round2': int, 'round3': int, 'round4': int,
                'round5': int, 'round6': int, 'round7': int
            },
            'transcript_entry_mistakes': {
                'missing_start_entries': int,
                'missing_end_entries': int
            },
            'total_mistakes': {
                'total': int,
                'round1': int, 'round2': int, 'round3': int, 'round4': int,
                'round5': int, 'round6': int, 'round7': int
            },
            'scores': {
                'overall': {
                    'round1': float, 'round2': float, 'round3': float, 'round4': float,
                    'round5': float, 'round6': float, 'round7': float
                },
                'matched': {
                    'round1': float, 'round2': float, 'round3': float, 'round4': float,
                    'round5': float, 'round6': float, 'round7': float
                }
            },
            'mistake_details': {
                'ordering': list,
                'boundary': list,
                'missing_start_entries': list,
                'missing_end_entries': list
            }
        }
    """
    
    # Create text to position mapping for looking up words
    text_to_position_map = []  # Maps text index to (row_idx, word_idx)
    full_text_words = []
    
    for row_idx, row in enumerate(transcript_data):
        for word_idx, word in enumerate(row['words_per_tr']):
            full_text_words.append(word)
            text_to_position_map.append((row_idx, word_idx))
    
    def get_words_from_text_range(start_idx, end_idx):
        """Helper function to get actual words from transcript data given text indices"""
        if start_idx < 0 or end_idx >= len(full_text_words) or start_idx > end_idx:
            return "Invalid range"
        return " ".join(full_text_words[start_idx:end_idx+1])
    
    # Filter dialogues that have been matched (matched_text_index_start != -1)
    matched_dialogues = [d for d in dialogue_list if d.get('matched_text_index_start', -1) != -1]
    
    # Filter dialogues by round
    round1_dialogues = [d for d in dialogue_list if d.get('round1_matched', False)]
    round2_dialogues = [d for d in dialogue_list if d.get('round2_matched', False)]
    round3_dialogues = [d for d in dialogue_list if d.get('round3_matched', False)]
    round4_dialogues = [d for d in dialogue_list if d.get('round4_matched', False)]
    round5_dialogues = [d for d in dialogue_list if d.get('round5_matched', False)]
    round6_dialogues = [d for d in dialogue_list if d.get('round6_matched', False)]
    round7_dialogues = [d for d in dialogue_list if d.get('round7_matched', False)]
    
    def check_transcript_entries(matched_dialogues_subset):
        """Helper function to check if matched dialogues have corresponding entries in transcript_data"""
        missing_start_entries = []
        missing_end_entries = []
        
        for dialogue in matched_dialogues_subset:
            dialogue_id = dialogue['id']
            
            # Check if dialogue has valid matched positions
            start_row = dialogue.get('matched_row_index_start', -1)
            start_word = dialogue.get('matched_word_index_start', -1)
            end_row = dialogue.get('matched_row_index_end', -1)
            end_word = dialogue.get('matched_word_index_end', -1)
            
            if start_row == -1 or start_word == -1:
                # Dialogue is marked as matched but doesn't have valid position info
                missing_start_entries.append({
                    'dialogue_id': dialogue_id,
                    'text': dialogue.get('text', 'N/A')[:50] + '...',
                    'reason': 'Invalid matched position info'
                })
                continue
            
            # Check start entry in transcript_data
            if (start_row >= len(transcript_data) or 
                start_word >= len(transcript_data[start_row]['dialogue_per_tr']) or
                transcript_data[start_row]['dialogue_per_tr'][start_word] != dialogue_id):
                
                actual_value = None
                if start_row < len(transcript_data) and start_word < len(transcript_data[start_row]['dialogue_per_tr']):
                    actual_value = transcript_data[start_row]['dialogue_per_tr'][start_word]
                
                missing_start_entries.append({
                    'dialogue_id': dialogue_id,
                    'expected_row': start_row,
                    'expected_word': start_word,
                    'expected_value': dialogue_id,
                    'actual_value': actual_value,
                    'text': dialogue.get('text', 'N/A')[:50] + '...'
                })
            
            # Check end entry in transcript_data (only if end position is different from start)
            if end_row != -1 and end_word != -1 and (end_row != start_row or end_word != start_word):
                expected_end_value = -dialogue_id
                
                if (end_row >= len(transcript_data) or 
                    end_word >= len(transcript_data[end_row]['dialogue_per_tr']) or
                    transcript_data[end_row]['dialogue_per_tr'][end_word] != expected_end_value):
                    
                    actual_value = None
                    if end_row < len(transcript_data) and end_word < len(transcript_data[end_row]['dialogue_per_tr']):
                        actual_value = transcript_data[end_row]['dialogue_per_tr'][end_word]
                    
                    missing_end_entries.append({
                        'dialogue_id': dialogue_id,
                        'expected_row': end_row,
                        'expected_word': end_word,
                        'expected_value': expected_end_value,
                        'actual_value': actual_value,
                        'text': dialogue.get('text', 'N/A')[:50] + '...'
                    })
        
        return missing_start_entries, missing_end_entries

    def check_ordering_mistakes(dialogues_subset, round_name=""):
        """Helper function to check ordering mistakes in a subset of dialogues"""
        if len(dialogues_subset) < 2:
            return 0, []
        
        # Sort by ID for order checking
        sorted_dialogues = sorted(dialogues_subset, key=lambda x: x['id'])
        
        mistakes = 0
        mistake_details = []
        
        for i in range(len(sorted_dialogues) - 1):
            current_dialogue = sorted_dialogues[i]
            next_dialogue = sorted_dialogues[i + 1]
            
            current_id = current_dialogue['id']
            next_id = next_dialogue['id']
            current_text_index = current_dialogue['matched_text_index_start']
            next_text_index = next_dialogue['matched_text_index_start']
            
            # Check if a dialogue with lower ID has higher matched_text_index_start than a dialogue with higher ID
            if current_text_index > next_text_index:
                mistakes += 1
                mistake_details.append({
                    'type': 'ordering',
                    'round': round_name,
                    'dialogue_1_id': current_id,
                    'dialogue_1_text_index': current_text_index,
                    'dialogue_2_id': next_id,
                    'dialogue_2_text_index': next_text_index,
                    'text_1': current_dialogue.get('text', 'N/A')[:50] + '...',
                    'text_2': next_dialogue.get('text', 'N/A')[:50] + '...'
                })
        
        return mistakes, mistake_details
    
    def check_boundary_overlaps(dialogues_subset, round_name=""):
        """Helper function to check boundary overlaps in a subset of dialogues"""
        if len(dialogues_subset) < 2:
            return 0, []
        
        overlaps = 0
        overlap_details = []
        
        # Check all pairs of dialogues for overlaps
        for i in range(len(dialogues_subset)):
            for j in range(i + 1, len(dialogues_subset)):
                dialogue1 = dialogues_subset[i]
                dialogue2 = dialogues_subset[j]
                
                # Get boundary indices
                start1 = dialogue1.get('matched_text_index_start', -1)
                end1 = dialogue1.get('matched_text_index_end', -1)
                start2 = dialogue2.get('matched_text_index_start', -1)
                end2 = dialogue2.get('matched_text_index_end', -1)
                
                # Skip if any dialogue doesn't have valid boundaries
                if start1 == -1 or end1 == -1 or start2 == -1 or end2 == -1:
                    continue
                
                # Check for overlap: dialogue1_start <= dialogue2_end AND dialogue2_start <= dialogue1_end
                if start1 <= end2 and start2 <= end1:
                    # Check dialogue ID distance - only consider as mistake if distance > max_dialogue_distance_for_overlap
                    dialogue_distance = abs(dialogue1['id'] - dialogue2['id'])
                    
                    if dialogue_distance > max_dialogue_distance_for_overlap:
                        # Calculate the overlapping range
                        overlap_start = max(start1, start2)
                        overlap_end = min(end1, end2)
                        overlap_words = get_words_from_text_range(overlap_start, overlap_end)
                        
                        overlaps += 1
                        overlap_details.append({
                            'type': 'boundary',
                            'round': round_name,
                            'dialogue_1_id': dialogue1['id'],
                            'dialogue_1_start': start1,
                            'dialogue_1_end': end1,
                            'dialogue_2_id': dialogue2['id'],
                            'dialogue_2_start': start2,
                            'dialogue_2_end': end2,
                            'dialogue_distance': dialogue_distance,
                            'overlap_start': overlap_start,
                            'overlap_end': overlap_end,
                            'overlap_words': overlap_words,
                            'dialogue_1_words': get_words_from_text_range(start1, end1),
                            'dialogue_2_words': get_words_from_text_range(start2, end2),
                            'text_1': dialogue1.get('text', 'N/A')[:50] + '...',
                            'text_2': dialogue2.get('text', 'N/A')[:50] + '...'
                        })
        
        return overlaps, overlap_details

    # Check transcript entries for matched dialogues
    missing_start_entries, missing_end_entries = check_transcript_entries(matched_dialogues)
    
    # Check ordering mistakes for each round and overall
    total_ordering_mistakes, total_ordering_details = check_ordering_mistakes(matched_dialogues, "Total")
    ordering_mistakes_round1, round1_ordering_details = check_ordering_mistakes(round1_dialogues, "Round1")
    ordering_mistakes_round2, round2_ordering_details = check_ordering_mistakes(round2_dialogues, "Round2")
    ordering_mistakes_round3, round3_ordering_details = check_ordering_mistakes(round3_dialogues, "Round3")
    ordering_mistakes_round4, round4_ordering_details = check_ordering_mistakes(round4_dialogues, "Round4")
    ordering_mistakes_round5, round5_ordering_details = check_ordering_mistakes(round5_dialogues, "Round5")
    ordering_mistakes_round6, round6_ordering_details = check_ordering_mistakes(round6_dialogues, "Round6")
    ordering_mistakes_round7, round7_ordering_details = check_ordering_mistakes(round7_dialogues, "Round7")
    
    # Check boundary overlaps for each round and overall
    total_boundary_mistakes, total_boundary_details = check_boundary_overlaps(matched_dialogues, "Total")
    boundary_mistakes_round1, round1_boundary_details = check_boundary_overlaps(round1_dialogues, "Round1")
    boundary_mistakes_round2, round2_boundary_details = check_boundary_overlaps(round2_dialogues, "Round2")
    boundary_mistakes_round3, round3_boundary_details = check_boundary_overlaps(round3_dialogues, "Round3")
    boundary_mistakes_round4, round4_boundary_details = check_boundary_overlaps(round4_dialogues, "Round4")
    boundary_mistakes_round5, round5_boundary_details = check_boundary_overlaps(round5_dialogues, "Round5")
    boundary_mistakes_round6, round6_boundary_details = check_boundary_overlaps(round6_dialogues, "Round6")
    boundary_mistakes_round7, round7_boundary_details = check_boundary_overlaps(round7_dialogues, "Round7")
    
    # Calculate total mistakes (ordering + boundary)
    total_mistakes = total_ordering_mistakes + total_boundary_mistakes
    mistakes_round1 = ordering_mistakes_round1 + boundary_mistakes_round1
    mistakes_round2 = ordering_mistakes_round2 + boundary_mistakes_round2
    mistakes_round3 = ordering_mistakes_round3 + boundary_mistakes_round3
    mistakes_round4 = ordering_mistakes_round4 + boundary_mistakes_round4
    mistakes_round5 = ordering_mistakes_round5 + boundary_mistakes_round5
    mistakes_round6 = ordering_mistakes_round6 + boundary_mistakes_round6
    mistakes_round7 = ordering_mistakes_round7 + boundary_mistakes_round7

    # Calculate statistics
    total_dialogues = len(dialogue_list)
    matched_dialogues_count = len(matched_dialogues)
    
    # Calculate average scores
    def safe_avg(values):
        valid_values = [v for v in values if v != -1]
        return sum(valid_values) / len(valid_values) if valid_values else 0
    
    # Overall averages (including unmatched dialogues with -1 scores)
    all_round1_scores = [d.get('round1_score', -1) for d in dialogue_list]
    all_round2_scores = [d.get('round2_score', -1) for d in dialogue_list]
    all_round3_scores = [d.get('round3_score', -1) for d in dialogue_list]
    all_round4_scores = [d.get('round4_score', -1) for d in dialogue_list]
    all_round5_scores = [d.get('round5_score', -1) for d in dialogue_list]
    all_round6_scores = [d.get('round6_score', -1) for d in dialogue_list]
    all_round7_scores = [d.get('round7_score', -1) for d in dialogue_list]
    
    avg_round1_overall = safe_avg(all_round1_scores)
    avg_round2_overall = safe_avg(all_round2_scores)
    avg_round3_overall = safe_avg(all_round3_scores)
    avg_round4_overall = safe_avg(all_round4_scores)
    avg_round5_overall = safe_avg(all_round5_scores)
    avg_round6_overall = safe_avg(all_round6_scores)
    avg_round7_overall = safe_avg(all_round7_scores)
    
    # Matched dialogues averages
    matched_round1_scores = [d.get('round1_score', -1) for d in matched_dialogues]
    matched_round2_scores = [d.get('round2_score', -1) for d in matched_dialogues]
    matched_round3_scores = [d.get('round3_score', -1) for d in matched_dialogues]
    matched_round4_scores = [d.get('round4_score', -1) for d in matched_dialogues]
    matched_round5_scores = [d.get('round5_score', -1) for d in matched_dialogues]
    matched_round6_scores = [d.get('round6_score', -1) for d in matched_dialogues]
    matched_round7_scores = [d.get('round7_score', -1) for d in matched_dialogues]
    
    avg_round1_matched = safe_avg(matched_round1_scores)
    avg_round2_matched = safe_avg(matched_round2_scores)
    avg_round3_matched = safe_avg(matched_round3_scores)
    avg_round4_matched = safe_avg(matched_round4_scores)
    avg_round5_matched = safe_avg(matched_round5_scores)
    avg_round6_matched = safe_avg(matched_round6_scores)
    avg_round7_matched = safe_avg(matched_round7_scores)
    
    # Compute word count statistics
    # 1. Total words in dialogues text and unmatched words in dialogues text
    total_dialogue_words = 0
    unmatched_dialogue_words = 0
    unmatched_dialogues = [d for d in dialogue_list if d.get('matched_text_index_start', -1) == -1]
    
    for dialogue in dialogue_list:
        # Count words in dialogue text (split by whitespace)
        dialogue_word_count = len(dialogue['text'].split())
        total_dialogue_words += dialogue_word_count
        
        # If dialogue is unmatched, add to unmatched count
        if dialogue in unmatched_dialogues:
            unmatched_dialogue_words += dialogue_word_count
    
    # 2. Total words in transcript_data and unmatched words
    total_transcript_words = 0
    unmatched_transcript_words = 0
    
    # Create a set of all (row, word) positions that are covered by matched dialogues
    covered_positions = set()
    
    for dialogue in matched_dialogues:
        start_row = dialogue.get('matched_row_index_start', -1)
        start_word = dialogue.get('matched_word_index_start', -1)
        end_row = dialogue.get('matched_row_index_end', -1)
        end_word = dialogue.get('matched_word_index_end', -1)
        
        # Skip if dialogue doesn't have valid position info
        if start_row == -1 or start_word == -1 or end_row == -1 or end_word == -1:
            continue
            
        # Add all positions from start to end (inclusive) to covered set
        current_row = start_row
        current_word = start_word
        
        while True:
            covered_positions.add((current_row, current_word))
            
            # Break if we've reached the end position
            if current_row == end_row and current_word == end_word:
                break
                
            # Move to next position
            current_word += 1
            if current_word >= len(transcript_data[current_row]['words_per_tr']):
                current_row += 1
                current_word = 0
                
            # Safety check to prevent infinite loops
            if current_row > end_row or (current_row == end_row and current_word > end_word):
                break
    
    # Count total and unmatched words in transcript_data
    for row_idx, row in enumerate(transcript_data):
        for word_idx, word in enumerate(row['words_per_tr']):
            total_transcript_words += 1
            
            # If this position is not covered by any matched dialogue, it's unmatched
            if (row_idx, word_idx) not in covered_positions:
                unmatched_transcript_words += 1
    
    # Calculate coverage statistics
    dialogue_match_rate = ((total_dialogue_words - unmatched_dialogue_words) / total_dialogue_words * 100) if total_dialogue_words > 0 else 0
    transcript_match_rate = ((total_transcript_words - unmatched_transcript_words) / total_transcript_words * 100) if total_transcript_words > 0 else 0
    
    # Calculate average length of unmatched dialogues based on normalized text
    avg_unmatched_dialogue_length = 0
    if len(unmatched_dialogues) > 0:
        total_unmatched_normalized_length = sum(dialogue['length_normalized_text'] for dialogue in unmatched_dialogues)
        avg_unmatched_dialogue_length = total_unmatched_normalized_length / len(unmatched_dialogues)
    
    # Show summary
    print("=== DIALOGUE LIST VALIDITY CHECK ===")
    print(f"Total dialogues: {total_dialogues}")
    print(f"Matched dialogues: {matched_dialogues_count}")
    print(f"Match rate: {matched_dialogues_count/total_dialogues*100:.1f}%")
    print(f"Round1 matched: {len(round1_dialogues)}")
    print(f"Round2 matched: {len(round2_dialogues)}")
    print(f"Round3 matched: {len(round3_dialogues)}")
    print(f"Round4 matched: {len(round4_dialogues)}")
    print(f"Round5 matched: {len(round5_dialogues)}")
    print(f"Round6 matched: {len(round6_dialogues)}")
    print(f"Round7 matched: {len(round7_dialogues)}")
    print(f"Max dialogue distance for overlap: {max_dialogue_distance_for_overlap}")
    print()
    
    print("=== WORD COUNT STATISTICS ===")
    print(f"Total words in dialogues: {total_dialogue_words}")
    print(f"Unmatched words in dialogues: {unmatched_dialogue_words}")
    print(f"Dialogue word match rate: {dialogue_match_rate:.1f}%")
    print(f"Average length of unmatched dialogues (normalized): {avg_unmatched_dialogue_length:.2f}")
    print(f"Total words in transcript: {total_transcript_words}")
    print(f"Unmatched words in transcript: {unmatched_transcript_words}")
    print(f"Transcript word match rate: {transcript_match_rate:.1f}%")
    print()
    
    print("=== TRANSCRIPT ENTRY VALIDATION ===")
    print(f"Missing start entries: {len(missing_start_entries)}")
    print(f"Missing end entries: {len(missing_end_entries)}")
    if len(matched_dialogues) > 0:
        print(f"Start entry accuracy: {(len(matched_dialogues)-len(missing_start_entries))/len(matched_dialogues)*100:.1f}%")
        print(f"End entry accuracy: {(len(matched_dialogues)-len(missing_end_entries))/len(matched_dialogues)*100:.1f}%")
    print()
    
    print("=== ORDERING VALIDATION ===")
    print(f"Total ordering mistakes: {total_ordering_mistakes}")
    print(f"Round1 ordering mistakes: {ordering_mistakes_round1}")
    print(f"Round2 ordering mistakes: {ordering_mistakes_round2}")
    print(f"Round3 ordering mistakes: {ordering_mistakes_round3}")
    print(f"Round4 ordering mistakes: {ordering_mistakes_round4}")
    print(f"Round5 ordering mistakes: {ordering_mistakes_round5}")
    print(f"Round6 ordering mistakes: {ordering_mistakes_round6}")
    print(f"Round7 ordering mistakes: {ordering_mistakes_round7}")
    
    if len(matched_dialogues) > 1:
        print(f"Total ordering accuracy: {(len(matched_dialogues)-1-total_ordering_mistakes)/(len(matched_dialogues)-1)*100:.1f}%")
    if len(round1_dialogues) > 1:
        print(f"Round1 ordering accuracy: {(len(round1_dialogues)-1-ordering_mistakes_round1)/(len(round1_dialogues)-1)*100:.1f}%")
    if len(round2_dialogues) > 1:
        print(f"Round2 ordering accuracy: {(len(round2_dialogues)-1-ordering_mistakes_round2)/(len(round2_dialogues)-1)*100:.1f}%")
    if len(round3_dialogues) > 1:
        print(f"Round3 ordering accuracy: {(len(round3_dialogues)-1-ordering_mistakes_round3)/(len(round3_dialogues)-1)*100:.1f}%")
    if len(round4_dialogues) > 1:
        print(f"Round4 ordering accuracy: {(len(round4_dialogues)-1-ordering_mistakes_round4)/(len(round4_dialogues)-1)*100:.1f}%")
    if len(round5_dialogues) > 1:
        print(f"Round5 ordering accuracy: {(len(round5_dialogues)-1-ordering_mistakes_round5)/(len(round5_dialogues)-1)*100:.1f}%")
    if len(round6_dialogues) > 1:
        print(f"Round6 ordering accuracy: {(len(round6_dialogues)-1-ordering_mistakes_round6)/(len(round6_dialogues)-1)*100:.1f}%")
    if len(round7_dialogues) > 1:
        print(f"Round7 ordering accuracy: {(len(round7_dialogues)-1-ordering_mistakes_round7)/(len(round7_dialogues)-1)*100:.1f}%")
    print()
    
    print("=== BOUNDARY OVERLAP VALIDATION ===")
    print(f"Total boundary overlaps (distance > {max_dialogue_distance_for_overlap}): {total_boundary_mistakes}")
    print(f"Round1 boundary overlaps: {boundary_mistakes_round1}")
    print(f"Round2 boundary overlaps: {boundary_mistakes_round2}")
    print(f"Round3 boundary overlaps: {boundary_mistakes_round3}")
    print(f"Round4 boundary overlaps: {boundary_mistakes_round4}")
    print(f"Round5 boundary overlaps: {boundary_mistakes_round5}")
    print(f"Round6 boundary overlaps: {boundary_mistakes_round6}")
    print(f"Round7 boundary overlaps: {boundary_mistakes_round7}")
    print()
    
    print("=== COMBINED VALIDATION ===")
    print(f"Total mistakes (ordering + boundary): {total_mistakes}")
    print(f"Round1 total mistakes: {mistakes_round1}")
    print(f"Round2 total mistakes: {mistakes_round2}")
    print(f"Round3 total mistakes: {mistakes_round3}")
    print(f"Round4 total mistakes: {mistakes_round4}")
    print(f"Round5 total mistakes: {mistakes_round5}")
    print(f"Round6 total mistakes: {mistakes_round6}")
    print(f"Round7 total mistakes: {mistakes_round7}")
    print()
    
    # Show mistake details for each round
    all_ordering_mistakes = total_ordering_details + round1_ordering_details + round2_ordering_details + round3_ordering_details + \
                             round4_ordering_details + round5_ordering_details + round6_ordering_details + round7_ordering_details
    all_boundary_mistakes = total_boundary_details + round1_boundary_details + round2_boundary_details + round3_boundary_details + \
                             round4_boundary_details + round5_boundary_details + round6_boundary_details + round7_boundary_details
    
    # Show transcript entry mistake details
    if missing_start_entries:
        print("=== MISSING START ENTRY DETAILS ===")
        for i, mistake in enumerate(missing_start_entries[:15]):  # Show first 15 mistakes
            print(f"Missing Start Entry {i+1}:")
            print(f"  Dialogue ID: {mistake['dialogue_id']}")
            if 'expected_row' in mistake:
                print(f"  Expected position: row {mistake['expected_row']}, word {mistake['expected_word']}")
                print(f"  Expected value: {mistake['expected_value']}")
                print(f"  Actual value: {mistake['actual_value']}")
            else:
                print(f"  Reason: {mistake['reason']}")
            print(f"  Text: '{mistake['text']}'")
            print()
        if len(missing_start_entries) > 15:
            print(f"... and {len(missing_start_entries) - 15} more missing start entries")
        print()
    
    if missing_end_entries:
        print("=== MISSING END ENTRY DETAILS ===")
        for i, mistake in enumerate(missing_end_entries[:15]):  # Show first 15 mistakes
            print(f"Missing End Entry {i+1}:")
            print(f"  Dialogue ID: {mistake['dialogue_id']}")
            print(f"  Expected position: row {mistake['expected_row']}, word {mistake['expected_word']}")
            print(f"  Expected value: {mistake['expected_value']}")
            print(f"  Actual value: {mistake['actual_value']}")
            print(f"  Text: '{mistake['text']}'")
            print()
        if len(missing_end_entries) > 15:
            print(f"... and {len(missing_end_entries) - 15} more missing end entries")
        print()

    if all_ordering_mistakes:
        print("=== ORDERING MISTAKES DETAILS ===")
        for i, mistake in enumerate(all_ordering_mistakes[:15]):  # Show first 15 mistakes
            print(f"Ordering Mistake {i+1} ({mistake['round']}):")
            print(f"  Dialogue {mistake['dialogue_1_id']} (text_index: {mistake['dialogue_1_text_index']}) comes before")
            print(f"  Dialogue {mistake['dialogue_2_id']} (text_index: {mistake['dialogue_2_text_index']}) but has higher text index")
            print(f"  Text 1: '{mistake['text_1']}'")
            print(f"  Text 2: '{mistake['text_2']}'")
            print()
        if len(all_ordering_mistakes) > 15:
            print(f"... and {len(all_ordering_mistakes) - 15} more ordering mistakes")
        print()
    
    if all_boundary_mistakes:
        print("=== BOUNDARY OVERLAP MISTAKES DETAILS ===")
        for i, mistake in enumerate(all_boundary_mistakes[:15]):  # Show first 15 mistakes
            print(f"Boundary Overlap {i+1} ({mistake['round']}):")
            print(f"  Dialogue {mistake['dialogue_1_id']} (range: {mistake['dialogue_1_start']}-{mistake['dialogue_1_end']}) overlaps with")
            print(f"  Dialogue {mistake['dialogue_2_id']} (range: {mistake['dialogue_2_start']}-{mistake['dialogue_2_end']})")
            print(f"  Dialogue distance: {mistake['dialogue_distance']} (threshold: {max_dialogue_distance_for_overlap})")
            print(f"  Overlap range: {mistake['overlap_start']}-{mistake['overlap_end']}")
            print(f"  Overlapping words: '{mistake['overlap_words']}'")
            print(f"  Dialogue 1 words: '{mistake['dialogue_1_words']}'")
            print(f"  Dialogue 2 words: '{mistake['dialogue_2_words']}'")
            print(f"  Dialogue 1 text: '{mistake['text_1']}'")
            print(f"  Dialogue 2 text: '{mistake['text_2']}'")
            print()
        if len(all_boundary_mistakes) > 15:
            print(f"... and {len(all_boundary_mistakes) - 15} more boundary overlap mistakes")
        print()
    
    print("=== SCORE STATISTICS ===")
    print(f"Average round1_score (overall): {avg_round1_overall:.2f}")
    print(f"Average round1_score (matched): {avg_round1_matched:.2f}")
    print(f"Average round2_score (overall): {avg_round2_overall:.2f}")
    print(f"Average round2_score (matched): {avg_round2_matched:.2f}")
    print(f"Average round3_score (overall): {avg_round3_overall:.2f}")
    print(f"Average round3_score (matched): {avg_round3_matched:.2f}")
    print(f"Average round4_score (overall): {avg_round4_overall:.2f}")
    print(f"Average round4_score (matched): {avg_round4_matched:.2f}")
    print(f"Average round5_score (overall): {avg_round5_overall:.2f}")
    print(f"Average round5_score (matched): {avg_round5_matched:.2f}")
    print(f"Average round6_score (overall): {avg_round6_overall:.2f}")
    print(f"Average round6_score (matched): {avg_round6_matched:.2f}")
    print(f"Average round7_score (overall): {avg_round7_overall:.2f}")
    print(f"Average round7_score (matched): {avg_round7_matched:.2f}")
    print()
    
    # Return structured dictionary instead of tuple
    return {
        'summary': {
            'total_dialogues': total_dialogues,
            'matched_dialogues': matched_dialogues_count,
            'match_rate': matched_dialogues_count/total_dialogues*100 if total_dialogues > 0 else 0,
            'round_matches': {
                1: len(round1_dialogues),
                2: len(round2_dialogues),
                3: len(round3_dialogues),
                4: len(round4_dialogues),
                5: len(round5_dialogues),
                6: len(round6_dialogues),
                7: len(round7_dialogues)
            },
            'max_dialogue_distance_for_overlap': max_dialogue_distance_for_overlap
        },
        'word_statistics': {
            'dialogue_words': {
                'total': total_dialogue_words,
                'unmatched': unmatched_dialogue_words,
                'matched': total_dialogue_words - unmatched_dialogue_words,
                'match_rate': dialogue_match_rate,
                'avg_unmatched_length_normalized': avg_unmatched_dialogue_length
            },
            'transcript_words': {
                'total': total_transcript_words,
                'unmatched': unmatched_transcript_words,
                'matched': total_transcript_words - unmatched_transcript_words,
                'match_rate': transcript_match_rate
            }
        },
        'ordering_mistakes': {
            'total': total_ordering_mistakes,
            'round1': ordering_mistakes_round1,
            'round2': ordering_mistakes_round2,
            'round3': ordering_mistakes_round3,
            'round4': ordering_mistakes_round4,
            'round5': ordering_mistakes_round5,
            'round6': ordering_mistakes_round6,
            'round7': ordering_mistakes_round7
        },
        'boundary_mistakes': {
            'total': total_boundary_mistakes,
            'round1': boundary_mistakes_round1,
            'round2': boundary_mistakes_round2,
            'round3': boundary_mistakes_round3,
            'round4': boundary_mistakes_round4,
            'round5': boundary_mistakes_round5,
            'round6': boundary_mistakes_round6,
            'round7': boundary_mistakes_round7
        },
        'transcript_entry_mistakes': {
            'missing_start_entries': len(missing_start_entries),
            'missing_end_entries': len(missing_end_entries)
        },
        'total_mistakes': {
            'total': total_mistakes,
            'round1': mistakes_round1,
            'round2': mistakes_round2,
            'round3': mistakes_round3,
            'round4': mistakes_round4,
            'round5': mistakes_round5,
            'round6': mistakes_round6,
            'round7': mistakes_round7
        },
        'scores': {
            'overall': {
                'round1': avg_round1_overall,
                'round2': avg_round2_overall,
                'round3': avg_round3_overall,
                'round4': avg_round4_overall,
                'round5': avg_round5_overall,
                'round6': avg_round6_overall,
                'round7': avg_round7_overall
            },
            'matched': {
                'round1': avg_round1_matched,
                'round2': avg_round2_matched,
                'round3': avg_round3_matched,
                'round4': avg_round4_matched,
                'round5': avg_round5_matched,
                'round6': avg_round6_matched,
                'round7': avg_round7_matched
            }
        },
        'mistake_details': {
            'ordering': all_ordering_mistakes,
            'boundary': all_boundary_mistakes,
            'missing_start_entries': missing_start_entries,
            'missing_end_entries': missing_end_entries
        }
    }

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
        for word_idx, word in enumerate(row['words_per_tr']):
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
    if (transcript_data[row_idx]['dialogue_per_tr'][word_idx] == 0):
        transcript_data[row_idx]['dialogue_per_tr'][word_idx] = dialogue_id
    elif move_to_next_if_taken:
        print(f'move_to_next_if_taken: dialogue_id: {dialogue_id} row_idx: {row_idx}, word_idx: {word_idx}')
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
        elif transcript_data[set_row_idx]['dialogue_per_tr'][set_word_idx] != 0:
            able_to_set = False
            return transcript_data, able_to_set
        else:
            transcript_data[set_row_idx]['dialogue_per_tr'][set_word_idx] = dialogue_id
    return transcript_data, able_to_set
#from(inclusive) and to(non inclusive)
def try_squeeze_in_dialogue(transcript_data, dialogues_list, dialogue_id):
    fixed_dialogue = False
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
    

    #strategy 2: If 1-2 empty row, then squeeze in the dialogue
    if not fixed_dialogue: 
        empty_trs = get_empty_trs_in_range(transcript_data, from_row, to_row)
        if len(empty_trs) < 5 and len(empty_trs) > 0:
            fixed_dialogue = True
            fill_empty_tr_with_dialogue(transcript_data, dialogue_id, empty_trs[0])
        print('search_words', search_words)    
        #print('empty_trs', empty_trs)
    #gap_available = words_collected - prev_dialogue_length
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

def add_dialogues_to_transcript_v2(transcript_data_orig, dialogue_list, min_length, min_match, round, match_to_full_text=True):
    """
    Enhanced dialogue matching that processes larger dialogues first and matches against the transcript.
    
    Args:
        transcript_data_orig (list): Array of transcript objects from load_transcript_tsv
        dialogue_list (list): List of dialogues from get_dialogue_list output
        min_length (int): Minimum length of normalized_text to process
        min_match (int): Minimum match score required for a successful match
        round (int): Round number (1, 2, or 3) for tracking matched rounds and scores
        match_to_full_text (bool): If True, match against full text. If False, match within boundaries
        
    Returns:
        tuple: (enhanced_transcript_data, updated_dialogue_list, positional_conflicts_count)
    """
    
    # Assert round is valid
    assert round in [1, 2, 3, 4, 5, 6, 7], f"Round must be 1, 2, or 3, got {round}"
    
    # Initialize dialogue_per_tr for all rows
    for row in transcript_data_orig:
        if 'dialogue_per_tr' not in row:
            row['dialogue_per_tr'] = [0] * len(row['words_per_tr'])
        # Assert that words_per_tr and dialogue_per_tr have same length
        assert len(row['words_per_tr']) == len(row['dialogue_per_tr']), \
            f"Length mismatch: words_per_tr={len(row['words_per_tr'])}, dialogue_per_tr={len(row['dialogue_per_tr'])}"
    
    # Create full text from transcript data
    full_text_words = []
    text_to_position_map = []  # Maps text index to (row_idx, word_idx)
    
    for row_idx, row in enumerate(transcript_data_orig):
        for word_idx, word in enumerate(row['words_per_tr']):
            full_text_words.append(word)
            text_to_position_map.append((row_idx, word_idx))
    
    # Normalize the full text for matching
    normalized_full_text = [normalize_and_clean_word(word) for word in full_text_words]
    
    # Get round-specific field names
    score_field = f'round{round}_score'
    matched_field = f'round{round}_matched'
    
    # Filter dialogues: only process those that haven't been matched in any previous round
    # and meet the minimum length requirement
    def is_already_matched(dialogue):
        return dialogue.get('matched_text_index_start', -1) != -1
    
    eligible_dialogues = [d for d in dialogue_list 
                         if d['length_normalized_text'] >= min_length and not is_already_matched(d)]
    sorted_dialogues = sorted(eligible_dialogues, key=lambda x: x['length_normalized_text'], reverse=True)
    
    matched_count = 0
    positional_conflicts_count = 0  # Track positional conflicts
    total_eligible = len(sorted_dialogues)
    
    print(f"Round {round}: Processing {total_eligible} unmatched dialogues with min_length={min_length}, min_match={min_match}")
    print(f"Match to full text: {match_to_full_text}")
    
    def get_matched_dialogue_boundaries(dialogue_id):
        """Find nearest lower and upper matched dialogues to determine boundary text"""
        matched_dialogues = [d for d in dialogue_list if is_already_matched(d)]
        
        # Find nearest lower dialogue (highest ID that is lower than dialogue_id)
        lower_dialogue = None
        for d in matched_dialogues:
            if d['id'] < dialogue_id:
                if lower_dialogue is None or d['id'] > lower_dialogue['id']:
                    lower_dialogue = d
        
        # Find nearest upper dialogue (lowest ID that is higher than dialogue_id)
        upper_dialogue = None
        for d in matched_dialogues:
            if d['id'] > dialogue_id:
                if upper_dialogue is None or d['id'] < upper_dialogue['id']:
                    upper_dialogue = d
        
        # Determine boundary start and end
        if lower_dialogue is not None:
            boundary_start = lower_dialogue.get('matched_text_index_end', -1) + 1
        else:
            boundary_start = 0
            
        if upper_dialogue is not None:
            boundary_end = upper_dialogue.get('matched_text_index_start', -1) - 1
        else:
            boundary_end = len(normalized_full_text) - 1
            
        # Ensure valid boundaries
        boundary_start = max(0, boundary_start)
        boundary_end = min(len(normalized_full_text) - 1, boundary_end)
        
        if boundary_start > boundary_end:
            # No valid boundary found
            return None, None, None, None
            
        return boundary_start, boundary_end, lower_dialogue, upper_dialogue
    
    for dialogue in sorted_dialogues:
        dialogue_id = dialogue['id']
        dialogue_text = dialogue['text']
        normalized_dialogue_words = dialogue['normalized_text']
        
        if not normalized_dialogue_words:
            continue  # Skip if no valid words after normalization
        
        print(f"Processing dialogue {dialogue_id} (length: {len(normalized_dialogue_words)}): '{dialogue_text}'")
        
        # Determine search text and positions based on match_to_full_text
        if match_to_full_text:
            search_text = normalized_full_text
            search_start_offset = 0
            print(f"  Matching against full text ({len(search_text)} words)")
        else:
            boundary_start, boundary_end, lower_dialogue, upper_dialogue = get_matched_dialogue_boundaries(dialogue_id)
            
            if boundary_start is None or boundary_end is None:
                print(f"  ✗ No valid boundary found for dialogue {dialogue_id}")
                dialogue[score_field] = 0
                continue
                
            search_text = normalized_full_text[boundary_start:boundary_end + 1]
            search_start_offset = boundary_start
            
            print(f"  Matching within boundary [{boundary_start}:{boundary_end}] ({len(search_text)} words)")
            if lower_dialogue:
                print(f"    Lower boundary: dialogue {lower_dialogue['id']} ends at {lower_dialogue.get('matched_text_index_end', -1)}")
            if upper_dialogue:
                print(f"    Upper boundary: dialogue {upper_dialogue['id']} starts at {upper_dialogue.get('matched_text_index_start', -1)}")
        
        if not search_text:
            print(f"  ✗ Empty search text for dialogue {dialogue_id}")
            dialogue[score_field] = 0
            continue
        
        # Use fuzzy matching to find best match in the search text
        best_match = best_variable_fuzzy_match(normalized_dialogue_words, search_text)
        match_len = best_match['end_word'] - best_match['start_word'] + 1
        
        # Adjust match indices to account for boundary offset
        adjusted_start_word = best_match['start_word'] + search_start_offset
        adjusted_end_word = best_match['end_word'] + search_start_offset
        
        print(f'  match_len: {match_len}')
        print(f'  start_word: {adjusted_start_word} (local: {best_match["start_word"]})')
        print(f'  end_word: {adjusted_end_word} (local: {best_match["end_word"]})')
        
        # Update dialogue with round-specific score
        dialogue[score_field] = best_match['match_score']
        
        print(f"  Best match score: {best_match['match_score']}")
        print(f"  Matched text: '{best_match['matched_text']}'")
        
        # Check if match meets minimum threshold
        if best_match['match_score'] >= min_match:
            # Update dialogue with match information
            dialogue['matched_text_index_start'] = adjusted_start_word
            dialogue['matched_text_index_end'] = adjusted_end_word
            dialogue[matched_field] = True
            
            # Get row and word indices from the match
            start_row, start_word = text_to_position_map[adjusted_start_word]
            end_row, end_word = text_to_position_map[adjusted_end_word]
            
            dialogue['matched_row_index_start'] = start_row
            dialogue['matched_word_index_start'] = start_word
            dialogue['matched_row_index_end'] = end_row
            dialogue['matched_word_index_end'] = end_word
            
            # Check if the position is already taken
            if transcript_data_orig[start_row]['dialogue_per_tr'][start_word] == 0:
                # Mark the first word of the dialogue with dialogue_id
                transcript_data_orig[start_row]['dialogue_per_tr'][start_word] = dialogue_id
                
                # Mark the last word of the dialogue with -dialogue_id (only if it's different from start)
                if end_row != start_row or end_word != start_word:
                    if transcript_data_orig[end_row]['dialogue_per_tr'][end_word] == 0:
                        transcript_data_orig[end_row]['dialogue_per_tr'][end_word] = -dialogue_id
                    else:
                        print(f"  ⚠ End position already taken for dialogue {dialogue_id}")
                
                matched_count += 1
                
                print(f"  ✓ Matched dialogue {dialogue_id} at row {start_row}, word {start_word}")
                print(f"    Spans from row {start_row} word {start_word} to row {end_row} word {end_word}")
            else:
                # Count start position conflict
                existing_dialogue_id = transcript_data_orig[start_row]['dialogue_per_tr'][start_word]
                print(f"  ✗ Position already taken by dialogue {existing_dialogue_id}. Not able to set dialogue {dialogue_id}")
                
                # If it's an end marker (negative), overwrite it
                if existing_dialogue_id < 0:
                    transcript_data_orig[start_row]['dialogue_per_tr'][start_word] = dialogue_id
                    print(f"  ✓ Overwrote end position with dialogue {dialogue_id}")
                    matched_count += 1
                    print(f"  ✓ Matched dialogue {dialogue_id} at row {start_row}, word {start_word}")
                    print(f"    Spans from row {start_row} word {start_word} to row {end_row} word {end_word}")
                else:
                    print(f"  ✗ Position already taken and not able to overwrite it for dialogue {dialogue_id}")
                    positional_conflicts_count += 1
        else:
            print(f"  ✗ Match score {best_match['match_score']} below threshold {min_match}")
    
    # Final assertion check for all rows
    for row in transcript_data_orig:
        assert len(row['words_per_tr']) == len(row['dialogue_per_tr']), \
            f"Final length mismatch: words_per_tr={len(row['words_per_tr'])}, dialogue_per_tr={len(row['dialogue_per_tr'])}"
    
    print(f"\nRound {round} Summary:")
    print(f"  Total eligible dialogues: {total_eligible}")
    print(f"  Successfully matched: {matched_count}")
    print(f"  Positional conflicts encountered: {positional_conflicts_count}")
    print(f"  Match rate: {matched_count/total_eligible*100:.1f}%" if total_eligible > 0 else "  Match rate: 0%")
    
    return transcript_data_orig, dialogue_list, positional_conflicts_count

def add_dialogues_to_transcript_phase2(transcript_data, dialogue_list):
    """
    Phase 2 dialogue matching that tries to match unmatched dialogues to empty rows
    based on their position relative to matched dialogues in the same scene.
    
    Args:
        transcript_data (list): Array of transcript objects from load_transcript_tsv
        dialogue_list (list): List of dialogues from get_dialogue_list output
        
    Returns:
        tuple: (updated_transcript_data, updated_dialogue_list, matched_count)
    """
    
    def is_already_matched(dialogue):
        return dialogue.get('matched_text_index_start', -1) != -1
    
    def get_matched_dialogues_in_scene(scene_id):
        """Get all matched dialogues in a specific scene, sorted by in_scene_index"""
        matched_in_scene = [d for d in dialogue_list 
                           if d['scene_id'] == scene_id and is_already_matched(d)]
        return sorted(matched_in_scene, key=lambda x: x['in_scene_index'])
    
    def find_last_matched_dialogue_in_scene(scene_id, before_in_scene_index):
        """Find the last matched dialogue in the scene before the given in_scene_index"""
        matched_in_scene = get_matched_dialogues_in_scene(scene_id)
        candidates = [d for d in matched_in_scene if d['in_scene_index'] < before_in_scene_index]
        return candidates[-1] if candidates else None
    
    def find_first_matched_dialogue_in_scene(scene_id, after_in_scene_index):
        """Find the first matched dialogue in the scene after the given in_scene_index"""
        matched_in_scene = get_matched_dialogues_in_scene(scene_id)
        candidates = [d for d in matched_in_scene if d['in_scene_index'] > after_in_scene_index]
        return candidates[0] if candidates else None
    
    def has_empty_words_per_tr(row_idx):
        """Check if a row has empty words_per_tr"""
        if row_idx < 0 or row_idx >= len(transcript_data):
            return False
        return len(transcript_data[row_idx]['words_per_tr']) == 0
    
    def update_dialogue_match_info(dialogue, row_idx):
        """Update dialogue with match information for the given row"""
        # Create text to position mapping to get text indices
        text_index = 0
        for r_idx in range(row_idx):
            text_index += len(transcript_data[r_idx]['words_per_tr'])
        
        dialogue['matched_text_index_start'] = text_index
        dialogue['matched_text_index_end'] = text_index  # Same as start for empty row
        dialogue['matched_row_index_start'] = row_idx
        dialogue['matched_word_index_start'] = 0
        dialogue['matched_row_index_end'] = row_idx
        dialogue['matched_word_index_end'] = 0
        # Mark as matched in phase2
        dialogue['phase2_matched'] = True
    
    # Get unmatched dialogues
    unmatched_dialogues = [d for d in dialogue_list if not is_already_matched(d)]
    matched_count = 0
    
    print(f"Phase 2: Processing {len(unmatched_dialogues)} unmatched dialogues")
    
    for dialogue in unmatched_dialogues:
        dialogue_id = dialogue['id']
        scene_id = dialogue['scene_id']
        in_scene_index = dialogue['in_scene_index']
        
        print(f"Processing dialogue {dialogue_id} (scene {scene_id}, index {in_scene_index}): '{dialogue['text']}'")
        
        target_row = None
        
        # Check if this is the first dialogue in the scene (in_scene_index == 1)
        if in_scene_index == 1:
            # Find the first matched dialogue in the scene
            first_matched = find_first_matched_dialogue_in_scene(scene_id, in_scene_index)
            if first_matched:
                first_matched_row = first_matched.get('matched_row_index_start', -1)
                if first_matched_row > 0 and has_empty_words_per_tr(first_matched_row - 1):
                    target_row = first_matched_row - 1
                    print(f"  Found empty row {target_row} before first matched dialogue {first_matched['id']}")
                else:
                    print(f"  Row before first matched dialogue {first_matched['id']} is not empty")
            else:
                print(f"  No matched dialogue found after this dialogue in scene {scene_id}")
        else:
            # Find the last matched dialogue in the scene before this dialogue
            last_matched = find_last_matched_dialogue_in_scene(scene_id, in_scene_index)
            if last_matched:
                last_matched_row = last_matched.get('matched_row_index_end', -1)
                if last_matched_row >= 0 and last_matched_row + 1 < len(transcript_data) and has_empty_words_per_tr(last_matched_row + 1):
                    target_row = last_matched_row + 1
                    print(f"  Found empty row {target_row} after last matched dialogue {last_matched['id']}")
                else:
                    print(f"  Row after last matched dialogue {last_matched['id']} is not empty")
            else:
                print(f"  No matched dialogue found before this dialogue in scene {scene_id}")
        
        # If we found a target row, match the dialogue
        if target_row is not None:
            fill_empty_tr_with_dialogue(transcript_data, dialogue_id, target_row)
            update_dialogue_match_info(dialogue, target_row)
            matched_count += 1
            print(f"  ✓ Matched dialogue {dialogue_id} to row {target_row}")
        else:
            print(f"  ✗ Could not find suitable empty row for dialogue {dialogue_id}")
    
    print(f"\nPhase 2 Summary:")
    print(f"  Successfully matched: {matched_count}")
    print(f"  Match rate: {matched_count/len(unmatched_dialogues)*100:.1f}%" if unmatched_dialogues else "  Match rate: 0%")
    
    return transcript_data, dialogue_list, matched_count

def enhance_transcripts(transcript_data, dialogues_file, run_phase_2=True):
    dialogues = get_scene_dialogue(dialogues_file)
    still_skipped_dialogues = []
    transcript_data_enhanced, skipped_dialogues, total_dialogues = add_dialogues_to_transcript_v2(transcript_data, dialogues, 20, 80, 1)
    print(f"Skipped dialogues after phase 1:")
    for dialogue in skipped_dialogues:
        print(f"  {dialogue['id']}: {dialogue['text']}")
    num_skipped_p1 = len(skipped_dialogues)

    if run_phase_2:
        print(f"Phase 2:")
        counter = 0
        #sort dialogues by length of text in descending order
        sorted_dialogues = sorted(skipped_dialogues, key=lambda x: len(x['text']), reverse=True)
        for dialogue in sorted_dialogues:
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
        num_skipped_p2 = len(still_skipped_dialogues)

    return transcript_data_enhanced, (total_dialogues, num_skipped_p1, num_skipped_p2)

def enhance_transcripts_v2(transcript_data, dialogues_file, run_phase_2=True):
    dialogues = get_scene_dialogue(dialogues_file)
    dialogue_list = get_dialogue_list(dialogues)
    transcript_data_enhanced, dialogue_list, positional_conflicts_count = add_dialogues_to_transcript_v2(transcript_data, dialogue_list, 8, 80, 1, match_to_full_text=True)
    #round 2
    transcript_data_enhanced, dialogue_list, positional_conflicts_count = add_dialogues_to_transcript_v2(transcript_data_enhanced, dialogue_list, 5, 80, 2, match_to_full_text=False)
    #round 3
    transcript_data_enhanced, dialogue_list, positional_conflicts_count = add_dialogues_to_transcript_v2(transcript_data_enhanced, dialogue_list, 4, 80, 3, match_to_full_text=False)
    transcript_data_enhanced, dialogue_list, positional_conflicts_count = add_dialogues_to_transcript_v2(transcript_data_enhanced, dialogue_list, 3, 80, 4, match_to_full_text=False)
    transcript_data_enhanced, dialogue_list, positional_conflicts_count = add_dialogues_to_transcript_v2(transcript_data_enhanced, dialogue_list, 2, 80, 5, match_to_full_text=False)
    transcript_data_enhanced, dialogue_list, positional_conflicts_count = add_dialogues_to_transcript_v2(transcript_data_enhanced, dialogue_list, 1, 80, 6, match_to_full_text=False)
    transcript_data_enhanced, dialogue_list, positional_conflicts_count = add_dialogues_to_transcript_v2(transcript_data_enhanced, dialogue_list, 4, 60, 7, match_to_full_text=False)
    transcript_data_enhanced, dialogue_list, positional_conflicts_count = add_dialogues_to_transcript_v2(transcript_data_enhanced, dialogue_list, 2, 60, 7, match_to_full_text=False)
    transcript_data_enhanced, dialogue_list, positional_conflicts_count = add_dialogues_to_transcript_v2(transcript_data_enhanced, dialogue_list, 1, 60, 7, match_to_full_text=False)
    transcript_data_enhanced, dialogue_list, positional_conflicts_count = add_dialogues_to_transcript_v2(transcript_data_enhanced, dialogue_list, 3, 40, 7, match_to_full_text=False)
    transcript_data_enhanced, dialogue_list, positional_conflicts_count = add_dialogues_to_transcript_v2(transcript_data_enhanced, dialogue_list, 1, 40, 7, match_to_full_text=False)
    print(f"Positional conflicts: {positional_conflicts_count}")

    
    
    def is_already_matched(dialogue):
        return dialogue.get('matched_text_index_start', -1) != -1
    un_matched_dialogues = [d for d in dialogue_list if not is_already_matched(d)]

    num_un_matched_dialogues_p1 = len(un_matched_dialogues)
    num_un_matched_dialogues_p2 = num_un_matched_dialogues_p1

    

    if run_phase_2:
        print(f"Phase 2:")
        transcript_data_enhanced, dialogue_list, matched_count = add_dialogues_to_transcript_phase2(transcript_data_enhanced, dialogue_list)

    un_matched_dialogues = [d for d in dialogue_list if not is_already_matched(d)]
    for dialogue in un_matched_dialogues:
        print(f"  {dialogue['id']} {dialogue['in_scene_index']} {dialogue['total_dialogues_scene']}: {dialogue['text'] } (not matched)")

    num_un_matched_dialogues_p2 = len(un_matched_dialogues)
    validation_results = check_dialogue_list_validity(dialogue_list, transcript_data_enhanced, max_dialogue_distance_for_overlap=2)
    
    
    print(f"Total mistakes: {validation_results['total_mistakes']['total']}")
    print(f"Total boundary mistakes: {validation_results['boundary_mistakes']['total']}")
    print(f"Missing start entries: {validation_results['transcript_entry_mistakes']['missing_start_entries']}")
    print(f"Missing end entries: {validation_results['transcript_entry_mistakes']['missing_end_entries']}")
    print(f"Dialogue words: {validation_results['word_statistics']['dialogue_words']}")
    print(f"Transcript words: {validation_results['word_statistics']['transcript_words']}")

    return transcript_data_enhanced, (len(dialogue_list), num_un_matched_dialogues_p1, num_un_matched_dialogues_p2, validation_results['word_statistics']['dialogue_words']['match_rate'], validation_results['word_statistics']['transcript_words']['match_rate'])

