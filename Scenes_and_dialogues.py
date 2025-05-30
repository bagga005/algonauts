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

