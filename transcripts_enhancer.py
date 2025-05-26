import json
import re
import os
import utils

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
                   if line.strip() and 'Commercial Break' not in line and 'Closing Credits' not in line]
    
    if not valid_lines:
        return {"scenes": []}
    
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


# Example usage and test
if __name__ == "__main__":

    # Write test file
    file_path = os.path.join(utils.get_data_root_dir(),'stimuli', 'transcripts', 'friends','s1','full','friends_s01e01.txt')
    
    # Test the function
    try:
        result = get_scene_dialogue(file_path)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
