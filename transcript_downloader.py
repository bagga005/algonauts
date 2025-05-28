import requests
from lxml import html
import os
import utils
import transcripts_enhancer
from glob import glob
import html as ihtml

def print_dialogue_text(stim_path):
    dialogues = transcripts_enhancer.get_scene_dialogue(stim_path)
    print(len(dialogues['scenes']))
    for scene in dialogues['scenes']:
        print(f'scene: {scene["desc"]}')
        for dialogue in scene['dialogues']:
            print('-'*100)
            print(f'dialogue: {dialogue["text"]}')    
        print('*'*100)

def clean_up_dialogue_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    processed_lines = []
    
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            processed_lines.append('')
            continue

        stripped_line_lower = stripped_line.lower()
        if 'written by:' in stripped_line_lower or \
            'teleplay by:' in stripped_line_lower or \
            'story by:' in stripped_line_lower  or \
            "transcriber's note:" in stripped_line_lower or \
            "mmatting@indiana.edu" in stripped_line_lower or \
            "additions and adjustments" in stripped_line_lower:
            continue

        # Check if line has square brackets but doesn't start with one
        if '[' in stripped_line and ']' in stripped_line and not stripped_line.startswith('['):
            # Find the position of the opening bracket
            bracket_start = stripped_line.find('[')
            bracket_end = stripped_line.find(']', bracket_start)
            
            if bracket_end != -1:  # Found both brackets
                # Split the line: text before bracket, and bracket content
                before_bracket = stripped_line[:bracket_start].strip()
                bracket_content = stripped_line[bracket_start:bracket_end + 1]
                after_bracket = stripped_line[bracket_end + 1:].strip()
                
                # Add the part before bracket (if any)
                if before_bracket:
                    processed_lines.append(before_bracket)
                
                # Add the bracket content on its own line
                processed_lines.append(bracket_content)
                print('separated out a scene:', bracket_content)
                
                # Add the part after bracket (if any)
                if after_bracket:
                    processed_lines.append(after_bracket)
            else:
                # No closing bracket found, keep line as is
                processed_lines.append(stripped_line)
        else:
            # Line either starts with bracket or has no brackets, keep as is
            processed_lines.append(stripped_line)
    
    # Write the processed lines back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in processed_lines:
            file.write(line + '\n')
def test_loading_text_files():
    root_data_dir = utils.get_data_root_dir()
    files = glob(f"{root_data_dir}/stimuli/transcripts/friends/full/*.txt")
    exclude_list = ['s02', 's06e15']
    updated_files = []
    for file in files:
        exclude_found = False
        for exclude in exclude_list:
            if exclude in file:
                exclude_found = True
                break
        if not exclude_found:
            updated_files.append(file)
    files = updated_files
    files.sort()
    for file in files:
        print(f"Loading {file}")
        dialogues = transcripts_enhancer.get_scene_dialogue(file)

def clean_up_all_text_files():
    root_data_dir = utils.get_data_root_dir()
    files = glob(f"{root_data_dir}/stimuli/transcripts/friends/full/*.txt")
    for file in files:
        print(f"Cleaning up {file}")
        clean_up_dialogue_text(file)

def download_raw_text_for_all_episodes():
    root_data_dir = utils.get_data_root_dir()
    file_in_filter = ''
    exclude_list = []
    files = glob(f"{root_data_dir}/stimuli/transcripts/friends/s*/*.tsv")
    if file_in_filter:
        files = [f for f in files if file_in_filter in f]
    files.sort()

    stimuli = [f.split("/")[-1].split(".")[0] for f in files]

    s_and_e_list = [ f[9:11] + f[12:14] for f in stimuli ]
    unique_s_and_e_list = list(set(s_and_e_list))
    unique_s_and_e_list.sort(reverse=True)
    print(unique_s_and_e_list)

    counter =0

    for s_and_e in unique_s_and_e_list:
        url = f'https://edersoncorbari.github.io/friends-scripts/season/{s_and_e}.html'
        response = requests.get(url)
        response.encoding = response.apparent_encoding
        #response.encoding = 'windows-1252'
        tree = html.fromstring(response.text)
        paragraphs = tree.xpath('//p')
        text = '\n'.join([' '.join(p.itertext()).strip().replace('\n', ' ') for p in paragraphs])
        #text = ihtml.unescape(text) 
        # Fix Windows-1252 special chars (chr(146) = right single quote)
        WINDOWS_1252_MAP = {
            chr(130): '‚',   # single low-9 quotation mark
            chr(132): '„',   # double low-9 quotation mark
            chr(133): '…',   # ellipsis
            chr(145): '‘',   # left single quotation mark
            chr(146): '’',   # right single quotation mark
            chr(147): '“',   # left double quotation mark
            chr(148): '”',   # right double quotation mark
            chr(150): '–',   # en dash
            chr(151): '—',   # em dash
        }
        for bad, good in WINDOWS_1252_MAP.items():
            text = text.replace(bad, good)
        #text = tree.text_content()

        s_single = s_and_e[1]
        epi_full = s_and_e[:2] + 'e' + s_and_e[2:]
        stim_folder = os.path.join(root_data_dir, 'stimuli', 'transcripts', 'friends', 'full')
        if not os.path.exists(stim_folder):
            os.makedirs(stim_folder)
        stim_path = os.path.join(stim_folder, f'friends_s{epi_full}.txt')
        print(stim_path)
        with open(stim_path, 'w') as file:
            file.write(text)

        #do cleanup
        clean_up_dialogue_text(stim_path)
        counter += 1
        # if counter > 30:
        #     break

    # videos_iterator = enumerate(stimuli.items())
    # for i, (stim_id, stim_path) in videos_iterator:
    #         path_split = stim_path.split('/')

if __name__ == "__main__":
    #download_raw_text_for_all_episodes()
    #clean_up_all_text_files()
    test_loading_text_files()



