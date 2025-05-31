import requests
from lxml import html
import os
import utils
import transcripts_enhancer
from glob import glob
import html as ihtml
import statistics
from collections import Counter
from Scenes_and_dialogues import get_scene_dialogue

def print_dialogue_text(stim_path):
    dialogues = get_scene_dialogue(stim_path)
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
                #print('separated out a scene:', bracket_content)
                
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
    exclude_list = ['s02']
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
        #print(f"Loading {file}")
        dialogues = get_scene_dialogue(file)

def check_scenes_all_text_files(print_max=None):
    root_data_dir = utils.get_data_root_dir()
    files = glob(f"{root_data_dir}/stimuli/transcripts/friends/full/*.txt")
    
    all_dialogue_counts = []  # List to store dialogue count for each scene
    episode_stats = {}  # Dictionary to store per-episode statistics
    large_scenes = []  # List to store scenes with more than print_max dialogues
    
    total_scenes = 0
    total_episodes = len(files)
    
    print(f"Analyzing {total_episodes} episodes...")
    if print_max is not None:
        print(f"Will highlight scenes with more than {print_max} dialogues")
    print("="*80)
    
    for file in files:
        filename = file.split("/")[-1].split(".")[0]  # Extract episode name
        print(f"Processing {filename}")
        
        try:
            dialogues = get_scene_dialogue(file)
            episode_dialogue_counts = []
            
            for scene_idx, scene in enumerate(dialogues['scenes']):
                len_dialogues = len(scene['dialogues'])
                all_dialogue_counts.append(len_dialogues)
                episode_dialogue_counts.append(len_dialogues)
                total_scenes += 1
                
                # Check if scene exceeds print_max threshold
                if print_max is not None and len_dialogues > print_max:
                    scene_info = {
                        'file': filename,
                        'scene_id': scene['id'],
                        'scene_desc': scene['desc'],
                        'dialogue_count': len_dialogues,
                        'scene_index': scene_idx + 1
                    }
                    large_scenes.append(scene_info)
                    print(f"  🔥 LARGE SCENE: Scene {scene['id']} ({len_dialogues} dialogues) - '{scene['desc']}'")
            
            # Store episode-level statistics
            if episode_dialogue_counts:
                episode_stats[filename] = {
                    'scenes': len(episode_dialogue_counts),
                    'total_dialogues': sum(episode_dialogue_counts),
                    'avg_dialogues_per_scene': statistics.mean(episode_dialogue_counts),
                    'min_dialogues': min(episode_dialogue_counts),
                    'max_dialogues': max(episode_dialogue_counts)
                }
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue
    
    if not all_dialogue_counts:
        print("No valid data found!")
        return
    
    # Print large scenes summary if print_max was specified
    if print_max is not None and large_scenes:
        print("\n" + "="*80)
        print(f"SCENES WITH MORE THAN {print_max} DIALOGUES")
        print("="*80)
        print(f"Found {len(large_scenes)} scenes exceeding threshold:")
        print(f"{'Episode':<20} {'Scene ID':<10} {'Dialogues':<10} {'Description':<50}")
        print("-" * 95)
        
        # Sort by dialogue count (descending)
        large_scenes.sort(key=lambda x: x['dialogue_count'], reverse=True)
        
        for scene in large_scenes:
            desc_truncated = scene['scene_desc'][:47] + "..." if len(scene['scene_desc']) > 50 else scene['scene_desc']
            print(f"{scene['file']:<20} {scene['scene_id']:<10} {scene['dialogue_count']:<10} {desc_truncated:<50}")
    
    elif print_max is not None:
        print(f"\n✅ No scenes found with more than {print_max} dialogues")
    
    # Calculate overall statistics
    dialogue_distribution = Counter(all_dialogue_counts)
    max_dialogues = max(all_dialogue_counts)
    min_dialogues = min(all_dialogue_counts)
    avg_dialogues = statistics.mean(all_dialogue_counts)
    std_dialogues = statistics.stdev(all_dialogue_counts) if len(all_dialogue_counts) > 1 else 0
    median_dialogues = statistics.median(all_dialogue_counts)
    
    print("\n" + "="*80)
    print("DIALOGUE DISTRIBUTION BY SCENE")
    print("="*80)
    
    # Show distribution table
    print(f"{'Dialogues':<12} {'Scenes':<8} {'Percentage':<12} {'Cumulative %':<15}")
    print("-" * 50)
    
    cumulative_scenes = 0
    for dialogue_count in range(min_dialogues, max_dialogues + 1):
        scenes_with_count = dialogue_distribution.get(dialogue_count, 0)
        if scenes_with_count > 0:
            percentage = (scenes_with_count / total_scenes) * 100
            cumulative_scenes += scenes_with_count
            cumulative_percentage = (cumulative_scenes / total_scenes) * 100
            highlight = " 🔥" if print_max is not None and dialogue_count > print_max else ""
            print(f"{dialogue_count:<12} {scenes_with_count:<8} {percentage:<11.1f}% {cumulative_percentage:<14.1f}%{highlight}")
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total episodes analyzed: {total_episodes}")
    print(f"Total scenes: {total_scenes}")
    print(f"Total dialogues: {sum(all_dialogue_counts)}")
    print(f"Average dialogues per scene: {avg_dialogues:.2f}")
    print(f"Standard deviation: {std_dialogues:.2f}")
    print(f"Median dialogues per scene: {median_dialogues:.1f}")
    print(f"Minimum dialogues in a scene: {min_dialogues}")
    print(f"Maximum dialogues in a scene: {max_dialogues}")
    
    if print_max is not None:
        scenes_above_threshold = sum(1 for count in all_dialogue_counts if count > print_max)
        percentage_above = (scenes_above_threshold / total_scenes) * 100
        print(f"Scenes with > {print_max} dialogues: {scenes_above_threshold} ({percentage_above:.1f}%)")
    
    # Show most common dialogue counts
    print(f"\nMost common dialogue counts:")
    for count, frequency in dialogue_distribution.most_common(10):
        percentage = (frequency / total_scenes) * 100
        highlight = " 🔥" if print_max is not None and count > print_max else ""
        print(f"  {count} dialogues: {frequency} scenes ({percentage:.1f}%){highlight}")
    
    # Show episodes with interesting statistics
    print("\n" + "="*80)
    print("EPISODE HIGHLIGHTS")
    print("="*80)
    
    # Episode with most scenes
    most_scenes_episode = max(episode_stats.items(), key=lambda x: x[1]['scenes'])
    print(f"Most scenes: {most_scenes_episode[0]} ({most_scenes_episode[1]['scenes']} scenes)")
    
    # Episode with highest average dialogues per scene
    highest_avg_episode = max(episode_stats.items(), key=lambda x: x[1]['avg_dialogues_per_scene'])
    print(f"Highest avg dialogues/scene: {highest_avg_episode[0]} ({highest_avg_episode[1]['avg_dialogues_per_scene']:.1f} dialogues/scene)")
    
    # Episode with most total dialogues
    most_dialogues_episode = max(episode_stats.items(), key=lambda x: x[1]['total_dialogues'])
    print(f"Most total dialogues: {most_dialogues_episode[0]} ({most_dialogues_episode[1]['total_dialogues']} dialogues)")
    
    # Scene with maximum dialogues
    max_scene_count = max(all_dialogue_counts)
    print(f"Scene with most dialogues: {max_scene_count} dialogues")
    
    print("\n" + "="*80)
    print("DISTRIBUTION PERCENTILES")
    print("="*80)
    
    # Calculate percentiles
    sorted_counts = sorted(all_dialogue_counts)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    
    for p in percentiles:
        index = int((p / 100) * len(sorted_counts))
        if index >= len(sorted_counts):
            index = len(sorted_counts) - 1
        percentile_value = sorted_counts[index]
        highlight = " 🔥" if print_max is not None and percentile_value > print_max else ""
        print(f"{p}th percentile: {percentile_value} dialogues{highlight}")
    
    return {
        'total_episodes': total_episodes,
        'total_scenes': total_scenes,
        'distribution': dict(dialogue_distribution),
        'large_scenes': large_scenes if print_max is not None else None,
        'statistics': {
            'mean': avg_dialogues,
            'std': std_dialogues,
            'median': median_dialogues,
            'min': min_dialogues,
            'max': max_dialogues
        },
        'episode_stats': episode_stats
    }

def clean_up_all_text_files():
    root_data_dir = utils.get_data_root_dir()
    files = glob(f"{root_data_dir}/stimuli/transcripts/friends/full/*.txt")
    for file in files:
        print(f"Cleaning up {file}")
        clean_up_dialogue_text(file)

def check_scene_brackets_all_text_files():
    root_data_dir = utils.get_data_root_dir()
    files = glob(f"{root_data_dir}/stimuli/transcripts/friends/full/*.txt")
    counter =0
    for file in files:
        #print(f"Checking {file}")
        counter += find_non_scene_brackets(file)
    print(f"Total non-scene brackets: {counter}")

def checkForSubstituion(s_and_e):
    if s_and_e == '0615':
        return '0615-0616'
    if s_and_e == '0212':
        return '0212-0213'
    return s_and_e

def get_text_with_newlines(element):
    parts = []
    if element.text:
        parts.append(element.text)
    for child in element:
        if child.tag == 'br':
            parts.append('\n')
        if child.tail:
            parts.append(child.tail)
    return ''.join(parts).strip()

def download_raw_text_for_all_episodes():
    root_data_dir = utils.get_data_root_dir()
    file_in_filter = 'friends_s02e04'
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
        url_suffix = checkForSubstituion(s_and_e)

        url = f'https://edersoncorbari.github.io/friends-scripts/season/{url_suffix}.html'
        print(url)
        response = requests.get(url)
        response.encoding = response.apparent_encoding
        #response.encoding = 'windows-1252'
        tree = html.fromstring(response.text)
        paragraphs = tree.xpath('//p')

        print(f"Found {len(paragraphs)} paragraphs")
        if len(paragraphs) < 10:
            text = get_text_with_newlines(paragraphs[1])
            print(text)
            print(f"Skipping {s_and_e} because it has {len(paragraphs)} paragraphs")
            for p in paragraphs:
                print(len(p.text_content()))
        else:
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

def find_non_scene_brackets(file_path):
    """
    Reads a file and finds lines with square brackets that don't contain 'scene'.
    
    Args:
        file_path (str): Path to the text file to analyze
    
    Returns:
        int: Total number of exceptions found
    """
    exception_counter = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        for line_num, line in enumerate(lines, 1):
            stripped_line = line.strip()
            
            # Check if line contains square brackets
            if '[' in stripped_line and ']' in stripped_line:
                # Find all bracket pairs in the line
                start_pos = 0
                while True:
                    bracket_start = stripped_line.find('[', start_pos)
                    if bracket_start == -1:
                        break
                    
                    bracket_end = stripped_line.find(']', bracket_start)
                    if bracket_end == -1:
                        break
                    
                    # Extract text within brackets
                    bracket_content = stripped_line[bracket_start + 1:bracket_end]
                    
                    # Check if 'scene' is NOT in the bracket content (case-insensitive)
                    if 'scene' not in bracket_content.lower():
                        if exception_counter == 0:
                            print(f"File: {file_path}")
                        print(f"Line {line_num}: {stripped_line}")
                        exception_counter += 1
                        break  # Only count once per line, even if multiple bracket pairs
                    
                    start_pos = bracket_end + 1
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return 0
    except Exception as e:
        print(f"Error reading file: {e}")
        return 0
    
    return exception_counter

if __name__ == "__main__":
    #download_raw_text_for_all_episodes()
    #clean_up_all_text_files()
    check_scenes_all_text_files(print_max=90)
    #test_loading_text_files()
    #check_scene_brackets_all_text_files()


