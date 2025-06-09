import os
import json
import utils
from Scenes_and_dialogues import get_scene_dialogue, get_scene_and_dialogues_display_text, get_dialogue_list, get_scene_and_dialogues_display_text_till_scene, get_scene_and_dialogues_display_len
from glob import glob
from tqdm import tqdm
import transformers
import torch

def write_summary(file_path, scene_id, stim_id, summary, unsummarized_length):
    """
    Write summary to a file in JSON format.
    
    Args:
        file_path (str): Path to the JSON file
        scene_id (str/int): Scene identifier
        stim_id (str): Stimulus identifier
        summary (str): Summary text
        unsummarized_length (int): Length of the unsummarized text
    """
    # Check if file exists
    if os.path.exists(file_path):
        # Read existing data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Assert that scene_id is not already in existing entries
        existing_scene_ids = [entry['scene_id'] for entry in data]
        assert scene_id not in existing_scene_ids, f"Scene ID {scene_id} already exists in {file_path}"
        
        # Append new entry
        data.append({
            "scene_id": scene_id,
            "stim_id": stim_id,
            "summary": summary,
            "unsummarized_length": unsummarized_length
        })
    else:
        # Create new data with the entry
        data = [{
            "scene_id": scene_id,
            "stim_id": stim_id,
            "summary": summary,
            "unsummarized_length": unsummarized_length
        }]
    
    # Write data back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def test_summary_gen_all_episodes(min_length_for_summary=100):
    root_data_dir = utils.get_data_root_dir()
    
    #list of full text transcripts
    file_in_filter = ''
    exclude_list = []#['friends_s03e05b', 'friends_s03e06a']
    files = glob(f"{root_data_dir}/algonauts_2025.competitors/stimuli/transcripts/friends/full/*.txt")
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
    if file_in_filter:
        files = [f for f in files if file_in_filter in f]
    files.sort()

    stimuli = {f.split("/")[-1].split(".")[0]: f for f in files}
    print(len(stimuli), list(stimuli)[:3], list(stimuli)[-3:])
    total_l = 0
    more_than_1000 = 0
    trans_iterator = tqdm(enumerate(stimuli.items()), total=len(list(stimuli)))
    #trans_iterator = enumerate(stimuli.items())
    for i, (stim_id, stim_path) in trans_iterator:
        trans_iterator.set_description(f"Processing {stim_id}")
        total_len, more_than_1000_scenes = summary_gen_for_1_episode(stim_id, dialogue_file=stim_path, min_length_for_summary=min_length_for_summary)
        total_l += total_len
        more_than_1000 += more_than_1000_scenes
    print(f'total_len: {total_l}, more_than_1000_scenes: {more_than_1000}, {more_than_1000/total_l}')


def summary_gen_for_1_episode(stim_id, dialogue_file=None, min_length_for_summary=500):
    root_data_dir = utils.get_data_root_dir()
    episode_name = stim_id
    if dialogue_file is None:
        dialogue_file = os.path.join(root_data_dir, 'algonauts_2025.competitors','stimuli', 'transcripts', 'friends', 'full', f'{episode_name}.txt')
    out_folder = os.path.join(root_data_dir, 'algonauts_2025.competitors','stimuli', 'transcripts', 'friends', 'summaries')
    scenes_and_dialogues = get_scene_dialogue(dialogue_file)
    dialogue_list = get_dialogue_list(scenes_and_dialogues)
    total_len = 0
    more_than_1000_scenes = 0
    for scene in scenes_and_dialogues['scenes']:
        #get lengths of scenes
        scene_len = get_scene_and_dialogues_display_len(scenes_and_dialogues, dialogue_list, int(scene['id']))
        if scene_len > 1000:
            print(f'{episode_name} {scene["id"]} {scene_len}')
            more_than_1000_scenes += 1
        total_len += 1

        display_text = get_scene_and_dialogues_display_text_till_scene(scenes_and_dialogues, dialogue_list, int(scene['id']))
        if(display_text is None):
            continue
        len_display_text = len(display_text.split())
        if(len_display_text > min_length_for_summary):
            #get summary
            out_file = os.path.join(out_folder, f'{episode_name}.txt')
            write_summary(out_file, scene['id'], episode_name, display_text, len_display_text)
            # if len_display_text > 1000:
            #     print(f'{episode_name} {scene["id"]} {len_display_text}')
    return total_len, more_than_1000_scenes
    print(f'{episode_name} {more_than_1000_scenes} {total_len}, {more_than_1000_scenes/total_len}')

def setup_pipeline():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        token="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"
    )
    return pipeline

if __name__ == "__main__":
    # stim_id = "friends_s03e06"
    # summary_gen_for_1_episode(stim_id)
    utils.set_hf_home_path()
    #test_summary_gen_all_episodes(min_length_for_summary=100)

    pipeline = setup_pipeline()
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
    ]
    print(messages)

    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])


    # for scene in scenes_and_dialogues['scenes']:
    #     print(f'scene: {scene["desc"]}')
    #     display_text = get_scene_and_dialogues_display_text(scenes_and_dialogues, dialogue_list, int(scene['id']), max_words=100000)
    #     print(display_text['fancy_scene_text'])
    #     print("-"*100)
        