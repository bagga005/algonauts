import os
import json
import utils
from Scenes_and_dialogues import get_scene_dialogue, get_scene_and_dialogues_display_text, get_dialogue_list, get_scene_and_dialogues_display_text_till_scene, get_scene_and_dialogues_display_len
from glob import glob
from tqdm import tqdm
import transformers
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer 

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
    # Compute summary length as number of words
    summary_length = len(summary.split())
    
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
            "summary_length": summary_length,
            "unsummarized_length": unsummarized_length
        })
    else:
        # Create new data with the entry
        data = [{
            "scene_id": scene_id,
            "stim_id": stim_id,
            "summary": summary,
            "summary_length": summary_length,
            "unsummarized_length": unsummarized_length
        }]
    
    # Write data back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def summary_gen_all_episodes(pipeline, min_length_for_summary=500):
    root_data_dir = utils.get_data_root_dir()
    
    #list of full text transcripts
    file_in_filter = utils.get_stimuli_prefix()
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

    trans_iterator = tqdm(enumerate(stimuli.items()), total=len(list(stimuli)))
    #trans_iterator = enumerate(stimuli.items())
    for i, (stim_id, stim_path) in trans_iterator:
        trans_iterator.set_description(f"Processing {stim_id}")
        summary_gen_for_1_episode(stim_id, pipeline, dialogue_file=stim_path, min_length_for_summary=min_length_for_summary)

def get_query(display_text):
    preMsg = "Summarize below dialogue from part of a tv show in less than 300 words. This is not the full episode, just a part of it from the start. Output only the summary, no other text.\n"
    display_text = preMsg + display_text
    return display_text



def summary_gen_for_1_episode(stim_id, pipeline, dialogue_file=None, min_length_for_summary=500):
    root_data_dir = utils.get_data_root_dir()
    episode_name = stim_id
    if dialogue_file is None:
        dialogue_file = os.path.join(root_data_dir, 'algonauts_2025.competitors','stimuli', 'transcripts', 'friends', 'full', f'{episode_name}.txt')
    out_folder = os.path.join(root_data_dir, 'algonauts_2025.competitors','stimuli', 'transcripts', 'friends', 'summaries')
    out_file = os.path.join(out_folder, f'{episode_name}.json')
    scenes_and_dialogues = get_scene_dialogue(dialogue_file)
    dialogue_list = get_dialogue_list(scenes_and_dialogues)
    total_len = 0
    more_than_1000_scenes = 0
    all_texts = []
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
            all_texts.append({
                    'text': display_text,
                    'stim_id': stim_id,
                    'scene_id': scene['id'],
                    'scene_desc': scene['desc'],
                    'unsummarized_length': len_display_text,
                    'messages': [{"role": "user", "content": get_query(display_text)}]
                })
            
    if len(all_texts) > 0:
        dataset = Dataset.from_list(all_texts)
        
        def generate_summaries(batch):
            """Process a batch of texts"""
            summaries = []
            for messages in batch['messages']:
                try:
                    outputs = pipeline(
                        messages,
                        max_new_tokens=512,
                        # do_sample=False,  # For consistent results
                        # temperature=0.7,
                    )
                    output_text_obj = outputs[0]["generated_text"][-1]
                    if output_text_obj and 'content' in output_text_obj:
                        summary = output_text_obj['content']
                    else:
                        raise Exception("Summary generation failed")
                        summary = "Summary generation failed"
                    summaries.append(summary)
                except Exception as e:
                    print(f"Error generating summary: {e}")
                    summaries.append("Error in summary generation")
            
            return {'summary': summaries}
        # Process in batches
        print(f"Processing {len(dataset)} texts in batches of {2}")
        dataset = dataset.map(
            generate_summaries,
            batched=True,
            batch_size=2,
            desc="Generating summaries"
        )
        
        for item in dataset:
            print(f"|Scene: {item['scene_desc']}|")
            print(item['summary'])
            print("-" * 100)
            write_summary(
                out_file, 
                item['scene_id'], 
                item['stim_id'], 
                item['summary'], 
                item['unsummarized_length']
            )
            #get summary
            # print(f'|Scene: {scene["desc"]}|')
            # 
            
            # summary = get_summary_from_llm(display_text, pipeline)

            # write_summary(out_file, scene['id'], episode_name, summary, len_display_text)
            # if len_display_text > 1000:
            #     print(f'{episode_name} {scene["id"]} {len_display_text}')


    
def get_summary_from_llm(display_text, pipeline):
    preMsg = "Summarize below dialogue from part of a tv show in less than 300 words. This is not the full episode, just a part of it from the start. Output only the summary, no other text.\n"
    display_text = preMsg + display_text
    #display_text = "What is the capital of Uzbekistan?"
    messages = [
        {"role": "user", "content": display_text},
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=512,
    )
    output_text_obj = outputs[0]["generated_text"][-1]
    if output_text_obj:
        print('output_text_obj',output_text_obj)
        output_text = output_text_obj
    print(output_text)
    print("-"*100)
    return output_text

def setup_pipeline():
    hf_token = utils.get_hf_token()
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    # Load tokenizer and set pad_token before making pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", token=hf_token)

    # Now make pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        token=hf_token
    )
    return pipeline

if __name__ == "__main__":
    utils.set_hf_home_path()

    pipeline = setup_pipeline()
    summary_gen_all_episodes(pipeline, min_length_for_summary=500)
    # messages = [
    #     {"role": "user", "content": "What is the capital of France?"},
    # ]
    # print(messages)

    # outputs = pipeline(
    #     messages,
    #     max_new_tokens=256,
    # )
    # print(outputs[0]["generated_text"][-1])
    stim_id = "friends_s05e06"
    summary_gen_for_1_episode(stim_id, pipeline)

    # for scene in scenes_and_dialogues['scenes']:
    #     print(f'scene: {scene["desc"]}')
    #     display_text = get_scene_and_dialogues_display_text(scenes_and_dialogues, dialogue_list, int(scene['id']), max_words=100000)
    #     print(display_text['fancy_scene_text'])
    #     print("-"*100)
        