from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

import os
import gc
import argparse
from glob import glob

import sys
import utils
print(sys.executable)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from SentenceDataset import SentenceDataset, SentenceDataset_v2, get_transcript_dataSet

from rapidfuzz import fuzz
import re

def setup_environment():
    """Set up environment variables and check CUDA availability."""
    utils.set_hf_path()
    cuda_available = torch.cuda.is_available()
    import socket
    print("Hostname:", socket.gethostname())
    print("PID:", os.getpid())
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    return "cuda" if cuda_available else "cpu"

def load_model_and_tokenizer(checkpoint, device, hf_token, param_dtype, untrained=False):
    """Load tokenizer and model from Hugging Face."""
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, token=hf_token)
    if untrained:
        print("Loading untrained network from config.")
        config = AutoConfig.from_pretrained(checkpoint, token=hf_token)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=param_dtype).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=param_dtype, device_map="auto", token=hf_token
        )
    # set the padding id and token to be able to create padded batches
    tokenizer.truncation_side = "left"  # Keep the rightmost part
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

def collate_fn(batch, tokenizer):
    if utils.isMockMode():
        return None, None
    """Custom collate function for tokenization."""
    encoding = tokenizer.batch_encode_plus(batch, padding=True, truncation=True, 
                                           return_tensors="pt", return_attention_mask=True, 
                                           max_length=tokenizer.model_max_length)
    return encoding['input_ids'], encoding['attention_mask']





def collect_llm_activations(root_data_dir, model, tokenizer, batch_size, device, \
                            kept_tokens=6, n_used_words=500, stimuli="all", n_layers=4, untrained=False, prep_sentences=None):
    """Process transcript files and extract activations."""
    actv_dir = os.path.join(utils.get_output_dir(), utils.get_embeddings_dir())
    all_tsv_files = glob(f"{root_data_dir}/algonauts_2025.competitors/stimuli/transcripts/**/**/*.tsv")
    # Filter files if specific stimuli are requested
    if stimuli != "all":
        print(stimuli)
        all_tsv_files = [f for f in all_tsv_files if any(s in f for s in stimuli.split(','))]
    all_tsv_files.sort()
    print(len(all_tsv_files), list(all_tsv_files)[:3], list(all_tsv_files)[-3:])
    
    for j, transcript_file in tqdm(enumerate(all_tsv_files), total=len(all_tsv_files)):
        #transcript_file e.g. "../../friends_s01e01a.tsv" -> id: friends_s01e01a
        transcript_id = os.path.basename(transcript_file).split(".")[0]
        if utils.is_transcript_already_processed(transcript_id):
            print(f"Skipping {transcript_id} because it already exists")
            continue
        print(f"Extracting for {transcript_id}")
        postfix = f"{n_layers}L{kept_tokens}T{n_used_words}W" + ("+untr" if untrained else "")
        output_file = f"{actv_dir}/{transcript_id}.npy"
        # if os.path.exists(output_file):
        #     continue
        
        # read in the tsv file & replace nans
        df = pd.read_csv(transcript_file, sep='\t').fillna("")
        dataset = SentenceDataset(df["text_per_tr"].tolist(), mode="n_used_words", n_used_words=n_used_words)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer))
        # dataset2 = get_transcript_dataSet(transcript_id, n_used_words=n_used_words)
        # total_len  =0
        # matches =0
        # assert len(dataset) == len(dataset2), f"len(dataset) != len(dataset2): {len(dataset)} != {len(dataset2)}"

        embd_data = []
        for k, (input_ids, attention_mask) in tqdm(enumerate(dataloader), total=len(dataloader)):

            # if k > 0:
            #     advanced_txt = dataset2[k]
            #     ori_txt = dataset[k]

            #     # utils.log_to_file(advanced_txt['fancy_post'])
            #     # utils.log_to_file(advanced_txt['fancy_post'])
            #     pre = advanced_txt['fancy_pre']
            #     pst = advanced_txt['fancy_post']
            #     npre = advanced_txt['normal_pre']
            #     npst = advanced_txt['normal_post']
                
            #     if not pre: pre = ""
            #     if not pst: pst = ""
            #     if not npst: npst = ""
            #     if npre:
            #         full_advanced_txt = npre + ' '  +npst
            #         dirty_full_advanced_txt = npre + ' '  +npst
            #     else:
            #         full_advanced_txt = npst
            #         dirty_full_advanced_txt = npst
            
            #     full_advanced_txt = re.sub(r'\.{3,}', ' ', full_advanced_txt)
            #     last_2_advanced_words = utils.get_last_x_words(full_advanced_txt, 2)
            #     words_list = last_2_advanced_words.split()
            #     last_ori_word = utils.normalize_and_clean_word(utils.get_last_x_words(ori_txt, 1))
            #     highest_score = 0
            #     best_word = ""
            #     highest_score = 0
            #     for word in words_list:
            #         score = fuzz.ratio(utils.normalize_and_clean_word(word), last_ori_word)
            #         if score > highest_score:
            #             highest_score = score
            #             best_word = word
            #     total_len += 1
            #     if highest_score > 80:
            #         matches += 1
            #     #else:
            #         utils.log_to_file(k)
            #         utils.log_to_file(last_2_advanced_words,"orignal:",last_ori_word)
            #         utils.log_to_file(highest_score)
            #         utils.log_to_file(ori_txt)
            #         utils.log_to_file(dirty_full_advanced_txt)
            #         #utils.log_to_file(full_advanced_txt)
            #         #utils.log_to_file(pre + pst)
            #         utils.log_to_file("*"*200)
            
            # print(f"total_len: {total_len}, matches: {matches}, {matches/total_len}")

            if utils.isMockMode():
                continue
            with torch.no_grad():
                outputs = model(input_ids.to(device), attention_mask=attention_mask.to(device), output_hidden_states=True)
            
            # Extract hidden states
            xhs = torch.stack(outputs.hidden_states).detach().cpu()
            # Find the index of the last input text token for each sample before padding
            last_token_indices = attention_mask.sum(dim=1) - 1
            # Get indices for the last X tokens before padding
            indices = np.array([np.clip(range(idx - kept_tokens + 1, idx + 1), 0, None) for idx in last_token_indices])
            # Only keep activations for a few layers to manage file size
            layer_idxs = np.linspace(0, len(xhs) - 1, n_layers).round().astype(int)
            # Subset collected activations based on computed indices
            embds = xhs[:, torch.arange(xhs.shape[1]).unsqueeze(1), indices][layer_idxs]
            embd_data.append(embds.to(torch.float16).numpy())
            del outputs, xhs, embds
            torch.cuda.empty_cache()
            gc.collect()
        
        if utils.isMockMode():
            continue
        np.save(output_file, np.concatenate(embd_data, axis=1))
        utils.save_embedding_metadata(transcript_id, {"n_used_words": n_used_words, "kept_tokens": kept_tokens, "n_layers": n_layers})
        print(f"Saved: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Arguments for customization
    parser.add_argument("--stimuli", type=str, default="all", help="Stimuli set to use (e.g., 'all', 'friends_s07', 'friends_s07,friends_s06')")
    parser.add_argument("--checkpoint", type=str, default="meta-llama/Llama-3.1-8B", help="Model checkpoint")
    parser.add_argument("--param_dtype", type=str, default="auto", help="Model parameter dtype e.g. float16 or auto")
    parser.add_argument("--batch_size", type=int, default=60, help="Batch size for processing (e.g., 60 for A100)")
    parser.add_argument("--kept_tokens", type=int, default=6, help="")
    parser.add_argument("--n_used_words", type=int, default=500, help="")
    parser.add_argument("--n_layers", type=int, default=4, help="")
    parser.add_argument("--untrained", action="store_true", help="Use the untrained model version", default=False)
    parser.add_argument("--prep_sentences", type=str, default=None, help="Preprocessing routine for the sentence dataset")

    args = parser.parse_args()
    
    root_data_dir = utils.get_data_root_dir()
    device=setup_environment()
    
    model_name = args.checkpoint.split("/")[-1]
    param_dtype = "auto" if args.param_dtype=="auto" else getattr(torch, args.param_dtype);
    hf_token= "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"
    if utils.isMockMode():
        tokenizer, model = None, None
    else:
        tokenizer, model = load_model_and_tokenizer(args.checkpoint, device, hf_token, param_dtype, args.untrained)
    kwargs = dict(kept_tokens=args.kept_tokens, n_used_words=args.n_used_words, stimuli=args.stimuli, n_layers=args.n_layers, untrained=args.untrained,
                  prep_sentences=args.prep_sentences)
    collect_llm_activations(root_data_dir, model, tokenizer, args.batch_size, device, **kwargs)