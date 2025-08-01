import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer 
import torch
import utils

utils.set_hf_home_path()
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


# Single conversation example (your original code)
messages = [
    {"role": "user", "content": "What is the capital of France?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print("Single query result:")
print(outputs[0]["generated_text"][-1])
print("\n" + "="*50 + "\n")

# Batch processing example - multiple independent conversations
batch_conversations = [
    # Conversation 1
    [
        {"role": "user", "content": "What is the capital of France?"},
    ],
    # Conversation 2 
    [
        {"role": "user", "content": "Explain quantum computing in simple terms."},
    ],
    # Conversation 3 - multi-turn conversation
    [
        {"role": "user", "content": "What's 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "What about 3+3?"},
    ],
    # Conversation 4
    [
        {"role": "system", "content": "You are a helpful cooking assistant."},
        {"role": "user", "content": "How do I make scrambled eggs?"},
    ]
]

# Process all conversations in a batch
print("Processing batch of conversations...")
batch_outputs = pipeline(
    batch_conversations,
    max_new_tokens=128,
    batch_size=2,  # Process 2 conversations at a time
    return_full_text=False  # Only return the generated part
)

# Print results for each conversation
for i, output in enumerate(batch_outputs):
    print(f"Conversation {i+1} result:")
    print(output[0]["generated_text"])
    print("-" * 30)

#export HUGGINGFACE_HUB_TOKEN='"    token="'hf_YMVBuKkOefrCkPOVSCwGrihTdRHvnBBeg"