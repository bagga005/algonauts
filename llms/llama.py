import transformers
import torch

model_id = "meta-llama/Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    token="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"
)

messages = [
    {"role": "user", "content": "What is the capital of France?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])

#export HUGGINGFACE_HUB_TOKEN='"    token="'hf_YMVBuKkOefrCkPOVSCwGrihTdRHvnBBeg"