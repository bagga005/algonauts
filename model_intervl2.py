# ------------------------------------------------------------
# 1.  Setup: imports & model
# ------------------------------------------------------------
# from transformers import AutoTokenizer, AutoModelForVisionText2Text
import torch, textwrap, pprint
import utils
from transformers import AutoModel, AutoTokenizer, AutoConfig, BitsAndBytesConfig

hf_path = "OpenGVLab/InternVL3-1B-Pretrained"      # change if you use a different size
device = "cuda" if torch.cuda.is_available() else "cpu"
hf_path = utils.get_mvl_model()

    # For the second model loading instance
    # Configure quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)


model = AutoModel.from_pretrained(
        hf_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True, use_fast=True)

# one dummy 448×448 RGB frame (ViT default size); replace with real images
pixel_values = torch.randn(1, 3, 448, 448, dtype=torch.bfloat16, device=device)

# tokenise prompt exactly ONCE; no generation, just a forward pass
#tok = tokenizer(example_prompt, return_tensors="pt", add_special_tokens=False).to(device)
text_prompt = '| Scene: Central Perk, the whole gang is there including Janice. |\nFrame1: <image> More text'

response = model.chat(
        tokenizer, 
        pixel_values, 
        text_prompt, 
        dict(max_new_tokens=1),
        history=None, 
        return_history=False
    )

print(response)
#print(resp2)