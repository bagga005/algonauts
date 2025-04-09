import torch
from torch import nn
from peft import LoraConfig, get_peft_model

# 1. Load the pre-trained model
model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

# 2. Freeze the entire model first
for param in model.parameters():
    param.requires_grad = False

#print layers/modules and their names
# for name, module in model.named_modules():
#     print('name:', name)
#     print('module:', module)
#     print('--------------------------------')

# 3. Define LoRA configuration
lora_config = LoraConfig(
    r=8,                       # LoRA rank
    lora_alpha=16,             # LoRA alpha scaling factor
    target_modules=[           
        # Block 3 temporal convolutions (3×1×1)
        "blocks.3.res_blocks.0.branch2.conv_a",
        "blocks.3.res_blocks.1.branch2.conv_a",
        "blocks.3.res_blocks.2.branch2.conv_a",
        "blocks.3.res_blocks.3.branch2.conv_a",
        "blocks.3.res_blocks.4.branch2.conv_a",
        "blocks.3.res_blocks.5.branch2.conv_a",
        
        # Block 3 spatial convolutions (1×3×3)
        "blocks.3.res_blocks.0.branch2.conv_b",
        "blocks.3.res_blocks.1.branch2.conv_b",
        "blocks.3.res_blocks.2.branch2.conv_b",
        "blocks.3.res_blocks.3.branch2.conv_b",
        "blocks.3.res_blocks.4.branch2.conv_b",
        "blocks.3.res_blocks.5.branch2.conv_b",
        
        # Block 4 temporal convolutions (3×1×1)
        "blocks.4.res_blocks.0.branch2.conv_a",
        "blocks.4.res_blocks.1.branch2.conv_a",
        "blocks.4.res_blocks.2.branch2.conv_a",
        
        # Block 4 spatial convolutions (1×3×3)
        "blocks.4.res_blocks.0.branch2.conv_b",
        "blocks.4.res_blocks.1.branch2.conv_b",
        "blocks.4.res_blocks.2.branch2.conv_b",
    ],
    lora_dropout=0.1,          
    bias="none",               
    task_type="FEATURE_EXTRACTION",     
)

# 4. Apply LoRA to specific layers
# Create a mapping of layer names to their module objects
target_modules = {}
for name, module in model.named_modules():
    # Target the 3×3 spatial convolutions and 3×1×1 temporal convolutions in later blocks
    if (("blocks.3" in name or "blocks.4" in name) and 
        isinstance(module, nn.Conv3d) and
        (module.kernel_size == (3, 3, 3) or module.kernel_size == (3, 1, 1))):
        target_modules[name] = module

# 5. Create PEFT model
peft_model = get_peft_model(model, lora_config)

# 6. Set up the optimizer with weight decay for regularization
# Note: only LoRA parameters will have requires_grad=True at this point
optimizer = torch.optim.AdamW(
    peft_model.parameters(),
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

# 7. Learning rate scheduler - use cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=10,  # Number of epochs
    eta_min=1e-6
)

# 8. Print trainable parameters to verify
trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in peft_model.parameters())
print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")

# 9. Sample training loop (adapt as needed)
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    scheduler.step()