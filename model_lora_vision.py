import torch
from torchvision.models.feature_extraction import create_feature_extractor
import os
import utils
import random
import numpy as np
from torch import nn
from peft import LoraConfig, get_peft_model
import h5py
import time
from sklearn.model_selection import train_test_split
import pickle
import wandb
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from peft.tuners.lora import LoraModel
from torch.utils.checkpoint import checkpoint
import random
import sys

class VisionLinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size, device, dropout_rate=0.2):
        super(VisionLinearRegressionModel, self).__init__()
        self.v_model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        self.model_layer = 'blocks.5.pool'
        self.device = device
        self.linear4 = nn.Linear(input_size, output_size)
        nn.init.kaiming_normal_(self.linear4.weight)

        self.lora_config = LoraConfig(
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
        # 2. Freeze the entire model first
        for param in self.v_model.parameters():
            param.requires_grad = False
        # Create a custom wrapper for the video model

        # 2. Apply LoRA manually
        #self.visual_model = LoraModel(self.v_model, self.lora_config, adapter_name="default")
        self.visual_model = get_peft_model(self.v_model, self.lora_config)
        self.visual_model.to(self.device)


    def forward(self, x):
        b_size, window = x.shape[:2]
        x = x.view(b_size * window, *x.shape[2:])

        if len(x.shape) == 4:
            x = x.unsqueeze(2)

        with torch.no_grad():
            x = self.visual_model.model.blocks[0](x)
            x = self.visual_model.model.blocks[1](x)
            x = self.visual_model.model.blocks[2](x)
        with torch.enable_grad():
            # Blocks 3 and 4 contain LoRA parameters and require gradients
            x = checkpoint(self.visual_model.model.blocks[3], x, use_reentrant=False)
            x = checkpoint(self.visual_model.model.blocks[4], x, use_reentrant=False)

        
            layer_output = checkpoint(self.visual_model.model.blocks[5].pool, x, use_reentrant=False)
            layer_output = layer_output.reshape(layer_output.shape[0], -1)
            layer_output = layer_output.reshape(b_size, -1)
            
            prediction = checkpoint(self.linear4, layer_output, use_reentrant=False)

        return prediction


# Define setup and cleanup functions for distributed training at module level
def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    dist.destroy_process_group()

# Define VideoDataset class here so it's available to train_on_device
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, targets):
        self.input_data = input_data
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        videoname, frame_indices = self.input_data[idx]
        filename = os.path.join(utils.get_stimulus_pre_features_dir(), 'pre', 'visual', videoname+'.h5')

        if utils.isMockMode():
            return torch.randn(4,3,8,256,256), self.targets[idx]
        # For example:
        with h5py.File(filename, 'r') as f:
            frames = f[videoname]['visual']
            frames = torch.from_numpy(frames[frame_indices[0]:frame_indices[1]]).squeeze(1)
        return frames, self.targets[idx]

# Move train_on_device outside the class to make it picklable
def save_checkpoint(state, filename):
    """Save checkpoint to disk"""
    torch.save(state, filename)

def load_checkpoint(model, lora_optimizer, linear_optimizer, lora_scheduler, linear_scheduler, filename):
    """Load checkpoint from disk"""
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, weights_only=False)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        lora_optimizer.load_state_dict(checkpoint['lora_optimizer'])
        linear_optimizer.load_state_dict(checkpoint['linear_optimizer'])
        lora_scheduler.load_state_dict(checkpoint['lora_scheduler'])
        linear_scheduler.load_state_dict(checkpoint['linear_scheduler'])
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        # Also restore random states if they were saved
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'])
        if 'numpy_rng_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng_state'])
        if 'python_rng_state' in checkpoint and random is not None:
            random.setstate(checkpoint['python_rng_state'])
        
        print(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
        return start_epoch, best_val_loss, patience_counter
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, float('inf'), 0, [], []

def train_on_device(rank, world_size, model_params, train_data, val_data, config):
    # Unpack parameters
    start_time = time.time() 
    input_size, output_size, enable_wandb = model_params
    X_train, y_train = train_data
    X_val, y_val = val_data
    batch_size, epochs= config['batch_size'], config['epochs']
    num_gpus = config['num_gpus']
    resume = config.get('resume', False)
    resume_checkpoint = config.get('resume_checkpoint', None)
    # Get CPU count
    cpu_count = os.cpu_count()
    num_workers = 2
    if cpu_count is not None:
        # Follow the guideline: min(4 × num_GPUs, num_CPU_cores)
        num_workers = min(4 * max(1, num_gpus), cpu_count)
        num_workers = 1
    print(f'num_workers for dataloader: {num_workers}')
    print(f'variables gpu: {num_gpus} world_size: {world_size} rank: {rank}')

    # make epochs small if mode mode
    # if utils.isMockMode():
    #     X_train = X_train[:batch_size*2]
    #     y_train = y_train[:batch_size*2]
    #     X_val = X_val[:batch_size*2]
    #     y_val = y_val[:batch_size*2]

    exception_raised = False
    try:
        if rank == 0: print('distributed: start setup')
        # Setup distributed process
        setup_distributed(rank, world_size)
        torch.cuda.set_device(rank)
        
        if rank == 0: print('distributed: step 1')
        # Create model and move it to the correct device
        device = torch.device(f"cuda:{rank}")
        model = VisionLinearRegressionModel(input_size, output_size, device)
        model = model.to(device)
        model = DDP(model, device_ids=[rank])

        if rank == 0: print('distributed: model setup')
        # Create dataset and prepare data loaders with DistributedSampler
        train_dataset = VideoDataset(X_train, y_train)
        val_dataset = VideoDataset(X_val, y_val)
        if rank == 0: print('distributed: dataset setup')
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        if rank == 0: print('distributed: DistributedSampler setup')
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
        if rank == 0: print('distributed: DataLoader setup')
        # Training settings
        linear_learning_rate_initial = config.get('linear_learning_rate_initial', 1e-4)
        linear_learning_rate_final = config.get('linear_learning_rate_final', 1e-6)
        linear_weight_decay = config.get('linear_weight_decay', 1e-3)
        lora_learning_rate_initial = config.get('lora_learning_rate_initial', 1e-4)
        lora_learning_rate_final = config.get('lora_learning_rate_final', 1e-6)
        lora_weight_decay = config.get('lora_weight_decay', 1e-3)
        if rank == 0: print('distributed: config setup')
        # Set up optimizers
        lora_optimizer = torch.optim.AdamW(
            model.module.visual_model.parameters(),
            lr=lora_learning_rate_initial,
            weight_decay=lora_weight_decay,
            betas=(0.9, 0.999)
        )
        
        lora_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            lora_optimizer, 
            T_max=20,
            eta_min=lora_learning_rate_final
        )
        
        linear_optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=linear_learning_rate_initial, 
            weight_decay=linear_weight_decay
        )
        
        linear_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            linear_optimizer, 
            T_max=20,
            eta_min=linear_learning_rate_final
        )
        if rank == 0: print('distributed: linear_scheduler setup')
        criterion = torch.nn.MSELoss()
        
        # Initialize training state
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_model_state = None
        start_epoch = 0
        
        # Load checkpoint if resuming
        checkpoint_loaded = False
        if resume and resume_checkpoint and rank == 0:
            checkpoint_path = os.path.join(utils.get_output_dir(), 'models', resume_checkpoint + '.pth')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.module.load_state_dict(checkpoint['state_dict'])
                print('checkpoint restored')
                # Broadcast model parameters from rank 0 to all other processes
                for param in model.parameters():
                    dist.broadcast(param.data, src=0)
                lora_optimizer.load_state_dict(checkpoint['lora_optimizer'])
                linear_optimizer.load_state_dict(checkpoint['linear_optimizer'])
                lora_scheduler.load_state_dict(checkpoint['lora_scheduler'])
                linear_scheduler.load_state_dict(checkpoint['linear_scheduler'])
                best_val_loss = checkpoint['best_val_loss']
                patience_counter = checkpoint['patience_counter']
                start_epoch = checkpoint['epoch'] 
                print(f"Resuming from epoch {start_epoch}")
                checkpoint_loaded = True
            else:
                print('checkpoint not found')
                # Don't raise here, let all ranks know
                checkpoint_loaded = False

        # Broadcast a flag to all ranks indicating if checkpoint was loaded
        checkpoint_flag = torch.tensor([int(checkpoint_loaded)], dtype=torch.long).to(device) if rank == 0 else torch.zeros(1, dtype=torch.long).to(device)
        dist.barrier()
        dist.broadcast(checkpoint_flag, src=0)
        if checkpoint_flag.item() == 0:
            raise Exception("Checkpoint not loaded on rank 0")

        # Broadcast training state from rank 0 to all processes in a more robust way
        if rank == 0:
            print(f"Rank 0 broadcasting: start_epoch={start_epoch}, patience_counter={patience_counter}, best_val_loss={best_val_loss}")
            state_tensor = torch.tensor([int(start_epoch), int(patience_counter)], dtype=torch.int64, device=device)
            val_loss_tensor = torch.tensor([float(best_val_loss)], dtype=torch.float32, device=device)
        else:
            state_tensor = torch.zeros(2, dtype=torch.int64, device=device)
            val_loss_tensor = torch.zeros(1, dtype=torch.float32, device=device)
        dist.barrier()
        dist.broadcast(state_tensor, src=0)
        dist.broadcast(val_loss_tensor, src=0)
        print(f'Rank {rank} received state_tensor: {state_tensor}, val_loss_tensor: {val_loss_tensor}')
        # Unpack values
        start_epoch = int(state_tensor[0].item())
        patience_counter = int(state_tensor[1].item())
        best_val_loss = float(val_loss_tensor[0].item())
        print(f'Rank {rank}: start_epoch={start_epoch}, best_val_loss={best_val_loss:.6f}, patience_counter={patience_counter}')
        
        # Only log with wandb on the main process
        if rank == 0 and enable_wandb:
            wandb_config = {
                'epochs': epochs,
                'batch_size': batch_size * world_size,
                'num_gpus': world_size,
                'learning_rate_linear': linear_learning_rate_initial,
                'learning_rate_linear_final': linear_learning_rate_final,
                'linear_weight_decay': linear_weight_decay,
                'lora_learning_rate_initial': lora_learning_rate_initial,
                'lora_learning_rate_final': lora_learning_rate_final,
                'lora_weight_decay': lora_weight_decay,
            }
            project_name, model_name, _ = utils.get_wandb_config()
            wandb.init(
                id=model_name,
                project=project_name,
                name=model_name,
                config=wandb_config,
                resume="allow",
            )
        print(f'rank {rank} start_epoch {start_epoch}')
        for epoch in range(start_epoch, epochs):
            # Set epoch for distributed sampler
            train_loader.sampler.set_epoch(epoch)
            
            # Training phase
            model.train()
            total_loss = 0
            in_batch = 1
            
            for batch_X, batch_y in train_loader:
                # Zero gradients for both optimizers
                lora_optimizer.zero_grad()
                linear_optimizer.zero_grad()
                
                # Move batch to device
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Update weights with both optimizers
                lora_optimizer.step()
                linear_optimizer.step()
                
                total_loss += loss.item()
                
                if in_batch % 100 == 0 or in_batch < 3:
                    if rank == 0:  # Only print from main process
                        print(f'GPU {rank} | Epoch {epoch} | Batch {in_batch} | Loss(last batch): {loss.item():.4f} | Loss (Avg in epoch): {total_loss/in_batch:.4f}')
                        # for name, param in model.module.named_parameters():
                        #     if ('lora_B' in name or 'lora_A' in name or 'linear4' in name) and param.requires_grad:
                        #         grad_norm = torch.norm(param.grad).item() if param.grad is not None else 0
                        #         param_norm = torch.norm(param).item()
                        #         print(f"{name}: grad_norm={grad_norm:.6f}, param_norm={param_norm:.6f}")
                
                
                in_batch += 1
            
            # Calculate average loss across all processes
            avg_train_loss = total_loss / len(train_loader)
            train_loss_tensor = torch.tensor(avg_train_loss).to(device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            train_loss = train_loss_tensor.item() / world_size
            
            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    y_pred = model(batch_X)
                    loss = criterion(y_pred, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_loss_tensor = torch.tensor(avg_val_loss).to(device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss = val_loss_tensor.item() / world_size
            
            # Print progress on main process
            if rank == 0 and epoch % 1 == 0:
                print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

            # Step both schedulers at the end of each epoch
            lora_scheduler.step()
            linear_scheduler.step()

            # Early stopping check (only on rank 0)
            if rank == 0:
                if val_loss + 0.001 < best_val_loss:
                    best_val_loss = val_loss
                    #best_model_state = model.module.state_dict().copy()
                    patience_counter = 0
                    torch.save(model.module.state_dict(), 
                            os.path.join(utils.get_output_dir(), 'models', f'lora-best-distributed.pth'))
                else:
                    patience_counter += 1
            
            # Broadcast patience counter to all processes
            patience_tensor = torch.tensor(patience_counter).to(device)
            dist.broadcast(patience_tensor, src=0)
            patience_counter = patience_tensor.item()

            # Save checkpoint from the main process
            if rank == 0:
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'lora_optimizer': lora_optimizer.state_dict(),
                    'linear_optimizer': linear_optimizer.state_dict(),
                    'lora_scheduler': lora_scheduler.state_dict(),
                    'linear_scheduler': linear_scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                    'torch_rng_state': torch.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                }
                
                checkpoint_path = os.path.join(utils.get_output_dir(), 'models', f'lora-{epoch}-checkpoint.pth')
                save_checkpoint(checkpoint, checkpoint_path)
            
            # Save model periodically
            if rank == 0 and epoch % 1 == 0: #epoch != 0 and 
                # Save the DDP model's state dictionary
                torch.save(model.module.state_dict(), 
                          os.path.join(utils.get_output_dir(), 'models', f'lora-{epoch}-distributed.pth'))
            

            # Log to wandb
            if rank == 0 and enable_wandb:
                logs = {
                    'train/loss': train_loss,
                    'train/num_steps': epoch,
                    "train/lr_lora": lora_optimizer.param_groups[0]['lr'],
                    "train/lr_linear": linear_optimizer.param_groups[0]['lr'],
                    'test/loss': val_loss,
                    'test/num_steps': epoch
                }
                lora_a_grads = []
                lora_a_norms = []
                lora_b_grads = []
                lora_b_norms = []
                linear_grads = []
                linear_norms = []
                for name, param in model.module.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        if 'lora_A' in name:
                            lora_a_grads.append(torch.norm(param.grad).item())
                            lora_a_norms.append(torch.norm(param).item())
                        elif 'lora_B' in name:
                            lora_b_grads.append(torch.norm(param.grad).item())    
                            lora_b_norms.append(torch.norm(param).item())
                        elif 'linear4' in name:
                            linear_grads.append(torch.norm(param.grad).item())  
                            linear_norms.append(torch.norm(param).item())
                if linear_grads:
                    logs.update({
                        'gradients/linear4/mean': np.mean(linear_grads),
                        'gradients/linear4/max': np.max(linear_grads),
                        'gradients/linear4/histogram': wandb.Histogram(linear_grads),
                        'batch': epoch * len(train_loader) + in_batch
                    })
                if linear_norms:
                    logs.update({
                        'norms/linear4/mean': np.mean(linear_norms),
                        'norms/linear4/max': np.max(linear_norms),
                        'norms/linear4/histogram': wandb.Histogram(linear_norms),
                        'batch': epoch * len(train_loader) + in_batch
                    })
                if lora_a_grads:
                    logs.update({
                        'gradients/lora_A/mean': np.mean(lora_a_grads),
                        'gradients/lora_A/max': np.max(lora_a_grads),
                        'gradients/lora_A/histogram': wandb.Histogram(lora_a_grads),
                        'batch': epoch * len(train_loader) + in_batch
                    })
                if lora_a_norms:
                    logs.update({
                        'norms/lora_A/mean': np.mean(lora_a_norms),
                        'norms/lora_A/max': np.max(lora_a_norms),
                        'norms/lora_A/histogram': wandb.Histogram(lora_a_norms),
                        'batch': epoch * len(train_loader) + in_batch
                    })
                if lora_b_grads:
                    logs.update({
                        'gradients/lora_B/mean': np.mean(lora_b_grads),
                        'gradients/lora_B/max': np.max(lora_b_grads),
                        'gradients/lora_B/histogram': wandb.Histogram(lora_b_grads),
                        'batch': epoch * len(train_loader) + in_batch
                    })
                if lora_b_norms:
                    logs.update({
                        'norms/lora_B/mean': np.mean(lora_b_norms),
                        'norms/lora_B/max': np.max(lora_b_norms),
                        'norms/lora_B/histogram': wandb.Histogram(lora_b_norms),
                        'batch': epoch * len(train_loader) + in_batch
                    })
                wandb.log(logs)
            
            if patience_counter >= patience:
                if rank == 0:
                    print(f'\nEarly stopping triggered at epoch {epoch}')
                break
            
            
        
        # Save the best model state to a file that can be loaded by the main process
        # if rank == 0 and best_model_state is not None:
        #     best_model_path = os.path.join(utils.get_output_dir(), 'models', 'best_distributed_model.pth')
        #     torch.save(best_model_state, best_model_path)
        
    except KeyboardInterrupt:
        print(f"[GPU {rank}] Training interrupted by user")
        exception_raised = True
    except Exception as e:
        print(f"[GPU {rank}] Error during training: {str(e)}")
        exception_raised = True
    finally:
        # Make sure cleanup happens regardless of how the function exits
        if dist.is_initialized():
            cleanup_distributed()
            print(f"[GPU {rank}] Process group cleaned up properly")
        if exception_raised:
            raise Exception("Error message")

    return None if rank == 0 else None

class RegressionHander_Vision():
    def __init__(self, input_size, output_size,  pretrain_params_name=None, enable_wandb=False):
        print('Initializing RegressionHander_Vision')
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VisionLinearRegressionModel(input_size, output_size, self.device)
        if pretrain_params_name is not None:
            self.load_model(pretrain_params_name)
            print(f'loaded params from model {pretrain_params_name}')
        else:
            print('not loading existing model')
        self.model.to(self.device)
        self.enable_wandb = enable_wandb
        

    def train(self, features_train, fmri_train, features_train_val, fmri_train_val, num_gpus=1, resume_checkpoint=None):
        print('num_gpus', num_gpus, torch.cuda.device_count())
        if resume_checkpoint:
            resume = True
            print('doing a resume from checkpoint:', resume_checkpoint)
        else:
            resume = False
            print('not resumeing from checkpoint')
        if num_gpus > 1 and torch.cuda.device_count() > 1:
            return self.train_distributed(features_train, fmri_train, features_train_val, fmri_train_val, num_gpus, resume, resume_checkpoint)
        else:
            return self.train_single_gpu(features_train, fmri_train, features_train_val, fmri_train_val, resume, resume_checkpoint)
    
    def train_distributed(self, features_train, fmri_train, features_train_val, fmri_train_val, num_gpus, resume=False, resume_checkpoint=None):
        # Split the data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            features_train, fmri_train, 
            test_size=0.2, 
            random_state=42
        )
        
        # Determine batch size - we can use a larger batch size with multiple GPUs
        # The effective batch size will be batch_size * num_gpus
        batch_size = 4  # This is per GPU
        epochs = 5
        
        # Spawn processes for each GPU
        world_size = min(num_gpus, torch.cuda.device_count())
        print(f"Training with {world_size} GPUs")
        
        start_time = time.time()
        
        # Prepare the parameters to pass to train_on_device
        model_params = (self.input_size, self.output_size, self.enable_wandb)
        train_data = (X_train, y_train)
        val_data = (X_val, y_val)
        config = {
            'batch_size': batch_size,
            'epochs': epochs,
            'linear_learning_rate_initial': 1e-4,
            'linear_learning_rate_final': 1e-6,
            'linear_weight_decay': 1e-3,
            'lora_learning_rate_initial': 1e-3,
            'lora_learning_rate_final': 1e-5,
            'lora_weight_decay': 1e-5,
            'num_gpus': num_gpus,
            'resume': resume,
            'resume_checkpoint': resume_checkpoint,
        }
        
        mp.spawn(
            train_on_device,
            args=(world_size, model_params, train_data, val_data, config),
            nprocs=world_size,
            join=True
        )
        
        # Load the best model saved by rank 0
        best_model_path = os.path.join(utils.get_output_dir(), 'models', 'lora-best-distributed.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model from {best_model_path}")
        
        training_time = time.time() - start_time
        return self.model, training_time
    
    def train_single_gpu(self, features_train, fmri_train, features_train_val, fmri_train_val, resume=False, resume_checkpoint=None):
        start_time = time.time()  
        print('start training at', start_time)
        epochs = 2
        batch_size = 32
        
        linear_learning_rate_initial = 1e-4
        linear_learning_rate_final = 1e-6
        linear_weight_decay = 1e-3
        lora_learning_rate_initial = 1e-4
        lora_learning_rate_final = 1e-6
        lora_weight_decay = 1e-3
        wandb_config = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate_linear': linear_learning_rate_initial,
            'learning_rate_linear_final': linear_learning_rate_final,
            'linear_weight_decay': linear_weight_decay,
            'lora_learning_rate_initial': lora_learning_rate_initial,
            'lora_learning_rate_final': lora_learning_rate_final,
            'lora_weight_decay': lora_weight_decay,
        }
        project_name, model_name, _ = utils.get_wandb_config()
        print('single gpu train self.enable_wandb', self.enable_wandb)
        if self.enable_wandb:
            wandb.init(
                id=model_name,
                project=project_name,
                name=model_name,
                config=wandb_config,
                resume="allow",
            )

        X_train, X_val, y_train, y_val = train_test_split(
            features_train, fmri_train, 
            test_size=0.2, 
            random_state=42
        )   
        print('start preparing data at ', time.time())
        # make epochs small if mode mode
        if utils.isMockMode():
            X_train = X_train[:batch_size*2]
            y_train = y_train[:batch_size*2]
            X_val = X_val[:batch_size*2]
            y_val = y_val[:batch_size*2]
        train_loader = prepare_training_data(X_train, y_train, batch_size=batch_size)
        val_loader = prepare_training_data(X_val, y_val, batch_size=batch_size)
        print('done preparing data at ', time.time())
        
        
        # 6. Set up the optimizer with weight decay for regularization
        # Note: only LoRA parameters will have requires_grad=True at this point
        lora_optimizer = torch.optim.AdamW(
            self.model.visual_model.parameters(),
            lr=lora_learning_rate_initial,
            weight_decay=lora_weight_decay,
            betas=(0.9, 0.999)
        )

        # 7. Learning rate scheduler - use cosine annealing
        lora_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            lora_optimizer, 
            T_max=20,  # Number of epochs
            eta_min=lora_learning_rate_final
        )

        linear_optimizer = torch.optim.Adam(self.model.parameters(), lr=linear_learning_rate_initial, weight_decay=linear_weight_decay)
        linear_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            linear_optimizer, 
            T_max=20,  # Number of epochs
            eta_min=linear_learning_rate_final
        )

        criterion = torch.nn.MSELoss()
        # 8. Print trainable parameters to verify
        trainable_params = sum(p.numel() for p in self.model.visual_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.visual_model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
        print(f"Total parameters: {total_params:,}")

        # def make_hook(name):
        #     def hook_fn(module, input, output):
        #         print(f"Layer used: {name}")
        #     return hook_fn

        # for name, module in self.model.named_modules():
        #     if "lora_A" in name or "lora_B" or "linear" in name:
        #         print(f'Adding hook to: {name}')

        # print(f'Training lora vision: {X_train.shape[0]:,}')
        # print(f'Validation lora vision: {X_val.shape[0]:,}')
        print(f'starting lora vision training')

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_model_state = None
        start_epoch = 0
        
        # Load checkpoint if resuming training
        if resume and resume_checkpoint:
            checkpoint_path = os.path.join(utils.get_output_dir(), 'models', resume_checkpoint)
            start_epoch, best_val_loss, patience_counter = load_checkpoint(
                self.model, lora_optimizer, linear_optimizer, lora_scheduler, linear_scheduler, f'{checkpoint_path}.pth'
            )
            start_epoch += 1    
            print(f"Resuming from epoch {start_epoch}")
        
        total_loss = 0
        for epoch in range(start_epoch, epochs):
            total_loss = 0
            in_batch = 1
            in_batch_losses = []
            load_start = time.time()
            total_loading_time = 0
            #print('about to start new batch', time.time())
            for batch_X, batch_y in train_loader:
                # Zero gradients for both optimizers
                #print('start batch at ', time.time())
                loading_time = time.time() - load_start
                total_loading_time += loading_time
                #print(f'load time {loading_time:.2f} seconds')
                lora_optimizer.zero_grad()
                linear_optimizer.zero_grad()
                 # Move batch to device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                # Backward pass
                loss.backward()
                # Update weights with both optimizers
                lora_optimizer.step()
                linear_optimizer.step()
                total_loss += loss.item()
                if in_batch % 100 == 0 or in_batch < 3:
                    print('in_batch', in_batch, 'batch loss', loss.item(), 'avg_loss', total_loss/in_batch)
                    avg_loading_time = total_loading_time / in_batch
                    print(f'total loading time {total_loading_time:.2f}', f'avg loading time {avg_loading_time:.2f}')
                    in_batch_losses.append(total_loss/in_batch)
                    self.save_train_val_loss(True, in_batch_losses, f'{epoch}-{in_batch}')
                    print(f'Epoch {epoch} | Batch {in_batch} | Loss: {loss.item():.4f}')
                    # for name, param in self.model.named_parameters():
                    #     if ('lora_B' in name or 'lora_A' in name or 'linear4' in name) and param.requires_grad:
                    #         grad_norm = torch.norm(param.grad).item() if param.grad is not None else 0
                    #         param_norm = torch.norm(param).item()
                    #         print(f"{name}: grad_norm={grad_norm:.6f}, param_norm={param_norm:.6f}")
                in_batch += 1
                #print('batch done at ',time.time())
                load_start = time.time()
            
            train_loss = total_loss / len(train_loader)


            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    y_pred = self.model(batch_X)
                    loss = criterion(y_pred, batch_y)
                    val_loss += loss.item()
            
            val_loss = val_loss / len(val_loader)


            # save model
            if epoch != 0 and epoch % 5 == 0:
                self.save_model(f'lora-{epoch}')

            # Print average loss every 1 epochs
            # Print progress
            if epoch % 1 == 0:
                print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            

            # Step both schedulers at the end of each epoch
            lora_scheduler.step()
            linear_scheduler.step()

            # Early stopping check
            if val_loss + 0.001 < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'lora_optimizer': lora_optimizer.state_dict(),
                'linear_optimizer': linear_optimizer.state_dict(),
                'lora_scheduler': lora_scheduler.state_dict(),
                'linear_scheduler': linear_scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'torch_rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'python_rng_state': random.getstate() if 'random' in sys.modules else None,
            }
            
            checkpoint_path = os.path.join(utils.get_output_dir(), 'models', f'lora-{epoch}-checkpoint.pth')
            save_checkpoint(checkpoint, checkpoint_path)

            # Save model periodically
            if epoch != 0 and epoch % 5 == 0: #epoch != 0 and 
                # Save the DDP model's state dictionary
                torch.save(self.model.state_dict(), 
                          os.path.join(utils.get_output_dir(), 'models', f'lora-{epoch}-distributed.pth'))
            
            if self.enable_wandb:
                logs = {
                    'train/loss': train_loss,
                    'train/num_steps': epoch,
                    "train/lr_lora": lora_optimizer.param_groups[0]['lr'],
                    "train/lr_linear": linear_optimizer.param_groups[0]['lr'],
                    'test/loss': val_loss,
                    'test/num_steps': epoch
                }
                lora_a_grads = []
                lora_a_norms = []
                lora_b_grads = []
                lora_b_norms = []
                linear_grads = []
                linear_norms = []
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        if 'lora_A' in name:
                            lora_a_grads.append(torch.norm(param.grad).item())
                            lora_a_norms.append(torch.norm(param).item())
                        elif 'lora_B' in name:
                            lora_b_grads.append(torch.norm(param.grad).item())    
                            lora_b_norms.append(torch.norm(param).item())
                        elif 'linear4' in name:
                            linear_grads.append(torch.norm(param.grad).item())  
                            linear_norms.append(torch.norm(param).item())
                if linear_grads:
                    logs.update({
                        'gradients/linear4/mean': np.mean(linear_grads),
                        'gradients/linear4/max': np.max(linear_grads),
                        'gradients/linear4/histogram': wandb.Histogram(linear_grads),
                        'batch': epoch * len(train_loader) + in_batch
                    })
                if linear_norms:
                    logs.update({
                        'norms/linear4/mean': np.mean(linear_norms),
                        'norms/linear4/max': np.max(linear_norms),
                        'norms/linear4/histogram': wandb.Histogram(linear_norms),
                        'batch': epoch * len(train_loader) + in_batch
                    })
                if lora_a_grads:
                    logs.update({
                        'gradients/lora_A/mean': np.mean(lora_a_grads),
                        'gradients/lora_A/max': np.max(lora_a_grads),
                        'gradients/lora_A/histogram': wandb.Histogram(lora_a_grads),
                        'batch': epoch * len(train_loader) + in_batch
                    })
                if lora_a_norms:
                    logs.update({
                        'norms/lora_A/mean': np.mean(lora_a_norms),
                        'norms/lora_A/max': np.max(lora_a_norms),
                        'norms/lora_A/histogram': wandb.Histogram(lora_a_norms),
                        'batch': epoch * len(train_loader) + in_batch
                    })
                if lora_b_grads:
                    logs.update({
                        'gradients/lora_B/mean': np.mean(lora_b_grads),
                        'gradients/lora_B/max': np.max(lora_b_grads),
                        'gradients/lora_B/histogram': wandb.Histogram(lora_b_grads),
                        'batch': epoch * len(train_loader) + in_batch
                    })
                if lora_b_norms:
                    logs.update({
                        'norms/lora_B/mean': np.mean(lora_b_norms),
                        'norms/lora_B/max': np.max(lora_b_norms),
                        'norms/lora_B/histogram': wandb.Histogram(lora_b_norms),
                        'batch': epoch * len(train_loader) + in_batch
                    })
                wandb.log(logs)
            #now that logging is done, do early stopping if needed
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered at epoch {epoch}')
                break

        # # Restore best model
        # if best_model_state is not None:
        #     self.model.load_state_dict(best_model_state)
        
        training_time = time.time() - start_time
        return self.model, training_time
           
    def save_train_val_loss(self, isTrain, arr, epoch):
        file_name = f'train-{epoch}.pkl'
        if not isTrain:
            file_name = f'val-{epoch}.pkl'
        full_path = os.path.join(utils.get_output_dir(), 'models', file_name)
        pickle.dump(arr, open(full_path, 'wb'))
    
    def load_print_train_val_loss(self, isTrain, epoch):
        file_name = f'train-{epoch}.pkl'
        if not isTrain:
            file_name = f'val-{epoch}.pkl'
        full_path = os.path.join(utils.get_output_dir(), 'models', file_name)
        arr = pickle.load(open(full_path, 'rb'))
        print(arr)

    def save_model(self, model_name, separate_vision=False):
        if not separate_vision:
            utils.save_model_pytorch(self.model, model_name)
        else:
            utils.save_model_pytorch(self.model.v_model, f'{model_name}_vision')
            utils.save_model_pytorch(self.model.linear4, f'{model_name}_linear')

    def load_model(self, model_name):
        params = utils.load_model_pytorch(model_name)
        self.model.load_state_dict(params)

    def predict(self, features_val):
        print('prediction called')
        mock_fmri = np.random.randn(len(features_val), 1000).astype(np.float32)
        pred_loader = prepare_training_data(features_val, mock_fmri, batch_size=8, is_for_training=False)
        self.model.eval()
        fmri_val_pred = []
        with torch.no_grad():
            batch_counter = 0
            for batch_X, batch_y in pred_loader:
                batch_X = batch_X.to(self.device)
                output = self.model(batch_X)
                #print('output.shape', output.shape)
                output = output.cpu().numpy()
                fmri_val_pred.append(output)
                if batch_counter % 10 == 0:
                    print(f'batch_counter {batch_counter} | Total: {len(features_val)}')
                batch_counter += 1
        fmri_val_pred = np.concatenate(fmri_val_pred, axis=0)
        print('fmri_val_pred.shape', fmri_val_pred.shape)
        return fmri_val_pred
 


def prepare_training_data(input, target, batch_size=2, is_for_training=True, num_gpus=1):
    # Determine optimal number of workers based on CPU count
    import os
    
    # Get CPU count and set optimal num_workers
    cpu_count = os.cpu_count()
    if cpu_count is None:
        # Fallback if os.cpu_count() returns None
        num_workers = 2
    else:
        # A good rule of thumb is to use min(4 × num_GPUs, num_CPU_cores)
        # or simply num_CPU_cores // 2 as a starting point
        num_workers = min(4 * max(1, num_gpus), cpu_count)
    print(f'num_workers for dataloader: {num_workers}')
        
    # Create dataset
    dataset = VideoDataset(input, target)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_for_training,
        num_workers=num_workers,
        pin_memory=True  # Helps speed up data transfer to GPU
    )
    
    return dataloader
    
 
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# learner = RegressionHander_Vision(32768, 1000)
# input = [('friends_s02e01a', (0,4)), ('friends_s02e01a', (1,5)), ('friends_s02e01a', (2,6))]
# target = np.random.randn(3, 1000).astype(np.float32)
# prepare_training_data(input, target)
#learner.train(input, target, None, None)