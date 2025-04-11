import torch
from torchvision.models.feature_extraction import create_feature_extractor
import os
import utils
import numpy as np
from torch import nn
from peft import LoraConfig, get_peft_model
import h5py
import time
from sklearn.model_selection import train_test_split

class VisionLinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size, device, dropout_rate=0.2):
        super(VisionLinearRegressionModel, self).__init__()
        self.v_model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        self.model_layer = 'blocks.5.pool'
        
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
        class VideoModelWrapper(nn.Module):
            def __init__(self, model, model_layer):
                super().__init__()
                self.model = model
                self.model_layer = model_layer
                # Create feature extractor to get intermediate layer outputs
                self.feature_extractor = create_feature_extractor(
                    model,
                    return_nodes=[self.model_layer]
                )
            
            def forward(self, *args, **kwargs):
                # Extract the input tensor from args
                if len(args) > 0:
                    x = args[0]
                elif 'input_ids' in kwargs:
                    x = kwargs['input_ids']
                else:
                    raise ValueError("No input tensor provided")
                
                # Ensure the input is in the correct format
                if len(x.shape) == 4:  # (batch, channels, height, width)
                    x = x.unsqueeze(2)  # Add temporal dimension
                
                # Use feature extractor to get intermediate layer output
                features = self.feature_extractor(x)
                return features[self.model_layer]
        
        # Wrap the video model
        wrapped_model = VideoModelWrapper(self.v_model, self.model_layer)
        
        # 5. Create PEFT model with the wrapped model
        self.visual_model = get_peft_model(wrapped_model, self.lora_config)


    def forward(self, x):
        #Remove batch dimension
        # print('x.shape', x.shape)
        x = x.squeeze(0)
        # Pass the input directly to the wrapped model
        layer_output = self.visual_model(x)
        # print('layer_output.shape', layer_output.shape)
        layer_output = layer_output.flatten().unsqueeze(0)
        # print('layer_output.shape', layer_output.shape)
        #make prediction with linear layer
        prediction = self.linear4(layer_output)
        # print('prediction.shape', prediction.shape)
        #add batch dimension
        return prediction

class RegressionHander_Vision():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VisionLinearRegressionModel(input_size, output_size, self.device).to(self.device)

    def train(self, features_train, fmri_train, features_train_val, fmri_train_val):
        start_time = time.time()  
        X_train, X_val, y_train, y_val = train_test_split(
            features_train, fmri_train, 
            test_size=0.2, 
            random_state=42
        )   
        train_loader = prepare_training_data(X_train, y_train)
        val_loader = prepare_training_data(X_val, y_val)

        epochs = 100
        learning_rate_linear = 1e-5
        weight_decay_linear = 1e-4
        # 6. Set up the optimizer with weight decay for regularization
        # Note: only LoRA parameters will have requires_grad=True at this point
        lora_optimizer = torch.optim.AdamW(
            self.model.visual_model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # 7. Learning rate scheduler - use cosine annealing
        lora_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            lora_optimizer, 
            T_max=10,  # Number of epochs
            eta_min=1e-6
        )

        linear_optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate_linear, weight_decay=weight_decay_linear)

        criterion = torch.nn.MSELoss()
        # 8. Print trainable parameters to verify
        trainable_params = sum(p.numel() for p in self.model.visual_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
        print(f"Total parameters: {total_params:,}")

        # print(f'Training lora vision: {X_train.shape[0]:,}')
        # print(f'Validation lora vision: {X_val.shape[0]:,}')
        print(f'starting lora vision training')

        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        best_model_state = None
        train_losses = []
        val_losses = []
        total_loss =0
        for epoch in range(epochs):
            total_loss =0
            in_batch=1
            for batch_X, batch_y in train_loader:
                # Zero gradients for both optimizers
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
                print('in_batch', in_batch, 'batch loss', loss.item(), 'avg_loss', total_loss/in_batch)
                in_batch += 1
            
            train_loss = total_loss / len(train_loader)
            train_losses.append(train_loss)

            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    y_pred = self.model(batch_X)
                    loss = criterion(y_pred, batch_y)
                    val_loss += loss.item()
            
            val_loss = val_loss / len(val_loader)
            val_losses.append(val_loss)

            # Print average loss every 10 epochs
            # Print progress
            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            
             # Early stopping check
            if val_loss + 0.001 < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'\nEarly stopping triggered at epoch {epoch}')
                    break

            # Step both schedulers at the end of each epoch
            lora_scheduler.step()
        
        # Restore best model
        self.model.load_state_dict(best_model_state)
            
            
        training_time = time.time() - start_time
        return self.model, training_time
            #linear_scheduler.step()
    def save_model(self, model_name):
        utils.save_model_pytorch(self.model, model_name)

    def load_model(self, model_name):
        params = utils.load_model_pytorch(model_name)
        self.model.load_state_dict(params)

    def predict(self, features_val):
        features_val = torch.FloatTensor(features_val).to(self.device)
        self.model.eval()
        with torch.no_grad():
            fmri_val_pred = self.model(features_val).cpu().numpy()  # Move to CPU and convert to numpy
        return fmri_val_pred    

def prepare_training_data(input, target):
    # input = [('friends_s02e01a', (0,3)), ('friends_s02e01a', (1,4)), ('friends_s02e01a', (2,5))]
    # target = np.random.randn(3, 1000).astype(np.float32)
    
    # Create custom Dataset
    class VideoDataset(torch.utils.data.Dataset):
        def __init__(self, input_data, targets):
            self.input_data = input_data
            self.targets = torch.FloatTensor(targets)
            
        def __len__(self):
            return len(self.input_data)
        
        def __getitem__(self, idx):
            videoname, frame_indices = self.input_data[idx]
            filename = os.path.join(utils.get_stimulus_pre_features_dir(), 'pre', 'visual', videoname+'.h5')
            # print('frame_indices[0]', frame_indices[0])
            # print('frame_indices[1]', frame_indices[1])
            # Here you would load the video frames from the h5 file
            # For example:
            with h5py.File(filename, 'r') as f:
                # Assuming your h5 file has a 'frames' dataset
                frames = f[videoname]['visual']
                # print('frames.shape', frames.shape)
                frames = torch.from_numpy(frames[frame_indices[0]:frame_indices[1]]).squeeze(1)
                # print('frames.shape', frames.shape)
            return frames, self.targets[idx]
    
    # Create dataset
    dataset = VideoDataset(input, target)
    
    # Create dataloader
    batch_size = 1  # Since we're dealing with video data, smaller batch sizes are common
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True  # Helps speed up data transfer to GPU
    )
    
    return dataloader
    
 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# learner = RegressionHander_Vision(32768, 1000)
# learner.train(prepare_training_data())