import torch
import torch.nn as nn
import torch.optim as optim
import time
import utils
import numpy as np

class TransformerRegressionModel(nn.Module):
    def __init__(self, input_size, output_size, nhead=8, num_layers=2, dropout=0.1):
        super(TransformerRegressionModel, self).__init__()
        
        # Define sizes
        self.input_size = input_size
        self.output_size = output_size
        
        # Make sure input_size is divisible by nhead for attention mechanism
        self.d_model = (input_size // nhead) * nhead
        if self.d_model != input_size:
            self.input_projection = nn.Linear(input_size, self.d_model)
        else:
            self.input_projection = nn.Identity()
            
        # Batch normalization for input features
        self.input_norm = nn.BatchNorm1d(self.d_model)
        
        # Transformer Encoder Layer optimized for regression
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=2*self.d_model,  # Reduced from 4x to 2x for regression
            dropout=dropout,
            activation='gelu',  # GELU works well with continuous data
            batch_first=True,
            norm_first=False    # Pre-norm architecture for better training stability
        )
        
        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False 
        )
        
        # Output layers for regression
        self.output_layers = nn.Sequential(
            # nn.Linear(self.d_model, (self.d_model + output_size) // 2),
            # nn.BatchNorm1d((self.d_model + output_size) // 2),
            # nn.GELU(),
            # nn.Dropout(dropout),
            # nn.Linear((self.d_model + output_size) // 2, output_size)
            nn.Linear(self.d_model, output_size)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)  # Changed to normal for regression
                
    def forward(self, x):
        # Add batch dimension if not present
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Project input if needed and normalize
        x = self.input_projection(x)
        x = self.input_norm(x.squeeze(1)).unsqueeze(1)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Process output for regression
        x = x.squeeze(1)
        x = self.output_layers(x)
        
        return x

class RegressionHander_Transformer():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerRegressionModel(input_size, output_size, nhead=8, num_layers=1, dropout=0.2).to(self.device)

    def train(self, features_train, fmri_train, features_train_val, fmri_train_val):
        start_time = time.time()
        
        # Split data with stratification
        # X_train, X_val, y_train, y_val = train_test_split(
        #     features_train, fmri_train, 
        #     test_size=0.2, 
        #     random_state=42
        # )
        
        # # Convert to PyTorch tensors
        # X_train = torch.FloatTensor(X_train).to(self.device)
        # y_train = torch.FloatTensor(y_train).to(self.device)
        # X_val = torch.FloatTensor(X_val).to(self.device)
        # y_val = torch.FloatTensor(y_val).to(self.device)
        X_train = torch.FloatTensor(features_train).to(self.device)
        y_train = torch.FloatTensor(fmri_train).to(self.device)
        X_val = torch.FloatTensor(features_train_val).to(self.device)
        y_val = torch.FloatTensor(fmri_train_val).to(self.device)

        
        # Create DataLoaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=256,  # Adjusted for regression
            shuffle=True,
            pin_memory=False  # Faster data transfer to GPU
        )
        
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=256,
            shuffle=False,
            pin_memory=False
        )
        
        # MSE loss for regression
        criterion = nn.MSELoss()
        
        # AdamW with reduced weight decay for regression
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.0001,
            weight_decay=0.01,  # Reduced from 0.01
            betas=(0.9, 0.999)  # Standard betas work well for regression
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        best_val_loss = float('inf')
        patience = 25  # Increased patience for regression
        patience_counter = 0
        
        for epoch in range(300):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                y_pred = self.model(batch_X)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)  # Reduced for regression
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss = train_loss / len(train_loader)
            
            # Validation with correlation metric
            self.model.eval()
            val_loss = 0
            val_corr = 0
            y_pred_np = []
            y_true_np = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    y_pred = self.model(batch_X)
                    val_loss += criterion(y_pred, batch_y).item()
                    if epoch % 10 == 0:
                        y_pred_np = y_pred.clone().cpu().numpy()
                        y_true_np = batch_y.clone().cpu().numpy()

            
            val_loss = val_loss / len(val_loader)
            
            scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                utils.compute_encoding_accuracy(y_true_np, y_pred_np, "Val Subject", "Val Modality")
                print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Corr: {val_corr:.4f}')
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'\nEarly stopping triggered at epoch {epoch}')
                    break
        
        self.model.load_state_dict(best_model_state)
        return self.model, time.time() - start_time

    def save_model(self, model_name):
        utils.save_model_pytorch(self.model, model_name)

    def load_model(self, model_name):
        params = utils.load_model_pytorch(model_name)
        self.model.load_state_dict(params)

    def predict(self, features_val):
        """
        Predict fMRI responses for validation features
        """
        self.model.eval()
        with torch.no_grad():
            # Convert to tensor if numpy array
            if isinstance(features_val, np.ndarray):
                features_val = torch.FloatTensor(features_val).to(self.device)
                
            # Process in batches to avoid memory issues
            batch_size = 256
            predictions = []
            
            # Create DataLoader for validation features
            dataset = torch.utils.data.TensorDataset(features_val)
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size,
                shuffle=False
            )
            
            for (batch_x,) in dataloader:
                # Ensure input is properly shaped for transformer
                if batch_x.dim() == 2:
                    batch_x = batch_x.unsqueeze(1)  # Add sequence dimension
                    
                # Get predictions
                pred = self.model(batch_x)
                predictions.append(pred.cpu())
                
            # Concatenate all predictions
            fmri_val_pred = torch.cat(predictions, dim=0).numpy()
            
            return fmri_val_pred  
