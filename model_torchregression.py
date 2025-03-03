import torch
import torch.nn as nn
import torch.optim as optim
import time
import utils
from sklearn.model_selection import train_test_split

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.2):
        super(LinearRegressionModel, self).__init__()
        # Progressive dimensionality reduction
        self.linear1 = nn.Linear(input_size, 4096)
        self.batchnorm1 = nn.BatchNorm1d(4096)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.linear2 = nn.Linear(4096, 2048)
        self.batchnorm2 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        
        self.linear4 = nn.Linear(2048, output_size)
        
        self.activation = nn.GELU()
        
        # Initialize weights
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.kaiming_normal_(self.linear4.weight)

    def forward(self, x):
        x = self.dropout1(self.activation(self.batchnorm1(self.linear1(x))))
        x = self.dropout2(self.activation(self.batchnorm2(self.linear2(x))))
        #x = self.dropout3(self.activation(self.batchnorm3(self.linear3(x))))
        return self.linear4(x)

class RegressionHander_Pytorch():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LinearRegressionModel(input_size, output_size).to(self.device)

    def train(self, features_train, fmri_train, features_train_val, fmri_train_val):
        """
        Train a linear-regression-based encoding model to predict fMRI responses
        using movie features.

        Parameters
        ----------
        features_train : float
            Stimulus features for the training movies.
        fmri_train : float
            fMRI responses for the training movies.

        Returns
        -------
        model : object
            Trained regression model.
        training_time : float
            Time taken to train the model in seconds.
        """
        
        ### Record start time ###
        start_time = time.time()
        batch_size = 1024
        learning_rate_initial_1 = 1e-5
        learning_rate_initial_2 = 1e-4
        learning_rate = 1e-4
        warmup_epochs_1 = 50
        warmup_epochs_2 = 100
        epochs = 1000
        max_grad_norm = 1.0
        weight_decay = 1e-3
        #utils.analyze_fmri_distribution(fmri_train)
        ### Convert features_train and fmri_train to PyTorch tensors ###
        X_train, X_val, y_train, y_val = train_test_split(
        features_train, fmri_train, 
        test_size=0.2, 
        random_state=42
    )
    
        ### Convert to PyTorch tensors ###
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        print(f'Training samples: {X_train.shape[0]:,}')
        print(f'Validation samples: {X_val.shape[0]:,}')

        
        
        # Create DataLoaders for both training and validation
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        
        #model = LinearRegressionModel(features_train.shape[1], fmri_train.shape[1]).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate_initial_1, weight_decay=weight_decay)
        
        
        print('len dataloader', len(train_loader))
        # Early stopping parameters
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        best_model_state = None
        train_losses = []
        val_losses = []
        print('starting training iterations')
        self.model.train()
        total_loss =0
        for epoch in range(epochs):
            total_loss =0
            # Increase learning rate linearly during warmup
            if epoch > warmup_epochs_1:
                #print('update lr')
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate_initial_2
            if epoch > warmup_epochs_2:
                #print('update lr')
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
            for batch_X, batch_y in train_loader:
                # Forward pass
                optimizer.zero_grad()
                y_pred = self.model(batch_X)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                total_loss += loss.item()
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
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'\nEarly stopping triggered at epoch {epoch}')
                    break
        
        # Restore best model
        self.model.load_state_dict(best_model_state)
            # Early stopping check
            # if avg_loss < best_loss - min_delta:
            #     best_loss = avg_loss
            #     patience_counter = 0
            #     best_model_state = self.model.state_dict()
            # else:
            #     patience_counter += 1
            #     if patience_counter >= patience:
            #         print(f"Early stopping triggered at epoch {epoch}")
            #         self.model.load_state_dict(best_model_state)
            #         break

        ### Calculate training time ###
        training_time = time.time() - start_time

        ### Output ###
        return self.model, training_time
    
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

