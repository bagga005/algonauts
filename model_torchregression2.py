import torch
import torch.nn as nn
import torch.optim as optim
import time
import utils
from sklearn.model_selection import train_test_split

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.1):
        super(LinearRegressionModel, self).__init__()
        # hidden_size = (input_size + output_size) // 2  # 1600 -> 1300 -> 1000
        self.input_size = input_size
        self.output_size = output_size
        self.num_session_features = 50
        self.num_hidden_features = 100
        self.session_linear1 = nn.Linear(self.num_session_features, self.num_hidden_features)
        self.final_layer = nn.Linear(self.input_size - self.num_session_features + self.num_hidden_features, output_size)
        # self.linear2 = nn.Linear(hidden_size, output_size)
        # self.batchnorm = nn.BatchNorm1d(hidden_size)
        # self.dropout = nn.Dropout(dropout_rate)

        # self.activation = nn.GELU()
        
        nn.init.xavier_uniform_(self.final_layer.weight)
        nn.init.xavier_uniform_(self.session_linear1.weight)
        # nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        #x = self.dropout(self.activation(self.batchnorm(self.linear1(x))))
        #x = self.activation(self.linear1(x))
        session_features = x[:, :self.num_session_features]
        session_features = self.session_linear1(session_features)
        session_features = torch.relu(session_features)
        remaining_features = x[:,self.input_size - self.num_session_features:]
        combined_features = torch.cat([remaining_features, session_features], dim=1)
        return self.final_layer(combined_features)

class RegressionHander_PytorchSimple():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LinearRegressionModel(input_size, output_size).to(self.device)
        print('init RegressionHander_PytorchSimple')

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
        batch_size = 8192
        learning_rate = 0.0001
        epochs = 300
        max_grad_norm = 1.0
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
        
        print(f'Training simple samples: {X_train.shape[0]:,}')
        print(f'Validation simple samples: {X_val.shape[0]:,}')
        print(f'starting torch Simple training')
        
        
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        
        print('len dataloader', len(train_loader))
        # Early stopping parameters
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        best_model_state = None
        train_losses = []
        val_losses = []

        self.model.train()
        total_loss =0
        for epoch in range(epochs):
            total_loss =0
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
            if val_loss + 0.001 < best_val_loss:
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
    
    