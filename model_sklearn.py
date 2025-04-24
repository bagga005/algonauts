from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
import time
import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class LinearHandler_Sklearn():
    def __init__(self,input_size, output_size, alphas=[0.1, 1.0, 10.0, 100.0], alpha=1.0):
        self.input_size = input_size
        self.output_size = output_size
        #self.model = Ridge(alpha=alpha, solver='svd') #LinearRegression()
        self.model = RidgeCV(alphas=alphas, store_cv_values=True)
        #self.model = LinearRegression()

    def train(self,features_train, fmri_train, features_train_val, fmri_train_val, num_gpus=1):
        ### Record start time ###
        start_time = time.time()    
        self.model.fit(features_train, fmri_train)
        training_time = time.time() - start_time
        return self.model, training_time
    
    def save_model(self, model_name):
        utils.save_model_sklearn(self.model, model_name)

    def load_model(self, model_name):
        self.model = utils.load_model_sklearn(model_name)  # Direct assignment instead of load_state_dict

    def predict(self, features_val):
        fmri_val_pred = self.model.predict(features_val)
        return fmri_val_pred

    def visualize_feature_importance(self, top_k=10, output_indices=None):
        """
        Visualize the most important features for specified outputs based on coefficient magnitudes.
        
        Parameters
        ----------
        top_k : int
            Number of top features to show for each output
        output_indices : list of int, optional
            Indices of specific outputs to analyze. If None, shows first 5 outputs.
        """
        coefficients = self.model.coef_  # Shape: (n_outputs, n_features)
        
        # If no output indices specified, take first 5 (or less if fewer outputs)
        if output_indices is None:
            output_indices = range(min(5, self.output_size))
        
        # Create subplots for each selected output
        n_plots = len(output_indices)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        for idx, output_idx in enumerate(output_indices):
            # Get coefficients for this output
            output_coeffs = coefficients[output_idx]
            
            # Get indices of top k features by absolute magnitude
            top_k_indices = np.argsort(np.abs(output_coeffs))[-top_k:]
            
            # Create bar plot
            y_pos = np.arange(top_k)
            bars = axes[idx].barh(y_pos, output_coeffs[top_k_indices])
            
            # Color code based on coefficient sign
            for i, bar in enumerate(bars):
                if output_coeffs[top_k_indices[i]] < 0:
                    bar.set_color('red')
                else:
                    bar.set_color('blue')
            
            # Customize plot
            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels([f'Feature {i}' for i in top_k_indices])
            axes[idx].set_title(f'Top {top_k} Important Features for Output {output_idx}')
            axes[idx].set_xlabel('Coefficient Value')
            
            # Add vertical line at x=0
            axes[idx].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        return fig

    def plot_coefficient_heatmap(self, n_outputs_start, n_outputs_end, n_features_start, n_features_end):
        """
        Create a heatmap of model coefficients.
        
        Parameters
        ----------
        n_outputs : int, optional
            Number of outputs to show. If None, shows all.
        n_features : int, optional
            Number of features to show. If None, shows all.
        """
        coefficients = self.model.coef_  # Shape: (n_outputs, n_features)
        
        # Subset the coefficient matrix if specified
        if n_outputs_start is not None:
            coefficients = coefficients[n_outputs_start:n_outputs_end]
        if n_features_start is not None:
            coefficients = coefficients[:, n_features_start:n_features_end]
        print('coefficients.shape', coefficients.shape)
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(coefficients, 
                   cmap='RdBu_r',
                   center=0,
                   xticklabels=range(coefficients.shape[1]),
                   yticklabels=range(coefficients.shape[0]))
        plt.title('Coefficient Heatmap')
        plt.xlabel('Feature Index')
        plt.ylabel('Output Index')
        plt.show()
        return plt.gcf()
