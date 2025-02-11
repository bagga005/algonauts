from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
import time
import utils

class LinearHandler_Sklearn():
    def __init__(self,input_size, output_size, alphas=[0.1, 1.0, 10.0, 100.0], alpha=1.0):
        self.input_size = input_size
        self.output_size = output_size
        #self.model = Ridge(alpha=alpha, solver='svd') #LinearRegression()
        self.model = RidgeCV(alphas=alphas, store_cv_values=True)
        #self.model = LinearRegression()

    def train(self,features_train, fmri_train, features_train_val, fmri_train_val):
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
