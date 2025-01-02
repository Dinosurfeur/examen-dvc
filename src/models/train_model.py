
from pathlib import Path
import sklearn
import pandas as pd 
from sklearn import ensemble
import joblib
import numpy as np
from models_management import save_model

print(joblib.__version__)

project_dir = Path(__file__).resolve().parents[2]

X_train = pd.read_csv(str(project_dir)+"/data/processed_data/X_train.csv")
X_test = pd.read_csv(str(project_dir)+'/data/processed_data/X_test.csv')
y_train = pd.read_csv(str(project_dir)+'/data/processed_data/y_train.csv')
y_test = pd.read_csv(str(project_dir)+'/data/processed_data/y_test.csv')
X_train_scaled = pd.read_csv(str(project_dir)+'/data/processed_data/X_train_scaled.csv')
X_test_scaled = pd.read_csv(str(project_dir)+'/data/processed_data/X_test_scaled.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

gridCV = joblib.load(str(project_dir)+'/models/gridsearch_model.pkl')

#--Define the model
rf_regressor = ensemble.RandomForestRegressor(n_jobs = -1)
rf_regressor.set_params(**gridCV.best_params_)

#--Train the model
rf_regressor.fit(X_train, y_train)

#--Save the trained model to a file
model_filename = 'trained_model.pkl'
save_model(rf_regressor, str(project_dir)+"/models", model_filename)

print("Model trained and saved successfully.")
