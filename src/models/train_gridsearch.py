from pathlib import Path
import sklearn
import pandas as pd 
from sklearn import ensemble
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
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


# Define the parameter grid to use for the GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'criterion' : ['squared_error','friedman_mse','poisson'],
    'max_depth': [10, 20, 30, 40, 50],
    'min_samples_leaf': [1, 2, 4]
}
grid_clf = GridSearchCV(estimator=ensemble.RandomForestRegressor(), param_grid=param_grid, n_jobs=-1,
                    cv=2, verbose=1,refit= True)
grille = grid_clf.fit(X_train_scaled,y_train)

print("grid best params : ",grid_clf.best_params_)

save_model(grid_clf, str(project_dir)+"/models",'gridsearch_model.pkl')

