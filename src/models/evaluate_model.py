import pandas as pd 
import numpy as np
from joblib import load
import json
from pathlib import Path
from models_management import load_model

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

project_dir = Path(__file__).resolve().parents[2]

X_train = pd.read_csv(str(project_dir)+"/data/processed_data/X_train.csv")
X_test = pd.read_csv(str(project_dir)+'/data/processed_data/X_test.csv')
y_train = pd.read_csv(str(project_dir)+'/data/processed_data/y_train.csv')
y_test = pd.read_csv(str(project_dir)+'/data/processed_data/y_test.csv')
X_train_scaled = pd.read_csv(str(project_dir)+'/data/processed_data/X_train_scaled.csv')
X_test_scaled = pd.read_csv(str(project_dir)+'/data/processed_data/X_test_scaled.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Load your saved model
model_filename = 'trained_model.pkl'
loaded_model = load_model(str(project_dir)+"/models", model_filename)

predictions = loaded_model.predict(X_test)
pred_df = pd.DataFrame(predictions, columns=['predictions'])
predictions_path = str(project_dir)+"/metrics/scores.json"
pred_df.to_csv(predictions_path, index=False)

score = loaded_model.score(X_test,y_test)
mse = mean_squared_error(y_test,predictions)
CV_score = cross_val_score(loaded_model, X_test,y_test).mean()

metrics = {"mse": mse, "cross_val_score": CV_score, "score": score}
accuracy_path = str(project_dir)+"/metrics/scores.json"
# Save the data to a JSON file
with open(accuracy_path, 'w') as file:
    json.dump(metrics, file, indent=4)

