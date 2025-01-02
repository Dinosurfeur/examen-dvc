import os
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from check_structure import check_existing_file, check_existing_folder, create_folder_if_necessary

def split_data(df):
    # Split data into training and testing sets
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)

def save_normalization(X_train, X_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)



    
def main():
    create_folder_if_necessary(str(project_dir)+"/data/processed_data")
    df = import_dataset(str(project_dir)+"/data/raw_data/raw.csv")
    df = df.drop(['date'], axis=1)
    X_train, X_test, y_train, y_test = split_data(df)

    save_dataframes(X_train, X_test, y_train, y_test, str(project_dir)+"/data/processed_data")
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train))
    X_test_scaled = pd.DataFrame(scaler.transform(X_test))
    save_normalization(X_train_scaled, X_test_scaled, str(project_dir)+"/data/processed_data")
    

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]
    print(project_dir)
    main()