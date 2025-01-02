import os
import joblib

def save_model(model, output_folderpath,modelname='model.pkl'):
    # Save the model to the output file path
    output_filepath = os.path.join(output_folderpath, modelname)
    joblib.dump(model, output_filepath)

def load_model(model_filepath,modelname='model.pkl'):
    # Load the model from the input file path
    filepath = os.path.join(model_filepath, modelname)
    return joblib.load(filepath)