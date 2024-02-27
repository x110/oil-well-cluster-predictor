import sys
sys.path.append('./src/')
import argparse
import json
import pandas as pd
import joblib
from make_dataset import process_data

def predict(config_file, new_data_path):
    # Load configuration from JSON file
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Define paths
    model_path = config['model_path']

    # Load the model and class names
    model, class_names = joblib.load(model_path)

    # Process the new data
    # TODO: Remove 'outputdir' from process_data function
    # TODO: Maybe include process_data in sklearn pipeline
    X = process_data(new_data_path, 'del.csv').set_index('well')

    # Make predictions
    y_pred = model.predict(X)

    # Map predictions to class names
    preds = {well: class_names[pred] for pred, well in zip(y_pred, X.index)}
    return preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using a pre-trained model.")
    parser.add_argument("config_file", help="Path to the configuration JSON file")
    parser.add_argument("new_data_path", help="Path to the new data CSV file")
    args = parser.parse_args()

    predictions = predict(args.config_file, args.new_data_path)
    print(predictions)
