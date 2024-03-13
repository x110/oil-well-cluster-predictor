import sys
sys.path.append('./src/')
import argparse
import json
import pandas as pd
import joblib
from data import load_data

def predict(config_file, new_data_path):
    with open(config_file, 'r') as f:
        config = json.load(f)

    model_path = config['model_path']

    model = joblib.load(model_path)

    X = load_data(new_data_path)

    preds = model.predict(X)

    return preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using a pre-trained model.")
    parser.add_argument("config_file", help="Path to the configuration JSON file")
    parser.add_argument("new_data_path", help="Path to the new data CSV file")
    args = parser.parse_args()

    predictions = predict(args.config_file, args.new_data_path)
    print(predictions)
