import sys
sys.path.append('./src/')
import argparse
import json
import pandas as pd
import joblib
from data import load_data

def predict(model_path, data_path):

    model = joblib.load(model_path)

    X = load_data(data_path)

    preds = model.predict(X)

    return preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using a pre-trained model.")
    parser.add_argument("model_path", help="Path to the model")
    parser.add_argument("data_path", help="Path to the data CSV file")
    args = parser.parse_args()

    predictions = predict(args.model_path, args.data_path)
    print(predictions)
