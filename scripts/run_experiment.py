import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append('./src')
from data import load_data
from train import train_classifier

def main(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    train_data_path = config["train_data_path"]
    df = load_data(train_data_path)
    
    X = df.drop(columns='cluster')
    y = df.cluster


    # Define classifiers and perform training
    classifiers = {
        'RandomForest': {
            'model': RandomForestClassifier(),
            'params': {
                'clf__n_estimators': [100],
                'clf__max_depth': [10],
                'clf__class_weight': ['balanced']
            },
        },
    }
    grid_search = train_classifier(X,y, classifiers)
    # TODO: log results of gridsearch


if __name__ == "__main__":
    # Get configuration file path from command line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)
    config_file = sys.argv[1]
    main(config_file)

