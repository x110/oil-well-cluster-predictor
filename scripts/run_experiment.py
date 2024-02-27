import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append('./src')
from make_dataset import process_data
from train import train_classifier

def main(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    train_data_path = config["train_data_path"]
    processed_train_data_path = config["processed_train_data_path"]
    test_data_path = config["test_data_path"]
    processed_test_data_path = config["processed_test_data_path"]

    # TODO: change to exists or overwrite?
    try:
        df_train = pd.read_csv(processed_train_data_path)
        df_test = pd.read_csv(processed_test_data_path)
    except:
        df_train = process_data(train_data_path, processed_train_data_path)
        df_test = process_data(test_data_path, processed_test_data_path)

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
    grid_search = train_classifier(df_train, classifiers)
    # TODO: log results of gridsearch


if __name__ == "__main__":
    # Get configuration file path from command line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)
    config_file = sys.argv[1]
    main(config_file)
