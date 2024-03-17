import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import sys
import wandb
sys.path.append('./src')
from data import load_data
from train import train_classifier

def main(config_file):

    wandb.init(project="oil-well-cluster-predictor")

    with open(config_file, 'r') as f:
        config = json.load(f)

    train_data_path = config["train_data_path"]
    df = load_data(train_data_path)
    # TODO: Data cleanup
    
    X = df.drop(columns='cluster')
    y = df.cluster

    classifiers = {
        'RandomForest': {
            'model': RandomForestClassifier(),
            'params': {
                'clf__n_estimators': [500],
                'clf__max_depth': [16],
                'clf__class_weight': ['balanced']
            },
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(),
            'params': {
                'clf__n_estimators': [100],
                'clf__learning_rate': [0.1],
                'clf__max_depth': [3],
                'clf__min_samples_split': [2],
                'clf__min_samples_leaf': [1],
                'clf__subsample': [1.0],
                'clf__max_features': [None],
                'clf__random_state': [None]
            }
        }

    }

    grid_search = train_classifier(X,y, classifiers)
    wandb.log({"Best Score": grid_search.best_score_,
               "Best Parameters": grid_search.best_params_})


if __name__ == "__main__":
    # Get configuration file path from command line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)
    config_file = sys.argv[1]
    main(config_file)

