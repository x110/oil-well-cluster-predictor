import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
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
                'clf__n_estimators': [100, 200, 300],
                'clf__max_depth': [3, 6, 9, None],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 4],
                'clf__bootstrap': [True, False],
                'clf__class_weight': [None, 'balanced']
            },
        },
        'SVM': {
            'model': SVC(),
            'params': {
                'clf__C': [0.1, 1, 10, 100],
                'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'clf__gamma': ['scale', 'auto'],
                'clf__class_weight': [None, 'balanced']
            },
        },
        'LogisticRegression': {
            'model': LogisticRegression(),
            'params': {
                'clf__C': [0.1, 1, 10, 100],
                'clf__penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'clf__class_weight': [None, 'balanced']
            },
        },
        'XGBoost': {
            'model': XGBClassifier(),
            'params': {
                'clf__n_estimators': [100, 200, 300],
                'clf__max_depth': [3, 6, 9],
                'clf__learning_rate': [0.01, 0.1, 0.3],
                'clf__subsample': [0.6, 0.8, 1.0],
                'clf__colsample_bytree': [0.6, 0.8, 1.0],
                'clf__reg_alpha': [1e-3, 1e-2, 0.1, 1, 10],
                'clf__reg_lambda': [1e-3, 1e-2, 0.1, 1, 10],
                'clf__scale_pos_weight': [1],
                'clf__objective': ['binary:logistic']
            }
        }
    }




    grid_search = train_classifier(X,y, classifiers)
    # TODO: log results of gridsearch
    wandb.log({"Best Score": grid_search.best_score_,
               "Best Parameters": grid_search.best_params_})


if __name__ == "__main__":
    # Get configuration file path from command line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)
    config_file = sys.argv[1]
    main(config_file)

#200 min
#6 giga
#16 giga,
#125 dhs (6 months)

