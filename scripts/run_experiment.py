import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import wandb
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('./src')
from data import load_data
from data_transformer import CustomDataTransformer

def main(config_file):

    with open(config_file, 'r') as f:
        config = json.load(f)

    train_data_path = config["train_data_path"]
    df = load_data(train_data_path)
    # TODO: Data cleanup
    
    X = df.drop(columns='cluster')
    y = df.cluster

    clf = Pipeline([
            ('data_transformer', CustomDataTransformer()),
            ('preprocessor', StandardScaler()),
            ('gbc', GradientBoostingClassifier())
        ])
    
    param_grid = [{
                'gbc__n_estimators': [100],
                'gbc__learning_rate': [0.1],
                'gbc__max_depth': [3],
                'gbc__min_samples_split': [2],
                'gbc__min_samples_leaf': [1],
                'gbc__subsample': [1.0],
            }]
    
    grid_search = GridSearchCV(clf,
                               param_grid,
                               cv=5,
                               scoring=['precision_weighted','recall_weighted','f1_weighted', 'balanced_accuracy'],
                               refit='f1_weighted',
                               n_jobs=-1,
                               return_train_score=False,
                               error_score='raise')
    
    grid_search.fit(X, y)
    print(grid_search.best_params_)
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_

    if config["wandb"]:
        wandb.init(project="oil-well-cluster-predictor")
        pipeline_summary = {}        
        for name, step in clf.named_steps.items():
            parameters = step.get_params()
            pipeline_summary[name] = parameters
        wandb.config.update(pipeline_summary)
        results = pd.DataFrame(grid_search.cv_results_).filter(regex="^mean_test").iloc[grid_search.best_index_].to_dict()
        wandb.log(results)

if __name__ == "__main__":
    # Get configuration file path from command line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)
    config_file = sys.argv[1]
    main(config_file)

