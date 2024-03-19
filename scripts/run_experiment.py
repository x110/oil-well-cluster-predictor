import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import wandb
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
import sys
sys.path.append('./src')
from data import load_data
from data_transformer import CustomDataTransformer
import numpy as np

def main(config_file):

    with open(config_file, 'r') as f:
        config = json.load(f)

    train_data_path = config["train_data_path"]
    df = load_data(train_data_path)
    # TODO: Data cleanup
    
    X = df.drop(columns='cluster')
    y = df.cluster

    clf = Pipeline([
            ('data_tf', CustomDataTransformer()),
            ('preprocessor', StandardScaler()),
            #("kmeans", KMeans()),
            ('gbc', GradientBoostingClassifier())
        ])
    
    param_grid = [{

                #'kmeans__n_clusters': range(2, 100),
                #'data_tf__years': range(1, 20),
                'gbc__n_estimators': [2],
                'gbc__learning_rate': [0.01],
                'gbc__max_depth': [2],
                #'gbc__min_samples_split': [2],
                #'gbc__min_samples_leaf': [1],
                'gbc__subsample': [.25],
            }]
    
    grid_search = GridSearchCV(clf,
                               param_grid,
                               cv=5,
                               scoring=['precision_weighted','recall_weighted','f1_weighted', 'balanced_accuracy'],
                               refit='f1_weighted',
                               n_jobs=-1,
                               return_train_score=True,
                               error_score='raise')
    
    grid_search.fit(X, y)
    results = pd.DataFrame(grid_search.cv_results_)
    print(grid_search.best_params_)
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    train_score = results['mean_train_f1_weighted'].iloc[0]
    test_score = results['mean_test_f1_weighted'].iloc[0]
    print(f"Best parameters for: {best_model}")
    print(f"Best score for: {best_score}")
    print(f"train score: {train_score}, test_score: {test_score}")

    gbc = grid_search.best_estimator_.named_steps['gbc']
    preprocess = Pipeline([
        ('data_tf', CustomDataTransformer()),
        ('preprocessor', StandardScaler()),
        ])
    X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=.2, stratify=y)
    errors = [f1_score(y_val, y_pred,average='weighted') for y_pred in gbc.staged_predict(preprocess.fit_transform(X_val))]
    bst_n_estimators = np.argmin(errors)
    print(f"bst_n_estimators: {bst_n_estimators}")

    if config["wandb"]:
        wandb.init(project="oil-well-cluster-predictor")
        pipeline_summary = {}        
        for name, step in clf.named_steps.items():
            parameters = step.get_params()
            pipeline_summary[name] = parameters
        wandb.config.update(pipeline_summary)
        results = pd.DataFrame(grid_search.cv_results_).filter(regex="^mean_test|train_test").iloc[grid_search.best_index_].to_dict()
        wandb.summary.update(results)
    models_folder = config['models_folder']
    model_filepath = os.path.join(models_folder, 'model_bst.pkl')
    joblib.dump(best_model, model_filepath)
    print(f"Best model is saved successfully at: {model_filepath}")

if __name__ == "__main__":
    # Get configuration file path from command line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)
    config_file = sys.argv[1]
    main(config_file)

