import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from preprocessing import preprocess_data, encode_labels
import time
from data_transformer import CustomDataTransformer

def generate_unique_filename(prefix='file', extension='.pt'):
    timestamp = time.strftime('%Y%m%d%H%M%S')
    unique_filename = f"{prefix}_{timestamp}{extension}"
    return unique_filename

def train_classifier(X,y, classifiers, models_folder = './models'):

    for clf_name, clf_config in classifiers.items():
        pipeline = Pipeline([
            ('data_transformer', CustomDataTransformer()),
            ('preprocessor', StandardScaler()),
            ('clf', clf_config['model'])
        ])

        grid_search = GridSearchCV(pipeline,
                                   clf_config['params'],
                                   cv=5,
                                   scoring=['f1_weighted', 'balanced_accuracy'],
                                   refit='f1_weighted',
                                   n_jobs=-1,
                                   error_score='raise')
        grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    print(f"Best parameters: {best_model}")
    print(f"Best score : {best_score}")
    model_filepath = os.path.join(models_folder, generate_unique_filename(prefix='model', extension='.pkl'))
    joblib.dump(best_model, model_filepath)
    print(f"Best model is saved successfully at: {model_filepath}")

    return grid_search

if __name__ == "__main__":
    processed_train_data_path = './dataset/processed/train.csv'
    df_train = pd.read_csv(processed_train_data_path)
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
