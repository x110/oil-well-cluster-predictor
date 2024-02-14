from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from preprocessing import preprocess_data, encode_labels

def train_classifier(df_train, classifiers):
    X, y0 = preprocess_data(df_train)
    y = encode_labels(y0)

    for clf_name, clf_config in classifiers.items():
        
        pipeline = Pipeline([
            ('preprocessor', StandardScaler()),
            ('clf', clf_config['model'])
        ])

        grid_search = GridSearchCV(pipeline,
                               clf_config['params'],
                               cv=5,
                               scoring={'f1_weighted':'f1_weighted'},
                               refit='f1_weighted',
                               n_jobs=-1)
        grid_search.fit(X, y)
        print(f"Best parameters for {clf_name}: {grid_search.best_params_}")
        print(f"Best score for {clf_name}: {grid_search.best_score_}")
    return grid_search

if __name__=="__main__":
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
    train_classifier(df_train, classifiers)
