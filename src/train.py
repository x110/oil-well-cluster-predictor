from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from preprocessing import encode_labels
from preprocessing import preprocess_data, encode_labels
from models import classifiers
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def train_classifier(X, y, pipeline, params):
    grid_search = GridSearchCV(pipeline,
                               params,
                               cv=5,
                               scoring={'f1_weighted': 'f1_weighted', 'balanced_accuracy': 'balanced_accuracy'},
                               refit='f1_weighted',
                               n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search

def train(df_train, classifiers):
    X, y0 = preprocess_data(df_train)
    y = encode_labels(y0)

    for clf_name, clf_config in classifiers.items():
        pipeline = Pipeline([
            ('preprocessor', StandardScaler),
            ('clf', clf_config['model'])
        ])
        grid_search = train_classifier(X, y, pipeline, clf_config['params'])
    
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")
    
    return grid_search



if __name__=="__main__":
    processed_train_data_path = './dataset/processed/train.csv'
    df_train = pd.read_csv(processed_train_data_path)
    train(df_train, classifiers)
