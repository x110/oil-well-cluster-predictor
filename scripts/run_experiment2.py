import pandas as pd
import json
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
import sys
sys.path.append('./src')
from data import load_data
from data_transformer import DataFormattingTransformer, GenerateMonthlyDataTransformer, Imputer, UndoDataFormattingTransformer, PadArrayTransformer

config_file = "./config.json"
with open(config_file, 'r') as f:
    config = json.load(f)

train_data_path = config["train_data_path"]

df = load_data(train_data_path) 

X = df.drop(columns='cluster')
y = df.cluster

clf = Pipeline([
    ('data_formatting', DataFormattingTransformer()),
    ('monthly_data_generation', GenerateMonthlyDataTransformer(groupby_col='well')),
    ('undo_data_formatting', UndoDataFormattingTransformer()),
    ('simple_imputer',Imputer()),
    ('pad_array',PadArrayTransformer(num_cols=500)),
    ('preprocessor', StandardScaler()),
    ("kmeans", KMeans())
])
param_grid = [{'kmeans__n_clusters': [4]
               }]
    
grid_search = GridSearchCV(clf,
                            param_grid,
                            cv=5,
                            #scoring=['precision_weighted','recall_weighted','f1_weighted', 'balanced_accuracy'],
                            #refit='f1_weighted',
                            n_jobs=-1,
                            #error_score='raise'
                            )
    
grid_search.fit(X)
results = pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_params_)
best_model = grid_search.best_estimator_
best_score = grid_search.best_score_
print(f"Best parameters for: {best_model}")
print(f"Best score for: {best_score}")