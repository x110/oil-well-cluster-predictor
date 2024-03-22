import pandas as pd
import json
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import f1_score
import sys
sys.path.append('./src')
from data import load_data
from transformations import data_formating, generate_monthly_data, pad_groups_with_zeros
from sklearn.impute import SimpleImputer

import numpy as np

def adjust_array(arr, m):
    # Check if reshaping is possible
    if arr.shape[1] == 500 and arr.shape[0] == m:
        return arr
    # Trim if larger
    elif arr.shape[1] > 500:
        return arr[:, :500]
    # Pad if smaller
    else:
        padded_arr = np.zeros((m, 500))
        padded_arr[:arr.shape[0], :arr.shape[1]] = arr
        return padded_arr
    

class DataFormattingTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        index = X.well.drop_duplicates().to_list()
        df = data_formating(X)
        df['well'] = pd.Categorical(df['well'], categories=index, ordered=True)
        df= df.sort_values(by='well')
        return df

class GenerateMonthlyDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, groupby_col, date_col='date'):
        self.groupby_col = groupby_col
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        index = X.well.drop_duplicates().to_list()
        df = generate_monthly_data(X, self.groupby_col, self.date_col)
        df['well'] = pd.Categorical(df['well'], categories=index, ordered=True)
        df = df.sort_values(by=['well', 'date'])
        df = df.pivot(index='well', columns='date', values='value')
        df = df.fillna(0)
        df = df.reindex(sorted(df.columns), axis=1)
        df = adjust_array(df, df.shape[0])
        return df

class PadGroupsWithZerosTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, value_col):
        self.group_col = group_col
        self.value_col = value_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        index = X.well.drop_duplicates().to_list()
        df = pad_groups_with_zeros(X, self.group_col, self.value_col)
        df['well'] = pd.Categorical(df['well'], categories=index, ordered=True)
        df= df.sort_values(by='well')

        df = df.drop_duplicates()
        df = df.pivot(index='well', columns='date', values='value')
        df = df.reindex(sorted(df.columns), axis=1)
        df.head()
        return df


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
    #('pad_with_zeros', PadGroupsWithZerosTransformer(group_col='well', value_col='value')),
    ('kbins',KBinsDiscretizer(n_bins=22, encode='ordinal', strategy='quantile')),
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
                            return_train_score=True,
                            #error_score='raise'
                            )
    
grid_search.fit(X)
results = pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_params_)
best_model = grid_search.best_estimator_
best_score = grid_search.best_score_
print(f"Best parameters for: {best_model}")
print(f"Best score for: {best_score}")
import matplotlib.pyplot as plt
cluster_labels = best_model.predict(X)
# Assuming you have already performed clustering and have cluster_labels
# Visualize the clusters
plt.figure(figsize=(10, 6))

# Assign colors to clusters for better visualization
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
p  = Pipeline([
    ('data_formatting', DataFormattingTransformer()),
    ('monthly_data_generation', GenerateMonthlyDataTransformer(groupby_col='well')),
    ('kbins',KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='quantile'))
])
X_transformed = p.fit_transform(X)
# Plot each time series with its predicted cluster color
for i in range(4):
    cluster_data = X_transformed[y == i]
    for ts in cluster_data:
        plt.plot(ts, color=colors[i], alpha=0.5)

plt.title('Clustering of Time Series Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
plt.show()
