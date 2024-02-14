from sklearn.ensemble import RandomForestClassifier

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