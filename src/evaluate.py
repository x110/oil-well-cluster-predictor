import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from data import load_data

def evaluate_model(filepath, test_data_path):
    model= joblib.load(filepath)
    
    df_test = load_data(test_data_path)
    
    X_test = df_test.drop(columns={'cluster'})
    y_test = df_test['cluster']

    y_pred = model.predict(X_test)

    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print()

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
