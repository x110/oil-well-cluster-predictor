import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import encode_labels
from data import load_data

def evaluate_model(filepath, test_data_path):
    # Load the model and class names from the file
    model= joblib.load(filepath)
    
    df_test = load_data(test_data_path)
    
    X_test = df_test.drop(columns={'cluster'})
    y_test = df_test['cluster']

    # Make predictions using the loaded model
    y_pred = model.predict(X_test)

    # Print classification report
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print()

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Print confusion matrix
    print("Confusion Matrix:")
    print(cm)
