import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import encode_labels

def evaluate_model(filepath, test_data_path):
    # Load the model and class names from the file
    model, class_names = joblib.load(filepath)

    # Extract features and labels from test data
    df_test = pd.read_csv(test_data_path)
    y0_test = df_test['cluster']
    X_test = df_test.drop(columns={'cluster', 'well'})
    y_test, _ = encode_labels(y0_test)

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