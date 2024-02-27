import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import preprocess_data, encode_labels

def evaluate_model(filepath, df_test):
    # Load the model and class names from the file
    model, class_names = joblib.load(filepath)

    # Extract features and labels from test data
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

if __name__ == "__main__":
    processed_test_data_path = './dataset/processed/test.csv'
    df_test = pd.read_csv(processed_test_data_path)
    #todo pass name of the model , dont hard code it
    filepath = './models/model_20240227083910.pkl'

    evaluate_model(filepath, df_test)
