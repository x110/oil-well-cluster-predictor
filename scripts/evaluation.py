import json
import sys
sys.path.append('./src')
from evaluate import evaluate_model

def main(config_file):
    # Load configuration from JSON file
    with open(config_file, 'r') as f:
        config = json.load(f)

    test_data_path = config["test_data_path"]
    model_path = config["model_path"]
    evaluate_model(model_path, test_data_path)

if __name__ == "__main__":
    # Get configuration file path from command line argument
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <config_file>")
        sys.exit(1)
    config_file = sys.argv[1]
    main(config_file)
