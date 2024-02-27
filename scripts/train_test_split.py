import json
import sys
sys.path.append('./src')
from data import create_train_test_split

def main(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    timeseries_file = config["timeseries_file"]
    well_file = config["well_file"]
    interm_dir = config["data_interm_path"]

    create_train_test_split(timeseries_file, well_file, output_dir=interm_dir, overwrite=True)

if __name__ == "__main__":
    # Get configuration file path from command line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)
    config_file = sys.argv[1]
    main(config_file)
