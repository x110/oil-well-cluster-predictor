import json
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def load_and_merge_data(timeseries_file, well_file, output_file):

    df_timeseries = pd.read_csv(timeseries_file, parse_dates=['date'], index_col=0)
    df_well = pd.read_csv(well_file, index_col=0).reset_index(drop=True)
    
    df_merged = pd.merge(df_timeseries, df_well, on='well')
    
    df_merged.to_csv(output_file, index=False)
    
    print(f"Merged data saved to {output_file}")
    return df_merged


def create_train_test_split(data_file, target_column, output_dir="../dataset/interm", test_size=0.2, random_state=42, overwrite=False):
    train_file = os.path.join(output_dir, "train.csv")
    test_file = os.path.join(output_dir, "test.csv")
    
    # Check if train and test files already exist
    if not overwrite and os.path.exists(train_file) and os.path.exists(test_file):
        print("Train and test datasets already exist. Skipping creation. Use 'overwrite=True' to overwrite.")
        return
    
    # Load merged DataFrame from file
    df = pd.read_csv(data_file, index_col=0)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform stratified shuffle split
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(df, df[target_column]):
        train_wells_ids = df.iloc[train_index]
        test_wells_ids = df.iloc[test_index]
        
        # Filter train and test data
        df_train = df[df['well'].isin(train_wells_ids['well'])]
        df_test = df[df['well'].isin(test_wells_ids['well'])]
        
        # Save train and test data to CSV files
        df_train.to_csv(train_file, index=False)
        df_test.to_csv(test_file, index=False)
        
        print(f"Train and test datasets saved to '{train_file}' and '{test_file}'")


if __name__ == "__main__":
    config_file = "config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    timeseries_file = config["timeseries_file"]
    well_file = config["well_file"]
    merged_file = config["merged_file"]
    data_interm_path = config["data_interm_path"]

    #df_merged = load_and_merge_data(timeseries_file, well_file, merged_file)
    #print("Data loaded, merged, and saved successfully.")
    #print(df_merged.head())
    
    create_train_test_split(data_file=well_file,
     target_column="cluster",
     output_dir=data_interm_path,
     overwrite=True)