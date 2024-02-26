import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_train_test_split(timeseries_file, well_file, output_dir= "./dataset/interm", overwrite = False):
    train_file = os.path.join(output_dir, "train.csv")
    test_file = os.path.join(output_dir, "test.csv")

    if not overwrite and os.path.exists(train_file) and os.path.exists(test_file):
        print("Train and test datasets already exist. Skipping creation. Use 'overwrite=True' to overwrite.")
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        return df_train, df_test
    
    df_timeseries = pd.read_csv(timeseries_file, parse_dates=['date'], index_col=0)
    df_well = pd.read_csv(well_file, index_col=0).reset_index(drop=True)

    df_well_train, df_well_test, _, _ = train_test_split(df_well,
                                                df_well['cluster'],
                                                test_size=.2,
                                                random_state=42,
                                                stratify=df_well['cluster'])

    df_train= pd.merge(df_timeseries, df_well_train, on='well')
    df_test = pd.merge(df_timeseries, df_well_test, on='well')

    df_train.to_csv(train_file, index=False)
    df_test.to_csv(test_file, index=False)
        
    print(f"Train and test datasets saved to '{train_file}' and '{test_file}'")
    return df_train, df_test

if __name__ == "__main__":
    config_file = "config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    timeseries_file = config["timeseries_file"]
    well_file = config["well_file"]
    interm_dir= config["data_interm_path"]
    create_train_test_split(timeseries_file, well_file, output_dir=interm_dir, overwrite=False)