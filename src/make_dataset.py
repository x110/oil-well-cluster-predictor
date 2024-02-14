def process_data(data_path, processed_data_path):
    YEARS = 5
    WINDOW = 6

    df = (pd.read_csv(data_path, parse_dates=['date'])  
          .pipe(add_study_dates, years=YEARS)
          .pipe(filter_date_range)
          .pipe(create_date_range)
          .pipe(fill_missing_values)
          .pipe(drop_mostly_zero_wells)
          .assign(smoothed_value=lambda df: smooth_dataframe(df, column='value', window=WINDOW))
          .pipe(calculate_statistical_features, column='smoothed_value')
          .pipe(add_months_elapsed)
          .pipe(pivot_and_merge, column='trend')
          .drop(columns={'date', 'study_start_date', 'study_end_date', 'value', 'smoothed_value', 'months_elapsed', 'yhat', 'trend', 'yearly'})
          .drop_duplicates()
          )
    

    df_freq = (pd.read_csv(data_path, parse_dates=['date'])
                .pipe(add_study_dates, years=YEARS)
                .pipe(filter_date_range)
                .pipe(create_date_range)
                .pipe(fill_missing_values)
                .pipe(drop_mostly_zero_wells)
                .pipe(calculate_dc_magnitude_wells)
                .pipe(calculate_df_fft)
                )

    df_freq_expanded = df_freq.filter(['freq','well']).pipe(expand_column, column_name='freq')
    df_freq = df_freq.drop(columns = {'date','value','cluster','study_end_date','study_start_date','freq'})
    df_freq = df_freq.merge(df_freq_expanded, on='well')

    df = df.merge(df_freq, on='well')
    df.to_csv(processed_data_path, index=False)
    return df

train_data_path = '../dataset/interm/train.csv'
processed_train_data_path = '../dataset/processed/train.csv'
test_data_path = '../dataset/interm/test.csv'
processed_test_data_path = '../dataset/processed/test.csv'

df_train = process_data(train_data_path, processed_train_data_path)
df_test = process_data(test_data_path, processed_test_data_path)

df_train = pd.read_csv(processed_train_data_path)
df_test = pd.read_csv(processed_test_data_path)

display(df_train.head())
df_test.head()
