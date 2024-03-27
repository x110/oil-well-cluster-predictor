import pandas as pd
import numpy as np
from prophet import Prophet
from scipy.signal import spectrogram
pd.set_option('future.no_silent_downcasting', True)

def data_formating(X):
    exploded = X.explode(['date', 'value'])
    exploded['date'] = pd.to_datetime(exploded['date'])
    return exploded

def generate_monthly_data(df,groupby_col, date_col='date'):
    result_dfs = []
    for well, data in df.groupby(groupby_col, observed=True):
        data = data.set_index(date_col)
        idx = pd.date_range(data.index.min(), data.index.max(), freq='MS')
        data = data.reindex(idx)
        data = data.reset_index()
        data = data.rename(columns={'index':date_col})
        data[groupby_col] = well
        result_dfs.append(data)
    result_df = pd.concat(result_dfs)
    result_df = result_df.reset_index(drop=True)
    return result_df

def pad_groups_with_zeros(df, group_col,value_col):
    well_counts = df[group_col].value_counts()
    max_samples = well_counts.max()
    grouped = df.groupby(group_col, observed = True)
    padded_dfs = []
    for name, group in grouped:
        num_samples = len(group)
        num_zeros_to_pad = max_samples - num_samples
        zeros_to_pad = pd.DataFrame({group_col: [name] * num_zeros_to_pad, value_col: [0.0] * num_zeros_to_pad})
        padded_group = pd.concat([group, zeros_to_pad], ignore_index=True)
        padded_dfs.append(padded_group)

    padded_df = pd.concat(padded_dfs, ignore_index=True)
    return padded_df

def calculate_well_duration(df):
    """
    Calculate the duration for each well in the provided DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing well information.
    
    Returns:
    - DataFrame: A DataFrame containing the duration (difference between first and last dates) for each well.
    """
    well_dates = df.groupby('well')['date'].agg(['first', 'last'])
    well_dates['duration'] = well_dates['last'] - well_dates['first']
    return well_dates

def add_study_dates(df, years):
    """
    Add study start and end dates to the DataFrame based on the maximum date recorded for each well.

    Parameters:
    - df (DataFrame): The DataFrame to which study dates will be added.
    - years (int): Number of years to subtract from the study end date to calculate the study start date.
    
    Returns:
    - DataFrame: A DataFrame with study start and end dates added as columns.
    """
    study_dates = pd.DataFrame()
    study_dates['study_end_date'] = df.groupby('well')['date'].max()
    study_dates['study_start_date'] = study_dates['study_end_date'] - pd.DateOffset(years=years)
    study_dates.reset_index(inplace=True)
    df = df.merge(study_dates, on='well')
    return df

def filter_date_range(df):
    """
    Filter the DataFrame to include only the rows within the study start and end dates.

    Parameters:
    - df (DataFrame): The DataFrame to be filtered.

    Returns:
    - DataFrame: The filtered DataFrame containing only rows within the study start and end dates.
    """
    mask = (df['date'] >= df['study_start_date']) & (df['date'] <= df['study_end_date'])
    df = df[mask]
    return df

def create_date_range(df):
    """
    Create a DataFrame containing all dates within the desired range for each well and merge it with the original DataFrame.

    Parameters:
    - df (DataFrame): The original DataFrame.

    Returns:
    - DataFrame: The merged DataFrame containing all dates within the desired range for each well.
    """
    # Create a DataFrame containing all dates within the desired range for each well
    all_dates = pd.DataFrame()
    for well, group in df.groupby('well'):
        start_date = group['study_start_date'].iloc[0]
        end_date = group['study_end_date'].iloc[0]
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        all_dates = pd.concat([all_dates, pd.DataFrame({'date': date_range, 'well': well})])

    # Merge the DataFrame containing all dates with the original DataFrame
    df = pd.merge(all_dates, df, on=['date', 'well'], how='left')
    return df

def add_months_elapsed(df):
    """
    Add a column to the DataFrame representing the number of months elapsed since the study start date.

    Parameters:
    - df (DataFrame): The DataFrame to which the 'months_elapsed' column will be added.

    Returns:
    - DataFrame: The DataFrame with the 'months_elapsed' column added.
    """
    # Calculate the number of months elapsed since the study start date
    df['months_elapsed'] = df['date'].dt.to_period('M').astype(int) - df['study_start_date'].dt.to_period('M').astype(int)
    
    return df

def fill_missing_values(df, group_column='well', time_column='date'):
    """
    Fill missing values in a time series DataFrame grouped by a specific column.

    Parameters:
        df (DataFrame): The input DataFrame containing time series data.
        group_column (str): The name of the column used for grouping (default is 'well_id').
        time_column (str): The name of the column containing timestamps (default is 'timestamp').

    Returns:
        DataFrame: The DataFrame with missing values filled within each group.
    """
    # Sort the DataFrame by group_column and time_column
    df_sorted = df.sort_values(by=[group_column, time_column])
    
    # Group the DataFrame by group_column
    grouped = df_sorted.groupby(group_column)
    
    # Define a function to fill missing values within each group
    def fill_missing(group):
        # Forward fill missing values within the group
        group_filled = group.infer_objects(copy=False).ffill()
        # Backward fill any remaining missing values
        group_filled = group_filled.bfill()
        return group_filled
    
    # Apply the fill_missing function to each group
    df_filled = grouped.apply(fill_missing)
    
    # Reset the index
    df_filled.reset_index(drop=True, inplace=True)
    
    return df_filled

def pivot_and_merge(df,column='value'):
    """
    Pivot the DataFrame on 'months_elapsed' and merge it with the original DataFrame on 'well'.

    Parameters:
    - df (DataFrame): The original DataFrame.

    Returns:
    - DataFrame: The merged DataFrame with pivoted columns.
    """
    # Pivot the DataFrame while retaining other columns
    pivot_df = df.pivot_table(index='well', columns='months_elapsed', values=column)

    #rename columns
    pivot_df.columns = [f'{column}_{col}' if col != 'well' else col for col in pivot_df.columns]

    # Drop 'value' and 'months_elapsed' columns and remove duplicates
    #df = df.drop(columns=[column, 'months_elapsed']).drop_duplicates()

    # Merge the pivoted DataFrame with the original DataFrame on the 'well' column
    df = pd.merge(df, pivot_df, on='well')

    return df

def calculate_statistical_features(df,column='value'):
    """
    Calculate statistical features for each well in the input DataFrame and merge them back into the original DataFrame.

    Parameters:
    - df (DataFrame): Input DataFrame containing columns 'well' and 'value', where 'well' represents the well ID and 'value' represents the value associated with each observation.

    Returns:
    - df (DataFrame): DataFrame with statistical features added for each well.
    """
    statistical_features = df.groupby('well')[column].agg(['mean', 'std', 'min', 'max', 'median', 'skew'])
    statistical_features['kurtosis'] = df.groupby('well')[column].apply(pd.Series.kurtosis)
    statistical_features.columns = [f"{column}_{stat}" for stat in statistical_features.columns]
    statistical_features.reset_index(inplace=True)
    df = df.merge(statistical_features, on='well')
    return df

def remove_bias(df):
    """
    Remove bias from each term in the time series for each well.
    
    Args:
    - df: DataFrame containing a time series with 'well' and 'value' columns
    
    Returns:
    - df_demeaned: DataFrame with bias removed from each term in the time series for each well
    """
    # Calculate the mean value for each well
    well_means = df.groupby('well')['value'].transform('mean')
    
    # Remove the bias for each well
    df_demeaned = df.copy()  # Make a copy of the DataFrame to avoid modifying the original
    df_demeaned['value_demeaned'] = df['value'] - well_means
    
    return df_demeaned
 
def drop_mostly_zero_wells(df, threshold=0.9):
    """
    Drop time series for each well in a pandas DataFrame if the 'value' column is mostly zero.

    Args:
    - df: DataFrame containing time series data with 'well' and 'value' columns
    - threshold: Threshold to consider a time series mostly zero (default: 0.9)

    Returns:
    - df_filtered: DataFrame with time series for mostly zero wells dropped
    """
    # Calculate the proportion of zeros in the 'value' column for each well
    zero_proportions = df.groupby('well')['value'].apply(lambda x: (x == 0).mean())

    # Identify wells where the proportion of zeros exceeds the threshold
    zero_wells = zero_proportions[zero_proportions > threshold].index

    # Drop time series for mostly zero wells
    df_filtered = df[~df['well'].isin(zero_wells)]

    return df_filtered

def smooth_dataframe(df, column, window=5):
    """
    Smooth each well's time series while maintaining the date axis using rolling mean.

    Args:
    - df: DataFrame containing time series data with 'date' and 'well' columns
    - column: Name of the column containing the time series data to be smoothed
    - window: Size of the moving window for rolling mean (default: 5)

    Returns:
    - smoothed_df: DataFrame with the smoothed time series for each well
    """
    # Group by 'well' and apply rolling mean to each group
    smoothed_df = df.groupby('well').apply(lambda group: group[[column]].rolling(window=window, min_periods=1, center=True).mean()).reset_index(level=0, drop=True)
    
    return smoothed_df

def calculate_stft(y, nperseg=8, noverlap=None, fs=1):
    """
    Calculate the Short-Time Fourier Transform (STFT) of a signal.

    Parameters:
    - y : array_like
        Input signal.
    - nperseg : int, optional
        Length of each segment for STFT calculation. Default is 8.
    - noverlap : int, optional
        Number of points to overlap between segments. Default is None, which sets noverlap to nperseg//2.
    - fs : float, optional
        Sampling frequency of the signal y. Default is 1.

    Returns:
    - Sxx : ndarray
        STFT result, flattened to a 1-D array.
    """
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, Sxx = spectrogram(y, fs=fs, window='hamming', nperseg=nperseg, noverlap=noverlap)
    return Sxx.flatten()

def expand_column(df, column_name):
    """
    Expand a column containing nested data into separate columns.

    Parameters:
    - df : pandas.DataFrame
        Input DataFrame.
    - column_name : str
        Name of the column to expand.

    Returns:
    - pandas.DataFrame
        DataFrame with the specified column expanded into separate columns.
    """
    df_freq = df[column_name].apply(pd.Series)
    df_freq.columns = [f'{column_name}_{column}' for column in df_freq.columns]
    return pd.concat([df.drop(columns=[column_name]), df_freq], axis=1)

def unused_compute_stft_for_each_well(df, fs=1, nperseg=8, noverlap=None, plot=False):
    """
    Compute the Short-Time Fourier Transform (STFT) for each well's time series data.
    
    Parameters:
        df (DataFrame): DataFrame containing time series data for each well.
                        Columns should include 'well', 'date', and 'value'.
                        'well' column contains the well names.
                        'date' column contains the dates.
                        'value' column contains the corresponding values.
        fs (float): Sampling frequency of the time series data (default is 1).
        nperseg (int): Length of each segment (window) used in the STFT calculation (default is 256).
        noverlap (int or None): Number of overlapping samples between adjacent segments.
                                 If None, defaults to half of nperseg (default is None).
    
    Returns:
        stft_results (dict): Dictionary containing the STFT results for each well.
                             Keys are the well names, and values are tuples (frequencies, times, spectrograms).
                             - frequencies: Array of frequencies.
                             - times: Array of time points.
                             - spectrograms: 2D array of spectrogram values.
    """
    stft_results = {}
    
    # Group DataFrame by 'well' column
    grouped = df.groupby('well')
    
    # Iterate over each group (well)
    for well, group_df in grouped:        
        # Sort DataFrame by 'date' column
        group_df.sort_values(by='date', inplace=True)
        
        # Compute STFT for the 'value' column of the current well
        f, t, Sxx = spectrogram(group_df['value'], fs=fs, nperseg=nperseg, noverlap=noverlap)
        
        # Store STFT results in the dictionary
        stft_results[well] = (f, t, Sxx)
    if plot:
        # Plot spectrogram
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')  # Convert to dB
        plt.colorbar(label='Power (dB)')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (months)')
        plt.title('STFT Spectrogram')
        plt.show()
    
    return stft_results

def stft_feature_extraction(x, hop_length=4, n_fft=8, sr=1):
    Sxx = stft(x, hop_length=hop_length, n_fft=n_fft, window='hamming')
    S, phase = librosa.magphase(Sxx)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S).mean()
    spectral_centroid = librosa.feature.spectral_centroid(S=S).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr).mean()
    spectral_flatness = librosa.feature.spectral_flatness(S=S).mean()
    poly_features_0 = librosa.feature.poly_features(S=S, order=0).mean()
    poly_features_1 = librosa.feature.poly_features(S=S, order=1).mean()
    poly_features_2 = librosa.feature.poly_features(S=S, order=2).mean()
    
    return pd.Series({
        'spectral_bandwidth_mean': spectral_bandwidth,
        'spectral_centroid_mean': spectral_centroid,
        'spectral_rolloff_mean': spectral_rolloff,
        'spectral_flatness_mean': spectral_flatness,
        'poly_features_0_mean': poly_features_0,
        'poly_features_1_mean': poly_features_1,
        'poly_features_2_mean': poly_features_2
    })

def calculate_dc_magnitude(df,column='value',fs=1,ratio=False):
    signal = df[column]
    fft_result = np.fft.fft(signal)
    N = len(fft_result)
    freqs = np.fft.fftfreq(N, d=fs)[:N//2]
    magnitude_spectrum = np.abs(fft_result)[:N//2]
    max_magnitude = np.max(magnitude_spectrum)
    if ratio:
        magnitude_spectrum = magnitude_spectrum/max_magnitude
    dc_frequency_magnitude = np.abs(magnitude_spectrum[0])
    df = df.assign(dc_frequency_magnitude = dc_frequency_magnitude)
    return df

def calculate_fft(signal,ratio=True):
    fft_result = np.fft.fft(signal)
    N = len(fft_result)
    magnitude_spectrum = np.abs(fft_result)[:N//2]
    max_magnitude = np.max(magnitude_spectrum)
    if ratio:
        magnitude_spectrum = magnitude_spectrum/(max_magnitude+1e-12)
    return magnitude_spectrum[:10]
def calculate_df_fft(df, column='value'):
    magnitude_spectrum = df.groupby('well').apply(lambda group: calculate_fft(group[column])).reset_index(level=0)
    magnitude_spectrum = magnitude_spectrum.rename(columns={0: 'freq'})
    merged_df = pd.merge(df, magnitude_spectrum, on='well')
    return merged_df

def calculate_prophet_forecast(df_well):
    df = (df_well
        .filter(['date','value'])
        .rename(columns={'date':'ds','value':'y'})
        )
    m = Prophet()
    m.fit(df)
    #future = m.make_future_dataframe(periods=12*3, freq='MS')
    dates = df.filter(['ds'])
    forecast = m.predict(dates)
    forecast['yhat'] = forecast['yhat'].apply(lambda x: max(0, x))
    forecast = forecast.filter(['ds', 'yhat', 'trend', 'yearly'])
    merged_df = pd.merge(df_well, forecast, left_on='date', right_on='ds').drop(columns = 'ds')
    return merged_df

def calculate_prophet_forecast_wells(df):
    return df.groupby('well').apply(lambda df_well: calculate_prophet_forecast(df_well)).reset_index(level=0, drop=True)

def calculate_dc_magnitude_wells(df):
    return df.groupby('well').apply(lambda df_well: calculate_dc_magnitude(df_well)).reset_index(level=0, drop=True)

def pad_array(arr, num_columns):
    """
    Pad or trim the input array to have num_columns columns.

    Parameters:
        arr (numpy.ndarray or pandas dataframe): Input array.
        num_columns (int): Desired number of columns in the output array.

    Returns:
        numpy.ndarray: Padded or trimmed array with original number of rows and num_columns columns.
    """
    if arr.shape[1] == num_columns:
        return arr.copy()
    elif arr.shape[1] > num_columns:
        return arr[:, :num_columns]
    else:
        m = arr.shape[0]
        padded_arr = np.zeros((m, num_columns))
        padded_arr[:arr.shape[0], :arr.shape[1]] = arr
        return padded_arr
