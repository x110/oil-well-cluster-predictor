import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import transformations as tf

class CustomDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, years=5, window=6):
        self.years = years
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        index = X.well.drop_duplicates().to_list()
        
        df = (X.explode(['date','value'])
              .assign(date = lambda df: pd.to_datetime(df['date']))
              .pipe(tf.add_study_dates, years=self.years)
              .pipe(tf.filter_date_range)
              .pipe(tf.create_date_range)
              .pipe(tf.fill_missing_values)
              #.pipe(tf.drop_mostly_zero_wells)
              .assign(smoothed_value=lambda df: tf.smooth_dataframe(df, column='value', window=self.window))
              .pipe(tf.calculate_statistical_features, column='smoothed_value')
              .pipe(tf.add_months_elapsed)
              .drop(columns={'date', 'study_start_date', 'study_end_date', 'value', 'smoothed_value', 'months_elapsed'})
              .drop_duplicates()
              )
        
        df_freq = (X.explode(['date','value'])
                   .assign(date = lambda df: pd.to_datetime(df['date']))
                   .pipe(tf.add_study_dates, years=self.years)
                   .pipe(tf.filter_date_range)
                   .pipe(tf.create_date_range)
                   .pipe(tf.fill_missing_values)
                   #.pipe(tf.drop_mostly_zero_wells)
                   .pipe(tf.calculate_dc_magnitude_wells)
                   .pipe(tf.calculate_df_fft)
                  )
        
        df_freq_expanded = df_freq.filter(['freq','well']).pipe(tf.expand_column, column_name='freq')
        df_freq = df_freq.drop(columns = {'date','value','cluster','study_end_date','study_start_date','freq'}, errors='ignore')
        df_freq = df_freq.merge(df_freq_expanded, on='well')

        df = df.merge(df_freq, on='well')
        df = df.drop_duplicates()
        df['well'] = pd.Categorical(df['well'], categories=index, ordered=True)
        df= df.sort_values(by='well')
        df = df.drop(columns={'well'})
        return df

class Transformer1(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        index = X.well.drop_duplicates().to_list()
        
        df = (X.explode(['date','value'])
              .assign(date = lambda df: pd.to_datetime(df['date']))
              #.pipe(tf.add_study_dates, years=self.years)
              #.pipe(tf.filter_date_range)
              #.pipe(tf.create_date_range)
              #.pipe(tf.fill_missing_values)
              #.pipe(tf.drop_mostly_zero_wells)
              #.assign(smoothed_value=lambda df: tf.smooth_dataframe(df, column='value', window=self.window))
              #.pipe(tf.calculate_statistical_features, column='smoothed_value')
              #.pipe(tf.add_months_elapsed)
              #.drop(columns={'date', 'study_start_date', 'study_end_date', 'value', 'smoothed_value', 'months_elapsed'})
              .drop_duplicates()
              )
        
        df_freq = (X.explode(['date','value'])
                   .assign(date = lambda df: pd.to_datetime(df['date']))
                   .pipe(tf.add_study_dates, years=self.years)
                   .pipe(tf.filter_date_range)
                   .pipe(tf.create_date_range)
                   .pipe(tf.fill_missing_values)
                   #.pipe(tf.drop_mostly_zero_wells)
                   .pipe(tf.calculate_dc_magnitude_wells)
                   .pipe(tf.calculate_df_fft)
                  )
        
        df_freq_expanded = df_freq.filter(['freq','well']).pipe(tf.expand_column, column_name='freq')
        df_freq = df_freq.drop(columns = {'date','value','cluster','study_end_date','study_start_date','freq'}, errors='ignore')
        df_freq = df_freq.merge(df_freq_expanded, on='well')

        df = df.merge(df_freq, on='well')
        df = df.drop_duplicates()
        df['well'] = pd.Categorical(df['well'], categories=index, ordered=True)
        df= df.sort_values(by='well')
        df = df.drop(columns={'well'})
        return df


