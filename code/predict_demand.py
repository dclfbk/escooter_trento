import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import holidays
import pmdarima as pm
from pmdarima.metrics import smape
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


''' 
Functions to train ARIMA/SARIMA models predicting number of trips in the city.
Frequency of prediction (e.g., hourly prediction, daily prediction, etc.) can be changed based on the aggregation frequency specified when building the dataframe
(i.e., aggr_freq parameter in the temporal_aggregation function).
'''


def build_hourly_dataframe(data_path, add_weather_info=True):
    '''
    Input: 
        - data_path: path of the parquet file containing the e-scooters data, augmented with chosen predictors
            (predictors added in code\escooter_demand_prediction\code\cleaning\2_augment_data.py)
        - add_weather_info: whether or not to add information related to weather conditions (cm of snow, wind speed in kmph, temperature in Celsius, visibility 0-10, precipitation in mm)

    Aggregate data at hour level, add missing timestamps, and optionally additional information related to weather conditions at each timestep.
    Return final dataframe.
    '''
    # read data
    df = pd.read_parquet(data_path)
    # truncate at hour level & remove timezone
    df['date_hour'] = df['trip_start'].dt.floor('h').dt.tz_convert(None)

    # compute counts
    df = df.groupby(['date_hour']).aggregate({'leisure':['sum'], 'public_services':['sum'], 'building':['sum'], 
                'education':['sum'], 'food':['sum'], 'transport':['sum'], 'finantial':['sum'], 'healthcare':['sum'], 'office':['sum'], 'shop':['sum'], 'tourism':['sum'], 
                'unique_id':['count']}).reset_index()
    df.columns = df.columns.droplevel(1)
    df.rename(columns = {'unique_id':'trips_count'}, inplace = True)

    # add missing timestamps
    all_times = pd.DataFrame({'date_hour': pd.date_range(df.date_hour.min(), df.date_hour.max(), freq='H')}) # all timesteps
    df_all_times = pd.merge(all_times, df, left_on='date_hour', right_on='date_hour', how='left') # join
    # fill NaN in trips_count with 0 (if timestamp not in original df, then zero trips at that time)
    df_all_times.trips_count = df_all_times.trips_count.fillna(-1)
    # for POIs categories, fill NaN with respective columns median
    for col in ['leisure', 'public_services', 'building','education', 'food', 'transport', 'finantial', 'healthcare', 'office', 'shop', 'tourism']:
        md = df_all_times[col].dropna().median()
        df_all_times[col] = df_all_times[col].fillna(md)

    if add_weather_info == True:
        weather = pd.read_csv('C:/Users/luisa/Desktop/thesis_project/data/weather_info.csv')[['date_time', 'totalSnow_cm', 'windspeedKmph', 'tempC', 'visibility', 'precipMM']]
        weather['date_time'] = pd.to_datetime(weather.date_time, format="%Y-%m-%d %H:%M:%S")
        df_all_times = df_all_times.merge(weather, left_on=['date_hour'], right_on=['date_time']).drop(['date_time'], axis=1)

    return df_all_times.sort_values(by=['date_hour'])




def temporal_aggregation(df, aggr_freq):
    '''
    Input:
        - df: dataframe (e.g, obtained via build_hourly_dataframe function)
        - aggr_freq: frequency according to which data must be aggregated 

    Return aggregated dataframe according to the frequency passed as input.
    '''
    # aggregating multiple fields for each hour
    df = df.resample(aggr_freq, on='date_hour').agg({'leisure':'mean', 'public_services':'mean', 'building':'mean',
              'education':'mean', 'food':'mean', 'transport':'mean', 'finantial':'mean', 'healthcare':'mean', 'office':'mean',
              'shop':'mean', 'tourism':'mean', 'trips_count':'sum','totalSnow_cm':'mean', 'windspeedKmph':'mean', 'tempC':'mean', 
              'visibility':'median', 'precipMM':'mean'})

    # add holiday (boolean)
    df['holiday'] = [df.index[i] in holidays.Italy() for i in range(len(df))] 
    df['holiday'] = df.holiday.astype('int').to_numpy() 
    # add weekday as number (0-6)
    df['day'] = df.index.weekday 
    return df




def normalize(train, test):
    ''' 
    Function to normalize column-wise train and test set (based on train avg and sd).
    Return normalized train and test data, and avg and sd used to normalize trips_count.
    '''
    for col in train.columns.drop(['trips_count']):
        avg = train[col].mean()
        std = train[col].std()
        train[col] = (train[col] - avg) / std
        test[col] = (test[col] - avg) / std
    avg, sd = train.trips_count.mean(), train.trips_count.std()
    train['trips_count'] = (train.trips_count - avg) / sd
    test['trips_count'] = (test.trips_count - avg) / sd
    return train, test, avg, sd


def denorm_cnt(train, test, avg, sd):
    # de-normalize target variable
    train.trips_count = (train.trips_count * sd) + avg
    test.trips_count = (test.trips_count * sd) + avg
    return train, test



def train_test_split(data, split_timestep = pd.Timestamp('2022-04-04 00:00:00')):
    ''' 
    Input:
        - data: dataframe
        - split_timestep: pandas Timestamp according to which split train and test data
    Split train and test set, given a date, as pandas Timestamp, according to which to split.
    Return train and test set.
    '''
    train = data[data.index < split_timestep] 
    test = data[data.index >= split_timestep]
    return train, test




def fit_arima(train, start_p=2, start_q=0, max_p=3, max_q=3, seasonality=False):
    ''' 
    Input:
        - train: train set
        - start_p, max_p: min and max value for the p parameter (number of lags to include in the prediction)
        - start_q, max_q: min and max value for the q parameter (number of past errors to include in the prediction)
        - seasonality: boolean, whether to consider seasonality (SARIMA) or not (ARIMA)
    Fit ARIMA model
    '''
    return pm.auto_arima(train['trips_count'], # time-serie data to estimate
                        train[train.columns.drop('trips_count')], # list of exogenous variables to use for prediction (optional)
                        start_p=start_p, # starting number of lags (AR part) -> from PACF
                        start_q=start_q, # the order of moving average (MA part) -> from ACF
                        d=None, # order of first-differencing; if None, the value is determined by the model based on the results of the test selected (below: adf aka Augmented Dickey-Fuller))
                        test='adf', # Augmented Dickey-Fuller Test to find optimal 'd' parameter
                        max_p=max_p, 
                        max_q=max_q, 
                        # m=1, # frequency of series: 1 aka annual (if m==1, seasonal is set to FALSE automatically) 
                        seasonal=seasonality, # standard ARIMA, without seasonality
                        suppress_warnings=True,
                        stepwise=True # use the stepwise algorithm to dentify the optimal model parameters (significantly faster)
                    )




def predict_on_test(ARIMA_model, test):
    ''' 
    Input: 
        - ARIMA_model: trained model
        - test: test data

    Fit model on test data and return fitted series, with confidence interval
    '''
    # Forecast
    n_periods = len(test)
    fitted, confint = ARIMA_model.predict(n_periods = n_periods, X = test[test.columns.drop('trips_count')], return_conf_int = True) # X -> exogenous variables
    return fitted, confint




def plot_forecast(fitted, confint, train, test, aggr_freq, trips_cnt_avg, trips_cnt_sd, model_name='ARIMA'):
    ''' 
    Input: 
        - fitted: predictions on test set
        - confint: lower and upper bounds of the predictions
        - train: train data
        - test: test data
        - aggr_freq: temporal frequency of counts
        - trips_cnt_avg, trips_cnt_sd: training average and standard deviation of trips_count
        - model_name: optional model name

    Plot historic trend of trips over time (train data), predictions on test set and their confidence interval.
    '''

    n_periods = len(test)
    index_of_fc = pd.date_range(test.index[0], periods = n_periods, freq=aggr_freq) # freq='H' --> pred. per hour # + pd.DateOffset(hour=aggr_freq)

    # make series for plotting purpose
    fitted_series = pd.Series(fitted, index=index_of_fc)
    # confidence interval
    lower_series = pd.Series(confint[:, 0], index=index_of_fc) # lower bound conf. int.
    upper_series = pd.Series(confint[:, 1], index=index_of_fc) # upper bound conf. int.

    

    # de-norm
    fitted_series = ( fitted_series * trips_cnt_sd ) + trips_cnt_avg
    train_denorm, test_denorm = denorm_cnt(train, test, trips_cnt_avg, trips_cnt_sd)
    lower_series = ( lower_series * trips_cnt_sd ) + trips_cnt_avg
    upper_series = ( upper_series * trips_cnt_sd ) + trips_cnt_avg

    
    
    # Plot
    plt.figure(figsize=(15,7))
    plt.plot(train_denorm.trips_count, color='#1f76b4', label='Training Set')
    plt.plot(fitted_series, color='darkgreen', label='Predicted Value')

    plt.plot(test_denorm.trips_count, color='green', alpha=0.3, label='Actual Test Value')

    plt.fill_between(lower_series.index, 
                    lower_series, 
                    upper_series, 
                    color='k', alpha=.15)
    plt.legend(loc='upper left', labelcolor='linecolor')
    plt.title(model_name+" Forecasting")
    plt.show()

    # Plot with rolling mean
    plt.figure(figsize=(15,7))
    plt.plot(train_denorm.trips_count.rolling(window=12).mean(), color='#1f76b4', label='Training Set')
    plt.plot(fitted_series.rolling(window=12).mean(), color='darkgreen', label='Predicted Value')

    plt.plot(test_denorm.trips_count.rolling(window=12).mean(), color='green', alpha=0.2, label='Actual Test Value')

    plt.fill_between(lower_series.index, 
                    lower_series, 
                    upper_series, 
                    color='k', alpha=.15)
    plt.legend(loc='upper left', labelcolor='linecolor')
    plt.title(model_name+" Forecasting")
    plt.show()

    # ZOOM ON PREDICTED VS TRUE VALUES
    
    plt.plot(test_denorm.trips_count.rolling(window=12).mean(), color='orange', label='Actual Values', alpha=0.5)
    plt.plot(fitted_series.rolling(window=12).mean(), color='green', label='Predicted Values', alpha=0.9)
    plt.legend(loc='upper left', labelcolor='linecolor')
    plt.title(model_name+" Forecasting - Rolling Mean")
    plt.show()

    plt.plot(test_denorm.trips_count, color='orange', label='Actual Values', alpha=0.5)
    plt.plot(fitted_series, color='green', label='Predicted Values', alpha=0.9)
    plt.legend(loc='upper left', labelcolor='linecolor')
    plt.title(model_name+" Forecasting")
    plt.show()
