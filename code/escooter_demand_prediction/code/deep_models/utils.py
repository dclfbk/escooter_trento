import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geohash as pgh
import holidays
import warnings
warnings.filterwarnings("ignore")

## GEOHASH:
# Geohash functions, used to divide the city into cells. The dimension of the cells is given by the precision value, passed as intput.
# Highest precision means lower dimension of the cell:
# e.g., a precision equal to 6, produce cells of dimension ≤ 1.22km × 0.61km
# a precision equal to 7, produce cells of dimension ≤ 153m x 153m 
# For further information, demo and references check: http://www.movable-type.co.uk/scripts/geohash.html

def addToGeohash(m, latitude, longitude, precision):
    '''
    Add point to the mapping function connecting trips origin to the cell hash.
    The origin-point is added to the values associated to the identifier (hash) of its cell.
    '''
    p = (latitude, longitude)
    ph = pgh.encode(p[0], p[1], precision)  
    if ph not in m:
        m[ph] = []
    m[ph].append(p)
    return

def getGeohash(m, latitude, longitude, precision):
    '''
    Get geohash identifying the cell to which the passed point belongs to.
    '''
    p = (latitude, longitude)
    ph = pgh.encode(p[0], p[1], precision)
    if ph in m:
        return ph
    raise ValueError('Not found')



## BUILD DATAFRAME:

def build_df(precision=6):
    '''
    Build the dataframe used to train the ML models (without times where trips counts were zero).
    '''

    # read the data
    df = pd.read_parquet('C:/Users/luisa/Desktop/thesis_project/data/tripsv3_augmented.parquet')
    # round trip start to hour
    # df['trip_start'] = df.trip_start.round('h') # this rounds minutes above 30 to the next hour
    # truncate instead:
    df['trip_start'] = [df.at[i, 'trip_start'].replace(minute=0, second=0) for i in range(len(df))]

    # add geohash code based on the precision passed as input
    m = dict()
    gh = []
    for pt in range(len(df)):
        addToGeohash(m, df.at[pt, 'trip_origin_latitude'], df.at[pt, 'trip_origin_longitude'], precision) 
        gh.append(getGeohash(m, df.at[pt, 'trip_origin_latitude'], df.at[pt, 'trip_origin_longitude'], precision))
    df['geohash_origin'] = gh
    # decode: find center coordinates of each cell in the grid
    df['gh_centroid'] = df.apply(lambda rec: pgh.decode(rec['geohash_origin']), axis=1)

    # group the data to obtain the final dataframe 
    d = df.groupby(['gh_centroid', 'trip_start']).aggregate({'leisure':['sum'], 'public_services':['sum'], 'building':['sum'], 
                'education':['sum'], 'food':['sum'], 'transport':['sum'], 'finantial':['sum'], 'healthcare':['sum'], 'office':['sum'], 'shop':['sum'], 'tourism':['sum'], 
                'totalSnow_cm':['mean'], 'windspeedKmph':['mean'], 'tempC':['mean'], 'visibility':['mean'], 'precipMM':['mean'], 'count_bus_multimodality':['mean'], 
                'unique_id':['count']}).reset_index()
    d.columns = d.columns.droplevel(1)
    d.rename(columns = {'unique_id':'trips_count'}, inplace = True)


    # separate latitude and longitude of the cells center in two different columns
    d['gh_centroid_lat'] = [d.at[i, 'gh_centroid'][0] for i in range(len(d))]
    d['gh_centroid_lon'] = [d.at[i, 'gh_centroid'][1] for i in range(len(d))]

    # drop redundant information (already have lat and long in two separate columns)
    d = d.drop(['gh_centroid'], axis=1)

    # set timestamp as index
    d = d.set_index(['trip_start'])

    # add month
    d['month'] = d.index.month
    # add hour
    d['hour'] = d.index.hour
    # add number (from 0 to 6) representing the day of the week 
    d['day_of_week'] = d.index.dayofweek
    # add day of the month (from 1 to max 31)
    d['day_of_month'] = d.index.day

    # add boolean variable stating whether it's weekend or not
    is_weekend = []
    for el in d.index.weekday:
        if el >= 5:
            # 5-6 -> weekend
            is_weekend.append(True)
        else:
            # 0-4 -> weekday
            is_weekend.append(False)
    d['is_weekend'] = is_weekend

    # add holiday variable:
    # True if the date represent a holiday in Italy, False otherwise
    d['holiday'] = [el in holidays.Italy() for el in d.index]

    return d.sort_values(by='trip_start')



def build_df_sparse(precision=5, freq='H'):
    '''
    Build the dataframe used to train the ML models: sparse as it contains all the times by cell where trips count were zero.
    It does not use POIs nearby as the previous build_df function, as it would become problematic when considering 0 counts.
    NOTE: adding zero counts for each cell imply having len(all_times) * df.gh.nunique() records in the dataframe,
    a number that increases when increasing the number of cells (identified by the gh).
    Thus, to make the computation feasable the number of cells were limited to 6, using a precision = 5.
    This means that the city was divided into six very large cells, which limits the issue of sparse data and make the 
    computation feasable, but does not allow to be precise about the spatial aspect, providing a large surface for the relocation.
    '''
    # build dataframe
    df = build_df(precision)
    # select needed column
    df = df[['totalSnow_cm', 'windspeedKmph', 'tempC', 'visibility', 'precipMM',
        'trips_count', 'gh_centroid_lon', 'gh_centroid_lat', 'month', 'hour', 'day_of_week', 'day_of_month',
        'is_weekend', 'holiday']]
    # collect all timestamps from the beginning of data collection with one hour interval
    all_times = pd.DataFrame({'trip_start': pd.date_range(df.index.min(), df.index.max(), freq='h')})

    # add coordinate cell centroid as a single column
    gh_centroid = []
    for i,j in zip(df.gh_centroid_lat, df.gh_centroid_lon):
        gh_centroid.append((i,j))
    df['gh'] = gh_centroid

    # build dataframe with rows for all times and all cells, even when trips_count = 0 
    final_df = pd.DataFrame({'trip_start':[]})
    for g in df.gh.unique():
        gh_df = df[df.gh==g][['trips_count']]
        gh_all_times = pd.merge(all_times, gh_df, left_on='trip_start', right_on=gh_df.index, how='left').fillna(0)
        # AGGREGATE ON "freq" HOURS LEVEL -> already done if freq="H", necessary only if hours further aggregated
        gh_all_times = gh_all_times.groupby([pd.Grouper(key='trip_start', freq=freq)]).sum() # line to aggregate further timestamps based on inputed freq
        gh_all_times['gh'] = [g]*len(gh_all_times)
        final_df = final_df.append(gh_all_times)
        # final_df = pd.concat([gh_all_times, final_df], ignore_index=True)

    # after the Grouper, trip_start appears both as index and as column
    final_df = final_df.drop(['trip_start'], axis=1) 
    final_df = final_df.reset_index()
    final_df.rename(columns={'index':'trip_start'}, inplace=True)

    final_df = final_df.sort_values(by='trip_start').drop_duplicates().reset_index(drop=True)

    # add weather information
    final_df['trip_start'] = pd.to_datetime(final_df['trip_start'])
    final_df['trip_start'] = final_df['trip_start'].dt.tz_localize(None)
    weather = pd.read_csv('C:/Users/luisa/Desktop/thesis_project/data/weather_info.csv')
    weather = weather[['date_time', 'totalSnow_cm', 'precipMM', 'tempC', 'visibility', 'windspeedKmph']]
    weather['date_time'] = pd.to_datetime(weather['date_time'], format="%Y-%m-%d %H:%M:%S")
    # aggregate weather on "freq" hours basis, computing average of weather values in those hours
    # start aggregation from start date on final_df --> final_df.at[0, 'trip_start']
    weather = weather[weather.date_time>=final_df.at[0, 'trip_start']].reset_index(drop=True)
    weather = weather.groupby([pd.Grouper(key='date_time', freq=freq)]).mean().reset_index()
    # add weather info to final_df
    final_df = pd.merge(final_df, weather, left_on='trip_start', right_on='date_time', how='left')
    final_df = final_df.drop(['date_time'], axis=1)

    # add month, hour, day of the week and day of the month
    final_df = final_df.set_index(['trip_start'])
    final_df['month'] = final_df.index.month
    final_df['hour'] = final_df.index.hour
    final_df['day_of_week'] = final_df.index.dayofweek
    final_df['day_of_month'] = final_df.index.day

    # add boolean variable stating whether it's weekend or not
    is_weekend = []
    for el in final_df.index.weekday:
        if el >= 5:
            # 5-6 -> weekend
            is_weekend.append(True)
        else:
            # 0-4 -> weekday
            is_weekend.append(False)
    final_df['is_weekend'] = is_weekend

    # add boolean variable stating whether it's holiday or not
    final_df['holiday'] = [el in holidays.Italy() for el in final_df.index]

    return final_df



## LSTM DATA PREPARATION FUNCTION
def create_sequences(X, y, time_steps):
    '''
    This function is used to create sequences on which to train the model, 
    as LSTM takes data shaped as [batch_size, time_steps, number_of_features].
    Each sequence is going to contain:
    - *batch_size* number of samples in each batch during training and testing;
    - *time_steps* data points (sequence length);
    - *number_of_features* dimensions used to represent the data in each time_steps.
    '''
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


## PLOT TRAINING VS VALIDATION ERROR
def training_curve(fitted_model):
    '''
    Plot training vs validation error.
    '''
    plt.plot(fitted_model.history['loss'], label='train')
    plt.plot(fitted_model.history['val_loss'], label='test')
    plt.legend()
    plt.show()