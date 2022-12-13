import pandas as pd
import holidays
import geohash as pgh


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


def compute_geohash_block(df, precision = 5):
    '''
    Add to each row the code of the cell of origin of the trip.
    '''
    m = dict()

    gh = []

    for pt in range(len(df)):
        addToGeohash(m, df.at[pt, 'trip_origin_latitude'], df.at[pt, 'trip_origin_longitude'], precision) 
        gh.append(getGeohash(m, df.at[pt, 'trip_origin_latitude'], df.at[pt, 'trip_origin_longitude'], precision))

    df['cell_code'] = gh

    return df 


def create_dataframe(aggr_freq = '8H', precision = 5, aggr_function = 'median'):
    '''
    Create final dataframe aggregating temporal and spatial dimension based on the passed frequency and precision.
    Finally augment the dataframe with all the other relevant features.

    Requires:
     - aggr_freq: temporal interval to be used to aggregate values
     - precision: spatial precision to be used to divide the city into cells/blocks (the higher the precision, the smaller the cells)
     - aggr_function: the function to be used to aggregate the POIs categories counts per location per time-interval
    '''
    # read data
    df = pd.read_parquet('C:/Users/luisa/Desktop/thesis_project/data/tripsv3_augmented.parquet')
    # truncate trip_start at hour level
    df['trip_start'] = [df.at[i, 'trip_start'].replace(minute=0, second=0) for i in range(len(df))]
    # remove UTZ
    df['trip_start'] = pd.to_datetime(df.trip_start, format="%Y-%m-%d %H:%M:%S")
    df.trip_start = df.trip_start.dt.tz_localize(None)
    # add cell belonging
    df = compute_geohash_block(df = df, precision = precision)
    # function to use to aggregate POIs by location per hour
    aggr_poi_per_h = 'max'
    # compute trips count per location per hour
    df = df.groupby(['trip_start', 'cell_code']).agg(
        trips_count=pd.NamedAgg(column='unique_id', aggfunc='count'),
        leisure=pd.NamedAgg(column='leisure', aggfunc=aggr_poi_per_h),
        public_services=pd.NamedAgg(column='public_services', aggfunc=aggr_poi_per_h), 
        building=pd.NamedAgg(column='building', aggfunc=aggr_poi_per_h), 
        education=pd.NamedAgg(column='education', aggfunc=aggr_poi_per_h), 
        food=pd.NamedAgg(column='food', aggfunc=aggr_poi_per_h), 
        transport=pd.NamedAgg(column='transport', aggfunc=aggr_poi_per_h), 
        finantial=pd.NamedAgg(column='finantial', aggfunc=aggr_poi_per_h), 
        healthcare=pd.NamedAgg(column='healthcare', aggfunc=aggr_poi_per_h), 
        office=pd.NamedAgg(column='office', aggfunc=aggr_poi_per_h), 
        shop=pd.NamedAgg(column='shop', aggfunc=aggr_poi_per_h), 
        tourism=pd.NamedAgg(column='tourism', aggfunc=aggr_poi_per_h), 
        count_bus_multimodality=pd.NamedAgg(column='count_bus_multimodality', aggfunc=aggr_poi_per_h)
    ).reset_index()

    # first timestamp for aggregation
    start_time = df.trip_start.min()
    # collect all timestamps from the beginning of data collection with one hour interval
    all_times = pd.DataFrame({'trip_start': pd.date_range(df.trip_start.min(), df.trip_start.max(), freq=aggr_freq)})

    # build dataframe with rows for all times and all cells, even when trips_count = 0 
    final_df = pd.DataFrame({'trip_start':[]})
    for g in df.cell_code.unique():
        # create temporary dataframe with only data of one block
        gh_df = df[df.cell_code==g]
        # resample: for each cell/block, resample according to aggr_freq, starting from the same timestamp (start_time)
        # TO EACH CELL, ASSOCIATE THE MEDIAN OF THE POIS NEARBY FOR THE TRIPS BELONGING TO THAT CELL IN THAT TIME-INTERVAL
        gh_df = gh_df.resample(aggr_freq, on='trip_start', origin=start_time).agg({'leisure':aggr_function, 'public_services':aggr_function, 'building':aggr_function,
              'education':aggr_function, 'food':aggr_function, 'transport':aggr_function, 'finantial':aggr_function, 'healthcare':aggr_function, 'office':aggr_function,
              'shop':aggr_function, 'tourism':aggr_function, 'count_bus_multimodality':aggr_function, 'trips_count':'sum'})
        gh_all_times = pd.merge(all_times, gh_df, left_on='trip_start', right_on=gh_df.index, how='left')
        # fill NaN of trips count with 0 (when there were no trips)
        gh_all_times.trips_count = gh_all_times.trips_count.fillna(-1)
        # WHERE THERE ARE NaN, PUT THE MEDIAN PER CELL OF THAT COLUMN
        # fill NaN all other columns with the median of the column for that block/cell
        for col in ['leisure', 'public_services', 'building', 'education', 'food', 'transport', 
                    'finantial', 'healthcare', 'office', 'shop', 'tourism', 'count_bus_multimodality']:
            md = gh_all_times[col].dropna().median()
            gh_all_times[col] = gh_all_times[col].fillna(md)
        # add cell code
        gh_all_times['cell_code'] = g
        # add cell/block to final df
        final_df = final_df.append(gh_all_times)

    final_df = final_df.reset_index(drop=True)
    final_df['trip_start'] = pd.to_datetime(final_df.trip_start, format="%Y-%m-%d %H:%M:%S")


    # weather related variables 
    weather = pd.read_csv('C:/Users/luisa/Desktop/thesis_project/data/weather_info.csv')
    weather = weather[['date_time', 'totalSnow_cm', 'windspeedKmph', 'tempC', 'visibility', 'precipMM']]
    weather['date_time'] = pd.to_datetime(weather.date_time, format="%Y-%m-%d %H:%M:%S")
    # resample to have weather averages for the selected time interval (aggr_freq)
    weather = weather.resample(aggr_freq, on='date_time', origin=start_time).agg({'totalSnow_cm':'mean', 'windspeedKmph':'mean', 'tempC':'mean', 'visibility':'median', 'precipMM':'mean'})
    # add weather info
    final_df = final_df.merge(weather, left_on='trip_start', right_on='date_time')
    

    # set timestamp as index
    final_df = final_df.set_index('trip_start')
    # add time features
    # add month
    final_df['month'] = final_df.index.month
    # add hour
    final_df['hour'] = final_df.index.hour
    # add number (from 0 to 6) representing the day of the week 
    final_df['day_of_week'] = final_df.index.dayofweek
    # add day of the month (from 1 to max 31)
    final_df['day_of_month'] = final_df.index.day
    # add boolean variable stating whether it's weekend or not
    final_df['is_weekend'] = [1 if day in [5, 6] else 0 for day in final_df.day_of_week]
    # add holiday (boolean)
    final_df['holiday'] = [final_df.index[i] in holidays.Italy() for i in range(len(final_df))] 
    final_df['holiday'] = final_df.holiday.astype('int').to_numpy()

    return final_df


