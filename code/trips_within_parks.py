import pandas as pd
import geopandas as gpd
import numpy as np
import alphashape
from shapely.geometry import Point, Polygon, LineString
from shapely.geometry.multipolygon import MultiPolygon
from collections import Counter 
from pyproj import Geod
import plotly.express as px
import numpy as np
import pyproj
from shapely.ops import transform
import warnings
warnings.filterwarnings("ignore")


'''
Funzioni per investigare il movimento dei monopattini elettrici nei parchi a circolazione interdetta
e verificare/visualizzare le velocità adottate all'interno di queste aree in caso di violazione del divieto.
'''



def get_parks(path_shp="../data/static/parchi_con_circolazione_interdetta.shp", alpha_parameter=0.01):
    '''
    Input:
        - path_shp: path of the shape file for the parks
        - alpha_parameter: alpha for the alpha shape

    Get the data about parks geometries.
    '''
    # PARCHI A CIRCOLAZIONE INTERDETTA
    area = gpd.read_file(path_shp) 

    points = []
    for idx, row in area.iterrows():
        points= points + (list(area.geometry[idx].exterior.coords))
    points = np.array(points)
    alpha_shape = alphashape.alphashape(points, alpha_parameter)

    # change CRS
    project = pyproj.Transformer.from_proj(
        pyproj.Proj(init='epsg:25832'), # source
        pyproj.Proj(init='epsg:4326')) # destination

    # apply projection 
    parchi = transform(project.transform, alpha_shape)  

    return parchi



def count_trips_within_parks(parchi: MultiPolygon, path_trips_routes='../data/map_matched_routes.parquet'):
    '''
    Input:
        - parchi: multipolygon, in which each polygon represent a park
        - path_trips_routes: path of the map-matched routes of each e-scooter trip

    Checks whether trips pass or not within parks and return a dictionary with the counts of trips passing through each park.
    '''
    # check if e-scooters trips pass within the parks areas 
    # (by default using map-matched data)
    data = pd.read_parquet(path_trips_routes)

    is_within_gocciadoro=[]
    is_within_sol=[]
    is_within_albere=[]
    for r in range(len(data)):
        l = [list([s[1],s[0]]) for s in data.at[r, 'route']] # long-lat
        if len(l)>=2:
            is_within_gocciadoro.append(LineString(l).intersects(Polygon(parchi[0])))
            is_within_albere.append(LineString(l).intersects(Polygon(parchi[1])))
            is_within_sol.append(LineString(l).intersects(Polygon(parchi[2])))
        else:
            is_within_gocciadoro.append(Point(l[0]).intersects(parchi[0]))
            is_within_albere.append(Point(l[0]).intersects(parchi[1]))
            is_within_sol.append(Point(l[0]).intersects(parchi[2]))

    trips_intersecting = {}
    # pd.Series(is_within_gocciadoro).value_counts()
    trips_intersecting['Gocciadoro'] = Counter(is_within_gocciadoro)
    trips_intersecting['Solzenicy'] = Counter(is_within_sol)
    trips_intersecting['Albere'] = Counter(is_within_albere)
    return trips_intersecting



def inside_park(park: Polygon, path_trips_routes='../data/map_matched_routes.parquet'):
    '''
    Input:
        - park: polygon of the park 
        - path_trips_routes: path of the parquet file containing the map-matched routes of e-scooters trips

    Given a polygon representing a park, return all trips passing through it.
    '''
    data = pd.read_parquet(path_trips_routes)
    data['geometry'] = [[list([s[1],s[0]]) for s in data.at[r, 'route']] for r in range(len(data))]
    
    # check which trips pass through the park
    pass_within = []
    
    for r in range(len(data)):
        if len(data.at[r, 'geometry'])>=2:
            # LINESTRING
            pass_within.append(LineString(data.at[r, 'geometry']).intersects(park))
        else:
            # POINT
            pass_within.append(Point(data.at[r, 'geometry'][0]).intersects(park))
    
    data['pass_within'] = pass_within

    # keep only trips passing through the park
    return data[data.pass_within == True].reset_index(drop=True)



def stops_or_transit(routes_inside, park_geom: Polygon):
    '''
    Input:
        - routes_inside: dataset of trips passing through the park; can be obtained via the inside_park function
        - park_geom: geometry describing the park

    Returns a dictionary specifying, for the selected park, the number of trips passing through it, 
    the ones starting inside the park, the ones having destination within the park, and the one starting and ending in the park.
    '''
    # check if start or end inside or just transit through  
    end_pt=[]
    start_pt=[]
    for i in range(len(routes_inside)):
        end_pt.append(routes_inside.at[i, 'route'][-1])
        start_pt.append(routes_inside.at[i, 'route'][0])
    routes_inside['start_pt'] = start_pt
    routes_inside['end_pt'] = end_pt
    
    start_inside = []
    end_inside = []
    only_transit = []
    both = []
    for r in range(len(routes_inside)):
        start = [routes_inside.at[r, 'start_pt'][1], routes_inside.at[r, 'start_pt'][0]] # long-lat
        end = [routes_inside.at[r, 'end_pt'][1], routes_inside.at[r, 'end_pt'][0]]
        start_inside.append(Point(start).intersects(Polygon(park_geom)))
        end_inside.append(Point(end).intersects(Polygon(park_geom)))
        if Point(start).intersects(Polygon(park_geom)) and Point(end).intersects(Polygon(park_geom)):
            both.append(True)
        if Point(end).intersects(Polygon(park_geom)) == False and Point(start).intersects(Polygon(park_geom)) == False:
            transit_pts = [list([s[1],s[0]]) for s in routes_inside.at[r, 'route']]
            if len(transit_pts) >= 2:
                only_transit.append(LineString(transit_pts).intersects(Polygon(park_geom)))
                
            elif len(transit_pts) == 1:
                only_transit.append(Point(transit_pts[0]).intersects(Polygon(park_geom)))

    # print('According to map-matched data,', len(routes_inside), 'trips pass inside the', park_name, 'area, of which:\n',
    # ' - ', Counter(start_inside)[True],'trips have starting point inside the area\n',  
    # ' - ', Counter(end_inside)[True], 'trips stop inside the area\n', 
    # ' - ', len(both), 'trips start and end inside the area\n'
    # ' - ', Counter(only_transit)[True], 'only transit through the area.\n')

    return {'tot_trips_within': len(routes_inside),
            'starting_within': Counter(start_inside)[True],
            'ending_within': Counter(end_inside)[True],
            'start_AND_end_within': len(both),
            'transit_through_only': Counter(only_transit)[True]}



def compute_speed(park_geom: Polygon, 
                df_path = '../data/trips_pointv3_cleaned.parquet', # path of the escooter dataset
                path_trips_routes='../data/map_matched_routes.parquet'): # path as input for the inside_park function (defined above)
    ''' 
    Input:
        - park_geom: polygon representing a park
        - df_path: path of the parquet file containing the e-scooter dataset
        - path_trips_routes: path of the map-matched trips routes

    For the selected park (defined by the polygon given as input), returns dataset containing only trips intersecating it, 
    adding information about travelled distance and speed.
    '''
    # retrieve start and end timestamp from the original dataframe:
    # read data
    df = pd.read_parquet(df_path).sort_values(by=['point_timestamp'])
    # for each trip, report list of timestamps recorded (sorted)
    df_times = df.groupby('unique_id', as_index=False).aggregate({"point_timestamp": lambda x: x.to_list()})

    # create variable with start and end time of trips
    df_times['start_times'] = [s[0] for s in df_times.point_timestamp]
    df_times['end_times'] = [s[-1] for s in df_times.point_timestamp]
    # remove trips where same timestamps (not moving)
    df_times = df_times[df_times['start_times'] != df_times['end_times']]

    # get data of trips passing within park (previously defined function) # <--- path_trips_routes
    routes_inside = inside_park(park_geom, path_trips_routes)

    # join dataframes, so that only trips passing within the park are kept
    routes_inside = pd.merge(df_times, routes_inside, on='unique_id', how='inner')

    # remove trips having only one datapoint (not moving)
    routes_inside = routes_inside[routes_inside['route'].map(len) > 1].reset_index(drop=True)

    # compute distance (meters)
    geod = Geod(ellps="WGS84")
    routes_inside['distance_meters'] = [geod.geometry_length(LineString([[pt[1], pt[0]] for pt in routes_inside.at[i, 'route']])) for i in range(len(routes_inside))]

    # compute speed in meters/seconds
    routes_inside['speed'] = None
    for i in range(len(routes_inside)):
        time_delta = (routes_inside.at[i, 'end_times'] - routes_inside.at[i, 'start_times']).total_seconds()
        routes_inside.at[i, 'speed'] = routes_inside.at[i, 'distance_meters'] / time_delta # in m/s

    return routes_inside



def plot_distrib_speed(routes_inside, title_plot='Speed within the Park'):
    '''
    Plot speed of e-scooters in the park 
    '''
    fig = px.histogram(routes_inside, x="speed", title=title_plot)
    fig.show()



def parks_speed_boxplots(parchi: MultiPolygon, df_path = '../data/trips_pointv3_cleaned.parquet', # inputs compute_speed function
                        path_trips_routes='../data/map_matched_routes.parquet', # input inside_parks function
                        path_shp="../data/static/parchi_con_circolazione_interdetta.shp", alpha_parameter=0.01, # inputs get_parks function
                        remove_outliers = True, max_speed = 12): # values used to filter out outliers in the boxplots
    ''' 
    Input:
        - parchi: multipolygon representing a polygon for each park
        - df_path: path of the parquet file containing escooters trips (default provided)
        - path_trips_routes: path of the parquet file containing the map-matched routes (default provided)
        - path_shp: path of the shape file containing parks areas (default provided)
        - alpha_parameter: alpha parameters to be used in the alpha shape when creating parks geometries (default: 0.01)
        - remove_outliers: boolean, if True outliers are removed, according to the value specified by max_speed parameter (default True)
        - max_speed: max speed to be allowed in the data, used to filter out outliers, i.e., trips having anomalously high speeds (default: 12)

    Returns boxplots, one for each park, visualizing the distribution of the speed.
    NOTE: Outliers are removed if remove_outliers is True (default)
    '''
    parchi = get_parks(path_shp, alpha_parameter)

    albere_df = compute_speed(parchi[1])
    albere_df['park'] = 'Albere'

    gocciadoro_df = compute_speed(parchi[0])
    gocciadoro_df['park'] = 'Gocciadoro'

    sol_df = compute_speed(parchi[2])
    sol_df['park'] = 'Solzenicy'

    # dataframe with all parks
    parks_df = pd.concat([albere_df, gocciadoro_df, sol_df])[['unique_id', 'speed', 'park']]

    # REMOVE OUTLIERS:
    if remove_outliers == True:
        parks_df = parks_df[parks_df.speed <= max_speed]

    fig = px.box(parks_df, x='park', y="speed", color="park",
                notched=True, 
                title="Boxplots of speed by park")
    return fig.show()