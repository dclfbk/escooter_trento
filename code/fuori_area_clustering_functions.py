import pandas as pd
import geopandas as gpd
import numpy as np
import alphashape
from shapely.geometry import Point, Polygon
import pyproj
from shapely.ops import transform
import similaritymeasures
from sklearn.cluster import DBSCAN 
import folium
from branca.element import Figure
from folium.plugins import BeautifyIcon
import random
import warnings
warnings.filterwarnings('ignore')

'''
Obiettivo:
Individuare i punti fuori area e verificare se sono “recidivi”, ovvero se si tratta di tragitti comuni o simili che ripetutamente escono dall'area limite.

Per calcolare la similitudine tra le traiettorie aventi punti fuori area è stata utilizzata la distanza di Fréchet, 
e viene poi applicato DBSCAN per costruire clusters di traiettorie simili.
'''


def get_area_limite(area_limite_path="../data/static/ambito_esercizio_monopattini.shp",
                    alpha_shape_value=0.0001):
    '''
    Input:
        - area_limite_path: path of the shapefile containing the area of movements for e-scooters
        - alpha_shape_value: value of the alpha for the alphashape used to defined the area where movement is allowed

    Applies alpha shape and return area in which e-scooters movement is allowed.
    '''
    area = gpd.read_file(area_limite_path)
    points = []
    for idx, row in area.iterrows():
        points = points + (list(area.geometry[idx].exterior.coords))
    points = np.array(points)
    alpha_shape = alphashape.alphashape(points, alpha_shape_value)

    # change CRS
    project = pyproj.Transformer.from_proj(
        pyproj.Proj(init='epsg:25832'), # source coordinate system
        pyproj.Proj(init='epsg:4326')) # destination coordinate system
    shape_area = transform(project.transform, alpha_shape) 
    
    return shape_area



def get_trips_outside_area(escooter_data_path='../data/trips_pointv3.parquet',
                            area_limite_path="../data/static/ambito_esercizio_monopattini.shp",
                            alpha_shape_value=0.0001):
    '''
    Input:
        - escooter_data_path: path of the uncleaned e-scooters data (parquet file)
        - area_limite_path: path of the shapefile containing the area of movements for e-scooters
        - alpha_shape_value: value of the alpha for the alphashape used to defined the area where movement is allowed

    Returns dataframe containing only trips having at least one point of their trajectory outside the defined area.
    '''

    # GET E-SCOOTERS DATA:
    data = pd.read_parquet(escooter_data_path)
    data.point_latitude = data.point_latitude.astype('float')
    data.point_longitude = data.point_longitude.astype('float')
    data = data[(data.point_latitude >= 0)&(data.point_longitude >= 0)].reset_index(drop=True)
    # create UNIQUE ID as concatenation of point_trip_id and date
    data['unique_id'] = data.point_trip_id + '_' + data.point_timestamp.dt.date.astype('str')
    # convert to geodataframe
    data = gpd.GeoDataFrame(data, crs='EPSG:4326', geometry=gpd.points_from_xy(data.point_longitude, data.point_latitude))
    
    # GET AREA LIMITE:
    shape_area = get_area_limite(area_limite_path, alpha_shape_value)

    # FILTER ONLY TRIPS OUTSIDE AREA:
    # keep only records where unique_id is recorded outside the area in at least one point
    # check whether points are outside or inside the area
    data['point_inside_area'] = data.apply(lambda row: shape_area.intersects(Point(row["point_longitude"], row["point_latitude"])), axis=1)
    # get unique_id of trips going outside area
    trips_outside_area = data[data.point_inside_area == False].unique_id.to_list()
    # keep only data of the trips going outside area
    df_outside_area = data[data.unique_id.isin(trips_outside_area)].sort_values(by=['unique_id', 'point_sequence']).reset_index(drop=True)

    return df_outside_area



def trips_trajectories(df_outside_area):
    '''
    Input:
        - df_outside_area: dataframe containing only the data of trips going outside the area; must have fields unique_id, point_longitude, and point_latitude
    
    Returns a dataframe containing the unique identifier of the trips and their trajectory as a list of lists.
    '''

    traj_df = df_outside_area.groupby(['unique_id'], as_index=False).aggregate({"point_longitude": lambda x: x.to_list(), "point_latitude": lambda x: x.to_list()})
    traj_df["route"] = [ [[k,v] for k,v in zip(i,j)] for i,j in zip(traj_df["point_longitude"], traj_df["point_latitude"])]
    traj_df = traj_df[['unique_id', 'route']]
    # filter data so that each trip has AT LEAST 2 points in their trajectory 
    traj_df = traj_df[traj_df['route'].map(len) >= 2].reset_index(drop=True)
    return traj_df



def compute_distance_matrix(trajectories, method="Frechet"):
    '''
    Input:
        - trajectories: list of lists, each representing a point in the trip trajectory
        - method: method used to compute the similarity between two trajectories 
            (default: "Frechet" distance; alternative is "Area").
            The Frechet distance is defined as a measure of similarity between two curves, 
            taking into account the location and ordering of the points along the curves, 
            thus it is particularly suited for the computation of similarities among trajectories.

    Returns matrix of trajectories similarities.
    The methods used to compute distances are implemented in the similaritymeasures library
    [doc: https://jekel.me/similarity_measures/similaritymeasures.html#header-functions]
    '''
    n = len(trajectories)
    dist_m = np.zeros((n, n))
    for i in range(n - 1):
        p = trajectories[i]
        for j in range(i + 1, n):
            q = trajectories[j]
            if method == 'Frechet':
                dist_m[i, j] = similaritymeasures.frechet_dist(np.array(p), np.array(q))
            elif method == 'Area':
                dist_m[i, j] = similaritymeasures.area_between_two_curves(np.array(p), np.array(q))
            dist_m[j, i] = dist_m[i, j]
    return dist_m



def distance_matrix_trajectory_df(traj_df, method='Frechet'):
    '''
    Input: 
        - traj_df: dataset containing unique_id field and route field, with the trajectory of each trip as a list of lists
    Applies the compute_distance_matrix and returns the matrix of trajectories similaritity.
    '''
    return pd.DataFrame(compute_distance_matrix(traj_df.route.to_list(), method=method), columns=traj_df.unique_id.to_list(), index=traj_df.unique_id.to_list())



def clustering_trajectories(similarity_matrix, eps=0.01):
    '''
    Input:
        - similarity_matrix: matrix of similarity between trajectories.
        - eps: maximum distance between two samples for one to be considered as in the neighborhood of the other; 
                manipulate this parameter to change how much similar trajectories must be in order to belong to the same cluster.
                default value = 0.01 - inferred after observing distribution of trajectories distances.

    Returns clustering labels according to the DBSCAN algorithm as an array.
    '''
    cl = DBSCAN(eps=eps, min_samples=1, metric='precomputed')
    cl.fit(similarity_matrix)
    return cl.labels_



def viz_trajectories_clusters(traj_df, sim_matrix, eps=0.01,
                                plot_area = True,
                                area_limite_path="../data/static/ambito_esercizio_monopattini.shp",
                                alpha_shape_value=0.0001):
    '''
    Input:
        - traj_df: dataframe containing unique identifiers of trips and their trajectory as a list of lists;
            can be produced using the trips_trajectories function. 
        - sim_matrix: matrix of similarities among trajectories; can be computes via the compute_distance_matrix function.
        - eps: in DBSCAN, maximum distance between two samples for one to be considered as in the neighborhood of the other (default = 0.01).
        - plot_area: boolean, if True (default), plot the shape of the area in which e-scooters movement is allowed over the map.
        - area_limite_path: path of the shapefile containing the area of movements for e-scooters.
        - alpha_shape_value: value of the alpha for the alphashape used to defined the area where movement is allowed (default = 0.0001).

    Returns a folium map with trips grouped by cluster, i.e, similar trajectories are displayed as belonging to the same feature group.
    '''
    # clustering (DBSCAN)
    traj_df['cluster'] = clustering_trajectories(sim_matrix, eps=eps)

    # MAP
    fig = Figure(height=550,width=750)
    m = folium.Map(location=[46.066, 11.133], tiles='cartodbpositron', zoom_start=14)

    if plot_area == True:
        shape_area = get_area_limite(area_limite_path, alpha_shape_value)
        folium.GeoJson(shape_area).add_to(m)
        
    fig.add_child(m)

    for i in traj_df.cluster.unique():
        feat = folium.FeatureGroup('Cluster n.'+str(i), show=False) # group according to BDSCAN clustering
        
        traj_similar_to_ith = traj_df[traj_df.cluster == i].reset_index(drop=True)
        
        color = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
        
        for trajectory in traj_similar_to_ith.route.to_list(): 
            single_traj = [el[::-1] for el in trajectory] # reverse coordinates

            for ith_pt in range(len(single_traj)):
                # points numbered based on order (to get trajectory direction)
                icon_circle = BeautifyIcon(icon_shape='circle-dot', border_color='green', border_width=5)
                folium.Marker(single_traj[ith_pt], tooltip=ith_pt+1, icon=icon_circle).add_to(feat)
            
            folium.vector_layers.PolyLine(single_traj,
                                            popup='column n.'+str(i)+'; trips similar to ID: '+str(sim_matrix.columns[i]),
                                            color=color).add_to(feat)

            # plugins.AntPath(single_traj,popup='column n.'+str(i)+'; trips similar to ID: '+str(sim_matrix.columns[i]), color=color).add_to(feat)
            feat.add_to(m)            

    m.add_child(folium.LayerControl())

    return m