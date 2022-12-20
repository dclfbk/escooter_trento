# Content: data cleaning
# - remove wrong coordinates (latitude or longitude less then or equal to zero).
# - fix point_trip_id: some ID repeat themselves after some time: must modify them so that each trip ID is unique.
# - filter out data outiside the area of interest (where scooters are allowed) using alphashape.
# - remove anomalous trips (e.g., few trips last hours, which is very likely a mistake, maybe resulting from trip concatenation - also
# considering the fact that the average e-scooters trip last around 9 minutes;
# other trips last for just few seconds and/or do not travel significative distance from the origin of the trip, thus they are more likely
# to be done for fun, and not for travelling purposes).

import pandas as pd
import geopandas as gpd
import numpy as np
import alphashape
from shapely.geometry import Point
import fiona
import pyproj
from shapely.ops import transform
import warnings
warnings.filterwarnings("ignore")

# read dataset
data = pd.read_parquet('C:/Users/luisa/Desktop/thesis_project/data/tripsv3.parquet')

# change columns type
data = data.astype({'trip_origin_longitude':'float64', 'trip_origin_latitude':'float64', 'trip_destination_longitude':'float64', 'trip_destination_latitude':'float64'})

# drop columns filled with redundant and not needed information, or with None only
data = data.drop(['trip_properties', 'trip_user_id', 'trip_operator_id', 'trip_start_epoch', 'trip_end_epoch', 'trip_origin_time', 'trip_destination_time', 'trip_points_num', 'trip_points_numall', 'trip_mode', 'trip_accuracy'], axis=1) 

# drop null or negative cordinates values, as they are a result of GPS mistakes, given the area under study
data = data.loc[(data.trip_origin_latitude > 0) & (data.trip_origin_longitude > 0) & (data.trip_destination_latitude > 0) & (data.trip_destination_longitude > 0)].reset_index(drop=True)

# introduce unique trip identifiers:
# create a unique id as a concatenation of trip_id and day/hour of trips beginning
id=[]
for r in range(len(data)):
  id.append(str(data.trip_id[r])+'_'+str(data.trip_start[r].date())+'_h'+str(data.trip_start[r].hour))
data['unique_id'] = id
data = data.drop(['trip_id'], axis=1) 

# filter out trips that starting from outside the area where e-scooters are allowed (area under study) using alphashape
data = gpd.GeoDataFrame(data, crs='EPSG:4326', geometry=gpd.points_from_xy(data.trip_origin_longitude, data.trip_origin_latitude))
# load data about area where e-scooters are allowed
area = gpd.read_file("C:/Users/luisa/Desktop/thesis_project/data/static/ambito_esercizio_monopattini.shp") 

# alphashape
points = []
for idx, row in area.iterrows():
  points= points + (list(area.geometry[idx].exterior.coords))
points = np.array(points)
alpha_shape = alphashape.alphashape(points, 0.0001)

# get "area" reference system using fiona
file = fiona.open("C:/Users/luisa/Desktop/thesis_project/data/static/ambito_esercizio_monopattini.shp")
spatialRef = file.crs # spatialRef['init'] --> area reference system

# change CRS
project = pyproj.Transformer.from_proj(
    pyproj.Proj(init=spatialRef['init']), # source coordinate system
    pyproj.Proj(init='epsg:4326')) # destination coordinate system

g2 = transform(project.transform, alpha_shape)  # apply projection to alpha shape polygon

# check if trip points are within the alphashape and keep only those who are
# repeat for both origin and destination points
data["is_within_origin"] = data.apply(lambda row: g2.intersects(Point(row["trip_origin_longitude"], row["trip_origin_latitude"])), axis = 1)
data["is_within_destination"] = data.apply(lambda row: g2.intersects(Point(row["trip_destination_longitude"], row["trip_destination_latitude"])), axis = 1)
data = data.loc[(data.is_within_origin == True) & (data.is_within_destination == True)].reset_index(drop=True)
data = data.drop(['is_within_origin', 'is_within_destination', 'geometry'], axis=1)

# remove anomalous trips / reduce noise in the data
# compute trip duration 
time_duration = []
for r in range(len(data)):
    time_duration.append(data.trip_end[r] - data.trip_start[r])
data['time_trip'] = time_duration
# remove trips lasting more than one hour or less than one minute
data = data.loc[(data.time_trip >= pd.Timedelta('0 days 00:01:00')) & (data.time_trip < pd.Timedelta('0 days 01:00:00'))].reset_index(drop=True)
data = data.drop(['time_trip'], axis=1)

'''
# checking travelled distance
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize =(10, 5), tight_layout = True)

ax.hist(data.trip_length, bins = 100)
ax.set_ylim(ymin=0)
ax.set_xlim(xmin=0)
plt.show()
'''

# remove trips travelling less than 10 meters distance and more than 2500
data = data.loc[(data.trip_length > 10) & (data.trip_length < 2500)].reset_index(drop=True)

# saved cleaned dataset
data.to_parquet('C:/Users/luisa/Desktop/thesis_project/data/tripsv3_cleaned.parquet')