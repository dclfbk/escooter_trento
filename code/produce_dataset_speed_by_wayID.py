from shapely.geometry.linestring import LineString
from pyproj import Geod
import pandas as pd
import json
import requests
import json
import requests


# read map-matched df & original one
df = pd.read_parquet('C:/Users/luisa/Desktop/escooter_trento/data/map_matched_edges_ids.parquet')
df = df.reset_index(drop=True)
orig_df= pd.read_parquet('C:/Users/luisa/Desktop/escooter_trento/data/trips_pointv3_cleaned.parquet')
new_df = orig_df.groupby('unique_id', as_index=False).aggregate({"point_timestamp":lambda x: x.to_list()})

# collect start/end time trips
start_time = {}
end_time = {}
for i in range(len(new_df)):
    start_time[new_df.at[i, 'unique_id']] = min(new_df.at[i, 'point_timestamp'])
    end_time[new_df.at[i, 'unique_id']] = max(new_df.at[i, 'point_timestamp'])

# add start and end time of each trip to map-matched df
tmstmp_start = []
tmstmp_end = []
for row in range(len(df)):
    tmstmp_start.append(start_time[df.at[row, 'unique_id']])
    tmstmp_end.append(end_time[df.at[row, 'unique_id']])
df['start_time'] = tmstmp_start
df['end_time'] = tmstmp_end

# add column with hour
df['start_hour'] = [df.at[i, 'start_time'].hour for i in range(len(df))]


# TRIED TWO WAYS TO GET DISTANCES:

# 1.
# Compute speeds manually, using pyproj.Geod to compute distance
# compute distance using map-matched coordinates (more precise computation)
geod = Geod(ellps="WGS84")
distance_mm = []
for i in range(len(df)):
    distance_mm.append(geod.geometry_length(LineString([[pt[1], pt[0]] for pt in df.at[i, 'route']])))
df['distance'] = distance_mm

# compute speed in meters/seconds
speed = []
for i in range(len(df)):
    time_delta = (df.at[i, 'end_time'] - df.at[i, 'start_time']).total_seconds()
    if time_delta == 0:
        speed.append(0)
    else:
        speed.append(df.at[i, 'distance'] / time_delta) # in m/s
df['speed'] = speed


# 2.
# Valhalla request to get speeds
# get id of street segments, the speed with which they have been travelled, and the street/area name
edge_ids = []
speeds_for_edge = []
street_names = []
ids = df.unique_id.to_list()
for id in range(len(ids)):
    df_temp = df[df.unique_id==ids[id]]
    df_points = pd.DataFrame({'lon':[el[1] for i in df_temp.route for el in i], 'lat':[el[0] for i in df_temp.route for el in i]})

    # request
    meili_coordinates = df_points.to_json(orient='records')
    meili_head = '{"shape":'
    meili_tail = ""","search_radius": 300, "shape_match":"edge_walk", "costing":"bicycle", "format":"osrm"}""" # use 'shape_match' : 'edge_walk' for already map-matched or nearly mm routes
    meili_request_body = meili_head + meili_coordinates + meili_tail
    url = "http://localhost:8002/trace_attributes" # use /trace_attributes 
    headers = {'Content-type': 'application/json'}
    data = str(meili_request_body)

    r = requests.post(url, data=data, headers=headers)
    
    response_text = json.loads(r.text)

    if r.status_code == 200:
        l=[]
        s=[]
        n=[]
        for i in range(len(response_text['edges'])):
            l.append(response_text['edges'][i]['way_id']) # way ID
            s.append(response_text['edges'][i]['speed']) # speed it has been travelled to, in km/h
            if 'names' in response_text['edges'][i].keys():
                n.append(response_text['edges'][i]['names']) # street name if present
            else:
                n.append('')
        edge_ids.append(l)
        speeds_for_edge.append(s)
        street_names.append(n)
    else:
        edge_ids.append([])  
        speeds_for_edge.append([])
        street_names.append([])


df['edges_id'] = edge_ids
df['speeds_for_edge'] = speeds_for_edge
df['street_names'] = street_names

for i in range(len(df)):
    x=[]
    for lst_name in df.at[i, 'street_names']:
        if type(lst_name)==list:
            x.append(lst_name[0])
        else:
            x.append(lst_name)
    df.at[i, 'street_names'] = ' - '.join((str(n) for n in x))



df = df[df['edges_id'].map(lambda d: len(d)) > 0] # remove rows where edge_ids is empty
df.at[len(df)-1, 'edges_id'] = edge_ids[-1][:12]
df = df.drop(['edge_ids'], axis=1)

df.to_parquet('C:/Users/luisa/Desktop/escooter_trento/data/mm_wayID_speed_name.parquet')