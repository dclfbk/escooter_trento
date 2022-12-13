import pandas as pd
import pyrosm
import requests
import pyproj as proj
import shapely
import geopandas as gpd
import numpy as np
from scipy.spatial import KDTree
import gtfs_kit as gk 
from shapely.geometry.linestring import Point
from datetime import timedelta
import holidays
import warnings
warnings.filterwarnings("ignore")

'''
This script adds the count of POIs types within a 15 meters distance to the trips origin, 
to account for the influence the functionality of origin point has on the number of trips generated from there.

POIs types are grouped into 11 categories: leisure, public_services, building, education, food, transport, finantial, healthcare, office, shop, and tourism.
The data are augmented with the counts of each of these categories within a 15 meters buffer from the origin of the trip.

In the second part of the script, we also augment the data with weather-related information, previously collected, namely 
cm of snow, temperature (Celsius), perceived degrees (Celsius), wind speed (km/h), visibility (0-10), and level of precipitation in mm.

Finally, check the number of buses stopping nearby the e-scooter trip origin, as it may be useful to predict starting points of trips, given e-scooters
are often considered part of a multimodal means of transport and/or a last-mile means. Use the data of the Province of Trento GTFS for this purpose.
If a bus stopped near the trip origin within 10 minutes previous to the trip start, then add it to the count of buses. 
In this case, the rider just arrived with the bus may use an e-scooter, if available, to reach their final destination, or a stop in their way.

In order to account for some special events that may occur and influence e-scooters demand, add a boolean variable stating whether it is a holiday in Italy or not.
'''

# Download POI data: request data for POIs in Trento in PBF format
trento_download_pbf_url = "https://osmit-estratti.wmcloud.org/dati/poly/comuni/pbf/022205_Trento_poly.osm.pbf"
r = requests.get(trento_download_pbf_url, allow_redirects=True)
open('C:/Users/luisa/Desktop/thesis_project/data/trento_poi.pbf', 'wb').write(r.content)

# use pyrosm to read these data
osm = pyrosm.OSM("C:/Users/luisa/Desktop/thesis_project/data/trento_poi.pbf")

# get POIs in a dataframe, filtering the needed categories
custom_filter = {'building':['residential', 'commercial', 'government', 'public'], # buildings
'amenity': ['place_of_worship', # buildings
'college', 'kidergarden', 'school', 'university', 'driving_school', 'language_school', 'library', 'toy_library', 'music_school', # education
'bar', 'cafe', 'restaurant', 'biergarten', 'fast_food', 'food_court', 'ice_cream', 'pub', # Sustenance
'taxi', 'parking', 'parking_space', 'motorcycle_parking', 'car_sharing', 'car_rental', 'bus_station', 'bicycle_parking', # transport
'atm', 'bank', 'bureau_de_change', # finantial
'baby_hatch', 'clinic', 'dentist', 'doctors', 'hospital', 'nursing_home', 'pharmacy', 'social_facility', 'veterinary', # healthcare
'arts_centre', 'casino', 'cinema', 'community_centre', 'conference_centre', 'event_centre', 'events_venue', 'fountain', 'gambling', 'nightclub', 'planetarium', 'public_bookcase', 'social_centre', 'studio', 'theatre', 'internet_cafe', # entertainment
'courthouse', 'fire_station', 'police', 'post_box', 'post_depot', 'post_office', 'prison', 'ranger_station', 'townhall'], # public_services 
'leisure': True, 
'office': True, # yes means generic office type
'shop': True, # yes means unspecified shop type
'tourism': True # yes means tourist point of interest, described by some other tag
} 
pois = osm.get_pois(custom_filter=custom_filter)
pois = pois[['lon', 'lat', 'geometry', 'name', 'wikipedia', 'building', 'amenity', 'office', 'leisure', 'shop', 'tourism']]
# add wikipedia name to name column
for i in range(len(pois)):
    if pd.notna(pois.at[i, 'wikipedia']): 
        if pd.notna(pois.at[i, 'name']):
            pois.at[i, 'name'] = pois.at[i, 'wikipedia'].split(':')[1]
        else:
            pois.at[i, 'name'] = pois.at[i, 'name'] + ', ' + pois.at[i, 'wikipedia'].split(':')[1]
pois = pois.drop(['wikipedia'], axis=1)
# replace "yes" values with the unspecified category
for i in range(len(pois)):
    if pois.at[i, 'office'] == 'yes':
        pois.at[i, 'office'] = 'generic office place'
    if pois.at[i, 'shop'] == 'yes':
        pois.at[i, 'shop'] = 'unspecified shop'
    if pois.at[i, 'tourism'] == 'yes':
        pois.at[i, 'tourism'] = 'tourist POI'
    if pois.at[i, 'building'] == 'yes':
        pois.at[i, 'building'] = 'generic building'


# collect all the tags each POI has
lst_cols = list(pois.columns[4:])
all_categories = []
for i in range(len(pois)):
    detailed_category = []
    for cat in lst_cols:
        if pd.notna(pois.at[i, cat]) and pois.at[i, cat] not in detailed_category:
            detailed_category.append(pois.at[i, cat])
    all_categories.append(detailed_category)
pois['tags'] = all_categories


primary_categories = {
'leisure': ['pitch', 'playground', 'sports_centre', 'fitness_centre',
       'club', 'dance', 'tanning_salon', 'picnic_table', 'social_club',
       'adult_gaming_centre', 'park', 'slipway', 'fitness_station',
       'nature_reserve', 'track', 'garden', 'arts_centre', 'casino', 'cinema', 'community_centre', 'conference_centre', 'event_centre', 
       'events_venue', 'fountain', 'gambling', 'nightclub', 'planetarium', 'public_bookcase', 'social_centre', 'studio', 'theatre', 'internet_cafe'], 
'public_services': ['courthouse', 'fire_station', 'police', 'post_box', 'post_depot', 'post_office', 'prison', 'ranger_station', 'townhall'], 
'building':['residential', 'commercial', 'public', 'place_of_worship'],
'education': ['college', 'kidergarden', 'school', 'university', 'driving_school', 'language_school', 'library', 'toy_library', 'music_school'],
'food': ['bar', 'cafe', 'coffee', 'restaurant', 'biergarten', 'fast_food', 'food_court', 'ice_cream', 'pub', 'supermarket','bakery', 'pastry', 'kiosk', 'greengrocer','tea', 'pasta', 'seafood', 'butcher'],
'transport': ['taxi', 'parking', 'parking_space', 'motorcycle_parking', 'car_sharing', 'car_rental', 'bus_station', 'bicycle_parking'],
'finantial': ['atm', 'bank', 'bureau_de_change'],
'healthcare': ['baby_hatch', 'clinic', 'dentist', 'doctors', 'hospital', 'nursing_home', 'pharmacy', 'social_facility', 'veterinary', 'optician'],
'office': ['government', 'accountant', 'estate_agent',
       'political_party', 'ngo', 'administrative', 'insurance',
       'employment_agency', 'telecommunication',
       'educational_institution', 'company', 'notary', 'architect',
       'association', 'courier', 'energy_supplier', 'therapist',
       'newspaper', 'union', 'graphic_design', 'it', 'lawyer',
       'construction_company', 'generic office place',
       'advertising_agency', 'IT', 'consulenza_brevettuale', 'consulting',
       'foundation', 'research'],
'shop': ['newsagent', 'bag', 'clothes', 'jewelry','convenience', 'car_repair', 'electronics', 'computer',
       'hairdresser', 'florist', 'herbalist', 'tyres', 'laundry', 'books', 'shoes',
       'cosmetics', 'chemist', 'beauty', 'dry_cleaning',
       'furniture', 'sports', 'tobacco', 'boutique', 'curtain', 'deli',
       'outdoor', 'paint', 'hearing_aids', 'stationery', 'bicycle',
       'hardware', 'pet', 'travel_agency', 'perfumery', 'anime',
       'weapons', 'antiques', 'money_lender', 'trade', 'tailor',
       'copyshop', 'mobile_phone', 'interior_decoration', 'toys',
       'doityourself', 'craft', 'car', 'gift', 'beverages',
       'fashion_accessories', 'houseware', 'e-cigarette', 'hifi',
       'art', 'fabric', 'confectionery', 'erotic',
       'department_store', 'watches', 'tattoo', 'clothes;food', 'hunting',
       'kitchen', 'bed', 'motorcycle', 'variety_store', 'second_hand',
       'pawnbroker', 'frame', 'unspecified shop', 'musical_instrument',
       'farm', 'party', 'photo', 'vacant', 'funeral_directors', 'model',
       'carpet', 'camera', 'printer_cartridges', 'video_games',
       'garden_centre', 'car_parts', 'bathroom_furnishing', 'lottery',
       'pet_grooming', 'wine', 'music', 'baby_goods', 'video', 'candles',
       'storage_rental', 'radiotechnics', 'tiles', 'wholesale',
       'appliance', 'gas', 'car;car_repair'],
'tourism': ['museum', 'information', 'viewpoint', 'picnic_site',
       'attraction', 'hotel', 'hostel', 'guest_house', 'artwork',
       'apartment', 'alpine_hut']
}


# assign each POI to its primary category/ies
pc=[]
for lst_tags in pois.tags:
    for each_tag in lst_tags:
        primary_category = []
        for key in primary_categories.keys():
            if each_tag in primary_categories[key]:
                primary_category.append(key)
    pc.append(primary_category)

pois['primary_category'] = pc


# Where there is a polygon or multipolygon in the geometry column, 
# get the coordinates of the exterior boundaries, and convert all cordinates to epsg:32632 (meters)
# setup start and end projection
crs_lonlat = proj.Proj(init='epsg:4326') # start projection
crs_meters = proj.Proj(init='epsg:32632') # meters 32632 italia utm 32n wgs84

c = []
for i in pois.geometry:
    coords = []
    if type(i)==shapely.geometry.polygon.Polygon:
        # cicle on exterior coordinates of polygon:
        for pt in i.exterior.coords[:-1]: # the first and last points are the same, thus do not take the last one
            coords.append(proj.transform(crs_lonlat, crs_meters, pt[0], pt[1])) # change crs (meters)

    elif type(i)==shapely.geometry.multipolygon.MultiPolygon:
        multipoly = i.geoms # list of all polygons in multipolygon
        poly_coords = []
        for poly in multipoly:
            # exterior coordinates of polygon:
            poly_coords.extend(poly.exterior.coords[:-1]) 
        for pt in poly_coords:
            coords.append(proj.transform(crs_lonlat, crs_meters, pt[0], pt[1])) # change crs (meters)
        
    elif type(i)==shapely.geometry.point.Point:
        coords.append(proj.transform(crs_lonlat, crs_meters, i.x, i.y)) # change crs (meters)

    c.append(coords)
        

pois['coords'] = c


# put a single point per row --> useful later for using KDTree indexing and associating index to POI

index_to_drop = [] # indexes of polygons and multipolygons -> each point in their external boundaries will be on a different record in the dataframe
new_rows = [] # containing the records for each point in poly/multipoly

for i in range(len(pois)):

    if len(pois.at[i, 'coords']) > 1:
        index_to_drop.append(i)
        
        for pt in pois.at[i, 'coords']:
            new_row = {}
            new_row['coords'] = pt
            for col in pois.columns[:-1]:
                new_row[col] = pois.at[i, col]

            new_rows.append(new_row)

    else:
        pois.at[i, 'coords'] = pois.at[i, 'coords'][0]


# add the new rows with poly/multipoly divided in points & 
# delete the rows where all points are together in the coords column
pois = pois.append(new_rows).drop(index_to_drop).reset_index(drop=True)

# associate to each trip origin the 10 clostest POIs and then put a threshold on their distance,
# in order to consider only POIs within a maximum of 15 meters
df = pd.read_parquet('C:/Users/luisa/Desktop/thesis_project/data/tripsv3_cleaned.parquet')

# create geodataframe, with ORIGIN points as geometries
df = gpd.GeoDataFrame(df, crs='EPSG:4326', geometry=gpd.points_from_xy(df.trip_origin_longitude, df.trip_origin_latitude))
# convert CRS: to meters --> scipy KDTree works with Euclidean distance
df = df.to_crs(epsg=32632)


# array of point from the pois dataset
pois_x = []
for p in pois.coords.to_list():
    pois_x.append(p)
pois_x = np.array(pois_x)

# array of trips origins
trip_x = []
for pt in df.geometry:
    trip_x.append((pt.x, pt.y))
trip_x = np.array(trip_x)

# fit on POIs points
tree = KDTree(pois_x)

# finds the 10 nearest neighbors among POIs; 
# returns arrays of distances POI-NearestNeighbour and array of indexes of the nearest neighbours
distance_of_nn, index_of_nn = tree.query(trip_x, 10) 

# add a column for each primary category
df = df.reindex(df.columns.tolist() + list(primary_categories.keys()), axis=1) 
df = df.fillna(0)

# add one for each cateogy of POI that appears to be nearby the trip origin
# if it is within a 15 meters distance
i=0
for lst_nn_index in range(len(index_of_nn)):
    distance_lst = distance_of_nn[lst_nn_index]
    for nn in range(len(index_of_nn[lst_nn_index])):
        for categ in pois.at[index_of_nn[lst_nn_index][nn], 'primary_category']:
            if distance_lst[nn] < 15:
                df.at[i, categ] = df.at[i, categ] +1
    i+=1

# drop geometry column
df = df.drop(['geometry'], axis=1)





'''
Augment with weather information, with a hourly precision, adding: 
cm of snow, temperature (Celsius), perceived degrees (Celsius), wind speed (km/h), visibility (0-10), and level of precipitation in mm.
'''

weather = pd.read_csv('C:/Users/luisa/Desktop/thesis_project/data/weather_info.csv')
weather = weather[['date_time', 'totalSnow_cm', 'FeelsLikeC', 'windspeedKmph', 'tempC', 'visibility', 'precipMM']]

df['hour'] = [df.at[i, 'trip_start'].hour for i in range(len(df))]
df['date'] = [df.at[i, 'trip_start'].date() for i in range(len(df))]

weather['date_time'] = pd.to_datetime(weather.date_time, format="%Y-%m-%d %H:%M:%S")
weather['hour'] = [weather.at[i, 'date_time'].hour for i in range(len(weather))]
weather['date'] = [weather.at[i, 'date_time'].date() for i in range(len(weather))]

df = df.merge(weather, left_on=['date', 'hour'], right_on=['date', 'hour'])
df = df.drop(['date_time', 'date', 'hour'], axis=1)




'''
Finally, augment the data with the number of buses stopping nearby the e-scooter trip' origin, as e-scooters are often considered in the context of multimodality
and/or as a last-mile means of transport. Use the data of the Province of Trento GTFS to get bus stops positions and the times and days in which buses stops nearby.
If a bus stopped near the trip origin within 10 minutes previous to the trip start, then add it to the count of buses. 
In this case, the rider just arrived with the bus may use an e-scooter, if available, to reach their final destination, or a stop in their way.
'''
# read the feed with gtfs-kit
path = 'C:/Users/luisa/Desktop/thesis_project/data/google_transit_urbano_tte.zip' # valido dal 11-06-2021 fino al 24-06-2022
feed = (gk.read_feed(path, dist_units='km'))

# collect stops and buses stopping nearby, given time of the day and day of the week
# needed: feed.stops & feed.stop_times + feed.calendar for checking weekdays in which each bus runs
stops_times = feed.stops.merge(feed.stop_times, left_on=['stop_id'], right_on=['stop_id'])[['stop_id', 'stop_lat', 'stop_lon', 'arrival_time', 'trip_id']]
weekdays_service = feed.trips.merge(feed.calendar, left_on=['service_id'], right_on=['service_id'])[['trip_id', 'service_id', 'monday', 'tuesday', 'wednesday','thursday', 'friday', 'saturday', 'sunday']]
bus = stops_times.merge(weekdays_service, left_on=['trip_id'], right_on=['trip_id'])[['stop_lat', 'stop_lon', 'arrival_time', 'monday', 'tuesday', 'wednesday','thursday', 'friday', 'saturday', 'sunday']]

# add name of the day of the week as column in the dataframe
df['name_day'] = [df.at[i, 'trip_start'].day_name().lower() for i in range(len(df))]


# USE R-TREE SPATIAL INDEX TO CHECK IF EACH BUS STOP IS WITHIN A GIVEN BUFFER (10 meters buffer) FROM ORIGIN
# If a bus stopped near the trip origin within 10 minutes previous to the trip start, then add it to the count of buses. 
df = gpd.GeoDataFrame(df, crs='EPSG:4326', geometry=gpd.points_from_xy(df.trip_origin_longitude, df.trip_origin_latitude))
# R-TREE SPATIAL INDEX INTERSECTION
def intersect_using_spatial_index(source_gdf, intersecting_gdf):
    """
    Conduct spatial intersection using spatial index for candidates GeoDataFrame to make queries faster.
    Note, with this function, you can have multiple Polygons in the 'intersecting_gdf' and it will return all the points 
    intersect with ANY of those geometries.
    """
    source_sindex = source_gdf.sindex
    possible_matches_index = []
    
    # 'itertuples()' function is a faster version of 'iterrows()'
    for other in intersecting_gdf.itertuples():
        bounds = other.geometry.bounds
        c = list(source_sindex.intersection(bounds))
        possible_matches_index += c
    
    # Get unique candidates
    unique_candidate_matches = list(set(possible_matches_index))
    possible_matches = source_gdf.iloc[unique_candidate_matches]

    # Conduct the actual intersect
    result = possible_matches.loc[possible_matches.intersects(intersecting_gdf.unary_union)]
    return result



# POINT-IN-POLYGON search
# Check if trips origin are within a 10 meters buffer from each bus stop.
# If yes, take the index of the bus stops to retrieve the other information (e.g., time and weekday) of that stops from the "bus" dataframe.

trips=[] # e-scooters trips indexes
potential_stops=[] # list of lists; each internal list containing indexes of the possible bus stops at a 10m-distance from the origin
for i in range(len(bus)):
    # keep bus stop if it is within 10 meters buffers for trip origin
    bus_stop_pt = Point(bus.at[i, 'stop_lon'], bus.at[i, 'stop_lat'])
    bus_stop_pt = pd.DataFrame({'x':[bus_stop_pt.x], 'y':[bus_stop_pt.y]})
    bus_stop_pt = gpd.GeoDataFrame(bus_stop_pt, crs='EPSG:4326', geometry=gpd.points_from_xy(bus_stop_pt.x, bus_stop_pt.y))
    bus_stop_pt = bus_stop_pt.to_crs(epsg=32632) # meters 32632 italia utm 32n wgs84
    bus_stop_pt['area_buffer'] = bus_stop_pt.buffer(10).to_crs(epsg=4326).to_list() 
    bus_stop_pt = bus_stop_pt.drop(['geometry'], axis=1)
    bus_stop_buffer = gpd.GeoDataFrame(bus_stop_pt, crs='EPSG:4326', geometry=bus_stop_pt.area_buffer) 
    
    # list of df indexes with possible bus alternatives
    indexes_origin_inside_stop_buffer = list(intersect_using_spatial_index(source_gdf=df, intersecting_gdf=bus_stop_buffer).index)
    trips.append(indexes_origin_inside_stop_buffer)
    potential_stops.append(i) # list of the bus alternatives


# change 24 with 00 (for midnight), in order to convert arrival_time from str to datetime
for h in range(len(bus)):
    if bus.at[h, 'arrival_time'].startswith('24'):
        bus.at[h, 'arrival_time'] = '00'+bus.at[h, 'arrival_time'][2:]
# convert to timestamp
bus['arrival_time'] = pd.to_datetime(bus.arrival_time, format="%H:%M:%S")
# make it tz-aware (with same tz as e-scooters data)
bus.arrival_time = bus.arrival_time.dt.tz_localize('UTC').dt.tz_convert('UTC')


e_scooters_and_num_bus_near_origin={} # {e-scooter index : count of the bus stops near the origin in the previous 10 minutes}
for lst in range(len(trips)):
    if len(trips[lst])>0:
        # all the trips in the list "trips[lst]" have as bus stop near the origin the stop "potential_stops[lst]"
        stop_near = potential_stops[lst]
        for ind in trips[lst]:
            escooter_time = df.iloc[ind].trip_start
            bus_time = bus.iloc[stop_near].arrival_time
            # if the bus arrived at the stop in the 10 minutes previous to the e-scooter trip start
            # and if the bus run in the day of the week in which the e-scooter trip took place
            if bus_time.time() >= (escooter_time - timedelta(minutes = 10)).time() and bus_time.time() <= escooter_time.time() and bus.iloc[stop_near][df.iloc[ind].name_day] != 0:
                if ind in e_scooters_and_num_bus_near_origin.keys():
                    e_scooters_and_num_bus_near_origin[ind] += 1
                else:
                    e_scooters_and_num_bus_near_origin[ind] = 1

# add a column for the count of bus stopping near the e-scooter origin at that time
df = df.reindex(df.columns.tolist() + ['count_bus_multimodality'], axis=1) 
df = df.fillna(0)
# add count of the bus found
for k in e_scooters_and_num_bus_near_origin.keys():
    df.at[k, 'count_bus_multimodality'] = df.at[k, 'count_bus_multimodality'] + e_scooters_and_num_bus_near_origin[k]


# drop geometry
df = df.drop(['geometry'], axis=1)


'''
Add holiday variable as a boolean.
'''

# add holiday variable:
# True if the date represent a holiday in Italy, False otherwise
df['holiday'] = [df.at[i, 'trip_start'] in holidays.Italy() for i in range(len(df))]


# saved augmented dataset dataset
df.to_parquet('C:/Users/luisa/Desktop/thesis_project/data/tripsv3_augmented.parquet')