import pandas as pd
import numpy as np
import pyrosm
import requests 
import pyproj as proj
import shapely
import geopandas as gpd
import geohash as pgh
import networkx as nx
import tensorflow as tf
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from polygon_geohasher.polygon_geohasher import geohashes_to_polygon
from sklearn.metrics.pairwise import cosine_similarity
from utils import build_df_sparse
import warnings
warnings.filterwarnings("ignore")


## PREPARE DATA
## Returns:
# - DATAFRAME (grid cells as column, timestamps as rows, count of trips per cell per time as values)
# - TEMPORAL FEATURES (normalized)
# - NEIGHBOURING ADJ. MATRIX (refered to grid cells having common boundaries)
# - MAPPING NUMBER<->CELL GEOHASH (gh_to_int, int_to_gh)

def prepare_data(precision=6, freq='H'):
    df = build_df_sparse(precision=precision, freq=freq)
    df = df.reset_index()

    # BUILD ADJACENCY MATRIX: 
    # PUT CONNECTION IF THE CELLS ARE NEIGHBOURS
    # encode again to get all geohashes
    geoh = [pgh.encode(gh[0], gh[1], precision) for gh in df.gh.unique()]
    # get all neighbours cells, so that we can put a connection among them in the graph/adjacency matrix
    edges = []
    for current_gh in geoh:
        for neighbor in pgh.neighbors(current_gh):
            if neighbor in geoh: # check that the neighbor is not outside the study area of Trento
                edges.append((current_gh, neighbor))
    # build adjacency matrix based on cells neighbouring
    d = {'nodes':geoh,
        'edges':edges}
    # graph
    g = nx.DiGraph()
    g.add_nodes_from(d['nodes'])
    g.add_edges_from(d['edges'])
    # to adj. matrix
    adjacency_matrix = nx.to_pandas_adjacency(g)
    # add self-loops to adjacency matrix (to consider also the node 
    # itself in the update, as its previous state also influences the following)
    for i in range(len(adjacency_matrix)):
        adjacency_matrix.iloc[i,i] = 1.0

    # BUILD MAPPINGS NUM <-> GEOHASH: 
    # mapping centroid of cell to number and viceversa
    # useful in case you want to retrieve the cell and its location
    df['gh'] = df.gh.astype('str')
    all_gh = df.gh.unique()
    gh_to_int = {gh_id:i for i, gh_id in enumerate(all_gh)}
    int_to_gh = {i:gh_id for i, gh_id in enumerate(all_gh)}

    # replace centroids with the above defined numbers 
    df['gh'] = [gh_to_int[df.at[i, 'gh']] for i in range(len(df))]


    # temporal features
    temporal_feats = df.drop(['gh', 'trips_count'], axis=1).drop_duplicates().reset_index(drop=True).sort_values(by=['trip_start']).drop(['trip_start'], axis=1)
    # normalize each variable
    for col in temporal_feats.columns:
        temporal_feats[col] = (temporal_feats[col]-temporal_feats[col].mean())/temporal_feats[col].std()

    
    # df with only timeseries with trips count and cells geohash
    df = df.set_index('trip_start')[['trips_count', 'gh']]
    df = df[['gh', 'trips_count']]
    # each geohash as column, timestamps as rows, and as values trips count per time per cell
    df = pd.crosstab(df.index, df.gh, values=df.trips_count, aggfunc='sum')
    
    return df, adjacency_matrix, gh_to_int, int_to_gh, temporal_feats


## spatial index to find polygons intersections

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



## FUNCTIONAL SIMILARITY MATRIX

def pois_matrix(gh_to_int):
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

    # extract latitude and longitude and convert to float
    coords=[]
    for k in gh_to_int.keys():
        coords.append([float(k.split(',')[0][1:]), float(k.split(',')[1][:-1])])


    # encode centroid to retrieve geohash codes
    precision=6
    geoh = [pgh.encode(gh[0], gh[1], precision) for gh in coords]


    # create list of lists, each list must contain a single polygon/geohash
    geoh_ll = [[g] for g in geoh]
    # get polygons from geohashes
    polygons=[]
    for el in geoh_ll:
        polygons.append(geohashes_to_polygon(el))


    # build geodataframe of cells polygons and of POIs -> to compute intersection of poly-POIs using spatial indexes
    pois_gpd = gpd.GeoDataFrame(pois, crs='EPSG:4326', geometry=pois.geometry)
    poly_df = pd.DataFrame({})
    poly_df['geometry'] = polygons
    poly_gdf = gpd.GeoDataFrame(poly_df, crs='EPSG:4326', geometry=poly_df.geometry)


    # pois_per_cell -> dict: keys cells number (0-39), values dictionary, containing as keys the POI primary categories and as values their counts within the cell
    pois_per_cell={}
    for pl in range(len(poly_gdf)):
        # initialize a dictionary for the cell where the count of each POI type is zero
        pois_per_cell[pl] = {}
        for k in primary_categories.keys():
            pois_per_cell[pl][k] = -1
        # create geodataframe with only the current cell polygon
        single_poly = poly_gdf[poly_gdf.geometry==poly_gdf.at[pl, 'geometry']]
        # compute list of indices of the POIs that intersect the current cell 
        indx_intersection = intersect_using_spatial_index(source_gdf=pois_gpd, intersecting_gdf=single_poly).index
        # update counts of the primary category of POIs intersecting the current cell
        for idx in indx_intersection:
            if pois_per_cell[pl][pois_gpd.at[idx, 'primary_category'][0]] == -1:
                pois_per_cell[pl][pois_gpd.at[idx, 'primary_category'][0]] = 1
            pois_per_cell[pl][pois_gpd.at[idx, 'primary_category'][0]] += 1


    # build vectors of POIs counts per cell
    pois_vectors = []
    for k in pois_per_cell:
        pois_vectors.append(np.array(list(pois_per_cell[k].values())).reshape(1, -1))


    ## BUILD MATRIX OF COSINE SIMILARITY OF POIs VECTORS
    matrix_pois = pd.DataFrame({})

    for vect in range(len(pois_vectors)):
        row = []
        for other_vect in range(len(pois_vectors)):
            # vect -> number identifying the cell
            # pois_vectors[vect] -> vector of POIs counts of cell "vect"
            row.append(cosine_similarity(pois_vectors[vect], pois_vectors[other_vect])[0][0])
        matrix_pois[vect] = row
        
    return matrix_pois




## PREPROCESS DATA
def preprocess(df, train_size=0.65, validation_size=0.15, test_size=0.2):
    '''
    Splits data into trainining, validation and test sets
    and normalize the data based on training mean and standard deviation.
    Takes as input the dataframe with the data, having as rows the moment in time in which the data were recorded 
    and as columns the locations (geohash obtained cells) in which the recording was made.
    '''
    np_data = df.to_numpy()
    if train_size + validation_size + test_size != 1.0:
        raise ValueError('Percentages do not sum up to 100%')
        
    num_timestamps = np_data.shape[0] # number of rows in the dataframe (timestamps recorded)

    # get length of training and validation based on the passed percentages
    num_train = int(num_timestamps * train_size) # default: 65% of the data
    num_val = int(num_timestamps * validation_size) # default: 15% of the data
    
    # get training data as np array
    train_array = np_data[:num_train]
    # compute mean and std on training data
    avg = train_array.mean()
    std = train_array.std()

    # normalize based on train mean and std
    train = (train_array - avg) / std
    validation = (np_data[num_train : (num_train + num_val)] - avg) / std
    test = (np_data[(num_train + num_val) :] - avg) / std

    return train, validation, test, std, avg


## PRODUCE TENSORFLOW DATAFRAMES:
## modified function to comprehend temporal features
def create_tf_dataset(data_array: np.ndarray, temp_feats: np.ndarray, input_sequence_length: int, forecast_horizon: int, batch_size: int = 128, shuffle=False, multi_horizon=True):
    '''
    Creates tensorflow dataset from numpy array (np_data -> in the shape (recorded_times, num_gh_cells)).
    Each element of the dataset is a tuple (input, output), where the input is a tensor of shape
    (batch_size, input_sequence_length, num_gh_cells, 1+exog) containing "input_sequence_length" past values of the timeseries for each node/cell,
    where 1+exog corresponds to the target variable plus the 11 exogenous variables.
    The output is a tensor of shape (batch_size, forecast_horizon, num_gh_cells) containing the "forecast_horizon"-number of future values in time
    for each cell.

    NOTE: If multi_horizon = True, the output will be the valueS (plural) of the timeseries for the steps ahead from 1 to forecast_horizon. 
          If multi_horizon = False, the output will be the value (only one) of the timeseries ahead of forecast_horizon steps.
    '''

    # inputs to make prediction: previous value of the num_trips + all 11 features
    l = np.expand_dims(data_array[:-forecast_horizon], axis=-1)
    final=[]
    for single_df_row in range(len(l)):
        new = []
        for i in range(len(l[single_df_row])):
            new.append(np.append(temp_feats.iloc[single_df_row].to_list(), l[single_df_row][i])) # (inputs, target)
        final.append(np.array(new))
    final = np.array(final)

    inputs = timeseries_dataset_from_array(final, None, 
                                        sequence_length = input_sequence_length,
                                        shuffle = False,
                                        batch_size = batch_size)

    # target: the num_trips in the following timestep
    target_offset = ( input_sequence_length
                        if multi_horizon
                        else input_sequence_length + forecast_horizon - 1 )

    target_seq_length = forecast_horizon if multi_horizon else 1

    
    targets = timeseries_dataset_from_array( data_array[target_offset:], None,
                                            sequence_length = target_seq_length,
                                            shuffle = False,
                                            batch_size = batch_size)

    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)

    return dataset.prefetch(16).cache() 
