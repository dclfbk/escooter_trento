# The Spatio-Temporal Demand for E-Scooters: Analysing and Modelling Riders’ Behaviour in Trento

The following project departs from the analysis on e-scooters data carried out during the internship period at Fondazione Bruno Kessler (Trento) and aims to further improve the understanding of the phenomena by developing models to predict e-scooters demand.
All previous analysis, which can be considered a prerequisite to understand following considerations, are available in the following [Digital Commons Lab repository](https://github.com/DigitalCommonsLab/escooter_trento).

## Data description
This project mainly relies on the `tripsv3.parquet` dataset, containing data about e-scooters trips in the city of Trento from 2020-11-30 to 2022-05-04. Each row describes a trip, providing several temporal and spatial information; in particular, the raw dataset contains:
- `operator`: name of the operator providing the service (BIT or VENTO).            
- `trip_operator_id`: operator ID; assumes values "Bit Mobility", "WIND" e "Tier", where Wind and Tier represent the same operator, since Wind was bought by Tier, which was also successively acquired by Vento.
- `trip_vehicle_id`: unique identifier of the vehicle, for a total of 645 e-scooters.             
- `trip_id`: trip identifier; ID repeats themselves after an unspecified amount of time, thus there is the need to introduce some change in order for the trip ID to be unique.  
- `trip_start` and `trip_end`: start and end timestamps.        
- `trip_start_epoch` and `trip_end_epoch`: unix timestamps; describe the start and end time of a trip (redundant information).              
- `trip_origin_time` and `trip_destination_time`: datetime type, reporting information about starting and ending time of each trip; redundant information, as it can be obtained from previous timestamps. 
- `trip_origin_latitude` and `trip_origin_longitude`: coordinates of the vehicle when the trip starts.
- `trip_destination_latitude` and `trip_destination_longitude`: coordinates of the vehicle when it reaches the destination.   
- `trip_points_num`: number of GPS points recorded within the trip trajectory.            
- `trip_points_numall`: reporting information similar to *trip_points_num*.   
- `trip_length`: distance, in meters.    
- `trip_mode`: type of vehicle used for the trip; only assumes "scooter" as value.      
- `trip_accuracy`: accuracy, assumes values 15., 1. e nan.
- `trip_duration_break_excluded`: trip duration, excluded breaks within the trip.
- `trip_user_id`: all None values, as no information identifying physical people is available.
- `trip_properties`: all None values. 

### Other important data and data sources used:
- `ambito_esercizio_monopattini.shp`: shapefile describing the area where e-scooters transit is allowed. Trips outside this area are filtered out, as they are a result of mistakes, unallowed behaviours, or movements done by the operators for maintenance purposes.
- Weather data have been collected using [**NASA Power**](https://power.larc.nasa.gov/data-access-viewer/) and [**World Weather**](https://www.worldweatheronline.com/developer) services.
- Data concerning the point of interests (POIs) have been requested to **OpenStreetMap** and are available at the following [link](https://osmit-estratti.wmcloud.org/dati/poly/comuni/pbf/022205_Trento_poly.osm.pbf).
- The **General Transit Feed Specification (GTFS)** - *google_transit_urbano.zip* - was downloaded by the website of the [Province transport service](https://www.trentinotrasporti.it/open-data). GTFS references are available at [link](https://developers.google.com/transit/gtfs/reference).

## Goal

The goal of this project is to study riders' usage of e-scooters in Trento, to try to infer the places characterized by greatest demand and their characteristics, along with the time of the days, week, and year in which the demand peaks. After analysing users' behaviours (in previous analysis, and partially within this project), the aim is to predict e-scooters demand at different level of spatio-temporal aggregation. 

## Structure
List and brief description of scripts and notebooks.

*Code folder*:

*Analysis* folder:  
- `timeseries_decomp.ipynb`: seasonal decomposition using moving averages and analysis of the different time series characterizing each zone.
- `trips_purposes.ipynb`: inference and analysis of trips purposes using multiple-data sources (OpenStreetMap POIs, origin and destination of e-scooters trips).
- `vehicle_usage.ipynb`: analysis of cumulative time of usage per vehicle; space-time cube for visualizing vehicles usage; trajectories analysis; distribution of trips per travelled distance and time.
- `stats.ipynb` & `visualizations.ipynb`: few statistics and visualizations of the temporal and spatial distributions of trips, via calendar plot, boxplot, maps, networks, etc.
- `ST_forecast.R`, `IDW.R`: R scripts exploring spatio-temporal statistical models: in order, they present generalized linear model (glm), spatial and spatio-temporal models with fixed rank kriging (frk), and inverse distance weighting (idw).

*Cleaning* folder:  
- `0_data_cleaning.py`: contains the steps performed to clean the data and remove noises, preparing them for further processing. All the operations performed are described at the beginning of the script.
- `1_collect_weather_info.py`: script to collect data concerning the weather from the World Weather Portal; requires an API key in a congifuration file (check configuration paragraph below). 
- `2_augment_data.py`: script to augment the data concerning e-scooters movements with the ones about weather, POIs, multimodality, and holidays. 
- `2_compute_geohash.py`: script to compute geohash and perform grid division of the territory. 
- `3_feature_importance.ipynb`: analysis of feature importance in predicting the target variable and reduction of redundant/not significant information.

*Base_models* folder:  
- `prepare_data.py`: handle data augmentation and missing timesteps in the time series, preparing the final dataframe according to the chosen spatio-temporal aggregation.
- `pre_arima.ipynb` & `ARIMA.ipynb`: the first notebooks performs ACF and PACF analysis, while the second develop ARIMA and SARIMA for different temporal aggregations and seasonality periods.
- `random_forest.py` & `RF.ipynb` implementation of the random forest algorithm for different spatio-temporal aggregations. Optmization of the paramters was performed using random search and grid search, and features importance have been considered using Gini Information criterion.
- `knn.py` & `KNN.ipynb`: implementation of the k-nearest neighbours algorithm and optimization of the k parameters, for different spatio-temporal aggregations. Previous analysis in the `spatial_interpolation.ipynb`.

*Deep_models* folder:  
- `BidirectionalLSTM.py`: bidirectional LSTM model.
- `GNN_LSTN_basis.py`: first implementation of a model based on graph convolutions and LSTM, not considering exogenous factors that may influence the phenomena. 
- `GNN_LSTN_data_prep.py`: functions to preprocess the data to be used in the GNN-LSTM model.
- `SpatioTemporalModel.py`: keras implementation of a model using both graph convolutions, on the spatial component, and LSTM modules on the temporal axis.
- `GNN_LSTN_exog_feats.py`: training of the GNN-LSTM model, incorporating exogenous features to enhance the prediction.
- `utils.py`: utils functions.

The repository also contains a list of online resources and a literature review with references for improving the understanding of the phenomena under study, contained in the *docs* folder.

## Requirements
### Installing Dependencies
The project is mainly developed in Python. 
To install all the needed python libraries, use the following command: 

```bash
pip install -r requirements.txt
```

### Configuration
In order to run the collect_weather_info.py, it is necessary to create a config.py file in the current working directory and provide the API key of from the [World Weather Portal](https://www.worldweatheronline.com/developer/) as follows:

```python
my_worldweatheronline_api_key = YOUR_API_KEY
```