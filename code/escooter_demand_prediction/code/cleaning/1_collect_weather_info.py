from wwo_hist import retrieve_hist_data
from config import *

'''
Collect historical data related to weather using worldweatheronline service.

To run this script you must create an account in the worldweatheronline.com portal and use the personal API key that is provided to you.
You should create a config.py file with your key in it, specified as follows:
my_worldweatheronline_api_key = YOUR_API_KEY

The service is free for 3 months and does not require a credit card for the free trial.

Documentation: https://www.worldweatheronline.com/developer/api/docs/historical-weather-api.aspx
'''

frequency = 1 # specify frequency of data collection; here get data for every hour
start_date = '30-NOV-2020' 
end_date = '4-MAY-2022'
api_key = my_worldweatheronline_api_key # from config.py, get api key
location_list = ['46.067,11.133'] # specify location as lat,lon (Trento)

hist_weather_data = retrieve_hist_data(api_key,
                                location_list,
                                start_date,
                                end_date,
                                frequency,
                                location_label = False,
                                export_csv = True,
                                store_df = True)