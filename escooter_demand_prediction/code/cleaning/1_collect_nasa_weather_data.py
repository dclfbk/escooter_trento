import yaml
from typing import List, Union, Optional
from pathlib import Path
from datetime import date, datetime
import requests
import pandas as pd

'''
Script to download meteorological data records from NASA Power (https://power.larc.nasa.gov/data-access-viewer/).
Using this script, we collect information related to temperature, humidity and precipitation, wind and pressure. 
For information on queried parameters check: https://power.larc.nasa.gov/docs/tutorials/parameters/#availability.
Main contributions to this script from: https://github.com/kdmayer/nasa-power-api

Hourly data available at:
https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M&community=RE&longitude=11.133&latitude=46.067&start=20201130&end=20220504&format=CSV
'''

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class PowerAPI:
    """
    Query the NASA Power API.
    Check https://power.larc.nasa.gov/ for documentation
    Attributes
    ----------
    url : str
        Base URL
    """
    url = "https://power.larc.nasa.gov/api/temporal/daily/point?"

    def __init__(self,
                 start: Union[date, datetime, pd.Timestamp],
                 end: Union[date, datetime, pd.Timestamp],
                 long: float, lat: float,
                 use_long_names: bool = False,
                 parameter: Optional[List[str]] = None):
        """
        Parameters
        ----------
        start: Union[date, datetime, pd.Timestamp]
        end: Union[date, datetime, pd.Timestamp]
        long: float
            Longitude as float
        lat: float
            Latitude as float
        use_long_names: bool
            NASA provides both identifier and human-readable names for the fields. If set to True this will parse
            the data with the latter
        parameter: Optional[List[str]]
            List with the parameters to query.
            Default is ['T2M_RANGE', 'TS', 'T2MDEW', 'T2MWET', 'T2M_MAX', 'T2M_MIN', 'T2M', 'T10M', 'QV2M', 'RH2M',
                        'PRECTOTCORR', 'PS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX',
                        'WS50M_MIN', 'WS50M_RANGE']
        """
        self.start = start
        self.end = end
        self.long = long
        self.lat = lat
        self.use_long_names = use_long_names
        if parameter is None:
            self.parameter = ['T2M_RANGE', 'TS', 'T2MDEW', 'T2MWET', 'T2M_MAX', 'T2M_MIN', 'T2M', 'T10M', 'QV2M', 'RH2M',
                              'PRECTOTCORR', 'PS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX',
                              'WS50M_MIN', 'WS50M_RANGE']

        self.request = self._build_request()

    def _build_request(self) -> str:
        """
        Build the request
        Returns
        -------
        str
            Full request including parameter
        """
        r = self.url
        r += f"parameters={(',').join(self.parameter)}"
        r += '&community=RE'
        r += f"&longitude={self.long}"
        r += f"&latitude={self.lat}"
        r += f"&start={self.start.strftime('%Y%m%d')}"
        r += f"&end={self.end.strftime('%Y%m%d')}"
        r += '&format=JSON'

        return r

    def get_weather(self) -> pd.DataFrame:
        """
        Main method to query the weather data
        Returns
        -------
        pd.DataFrame
            Pandas DataFrame with DateTimeIndex
        """

        response = requests.get(self.request)

        assert response.status_code == 200

        data_json = response.json()

        records = data_json['properties']['parameter']

        df = pd.DataFrame.from_dict(records)

        return df




## MAIN

def main():

    # read configuration file
    config_file = 'nasa_config.yaml'

    with open(config_file, 'rb') as f:

        conf = yaml.load(f, Loader=yaml.FullLoader)

    latitude = conf.get('lat', 0)
    longitude = conf.get('lon', 0)
    start_date = conf.get('start_date', 0)
    end_date = conf.get('end_date', 0)
    output_dir = conf.get('output_dir', 0)

    output_dir = Path(output_dir) / f"{longitude};{latitude}.csv"

    nasa_weather = PowerAPI(start=pd.Timestamp(str(start_date)), end=pd.Timestamp(str(end_date)), long=longitude, lat=latitude)

    weather_df = nasa_weather.get_weather()

    weather_df.to_csv(output_dir, sep=";")

if __name__ == '__main__':

    main()