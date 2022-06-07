import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import Counter
import folium
import folium.plugins
import branca.colormap as cm
import param
import panel as pn
import warnings
warnings.filterwarnings("ignore")


def build_df():
    df = pd.read_parquet('C:/Users/luisa/Desktop/escooter_trento/produced_datasets/map_matched_edges_ids.parquet')
    df = df.reset_index(drop=True)
    orig_df= pd.read_parquet('C:/Users/luisa/Desktop/escooter_trento/data/trips_pointv3_cleaned.parquet')
    new_df = orig_df.groupby('unique_id', as_index=False).aggregate({"point_timestamp":lambda x: x.to_list()})

    start_time = {}
    for i in range(len(new_df)):
        start_time[new_df.at[i, 'unique_id']] = min(new_df.at[i, 'point_timestamp'])

    # add timestamp to map-matched dataframe
    tmstmp = []
    for row in range(len(df)):
        tmstmp.append(start_time[df.at[row, 'unique_id']])
    df['start_time'] = tmstmp

    f = open('C:/Users/luisa/Desktop/escooter_trento/produced_datasets/ways_id.json')
    d = json.load(f)
    return df, d


def compute_route_freq(df, d, start_h, end_h): 
    '''
    helper function to compute the routes and their frequency for the timewindow
    '''
    data_selected = df.loc[(df.start_time.dt.hour >= start_h) & (df.start_time.dt.hour < end_h)]
    data_selected = data_selected.reset_index()
    
    list_ways = []
    for row in range(len(data_selected)):
        list_ways.extend(data_selected.edge_ids[row])
    count_freq = Counter(list_ways) # count how many times passed through
    list_ways = list(set(list_ways))
    
    routes = []
    cnt = []
    for road in list_ways:
        route = d[str(road)]['coords'] # coordinates (from way_id.json)
        routes.append(route)
        cnt.append(count_freq[road]) # num. times passed through (for the given timewindow -> from counter above)
        
    return routes, cnt


def get_map():
    return folium.Map(location=[46.066667,11.133333], tiles='Stamen Toner', zoom_start=14) 


def render_map(map, routes, cnt):
    '''
    render the map 
    '''
    
    colormap = cm.LinearColormap(colors=['yellow', 'orange', 'red', 'darkred'], 
                                index = np.linspace(min(cnt), max(cnt), num=4),
                                vmin = min(cnt), vmax = max(cnt), 
                                caption='Number of times the segment was passed through')
    
    fg = folium.FeatureGroup(name='overpass_tratte_forti')    

    for way in range(len(routes)): 
        color = colormap(cnt[way])
        w = min(cnt[way]/10, 5)
        folium.vector_layers.PolyLine(routes[way], color=color, weight=w).add_to(fg)

    map.add_child(fg)
    map.add_child(colormap)

    return map




class PanelFoliumMap(param.Parameterized):
    start_hour = param.Integer(14, bounds=(0,23))
    end_hour = param.Integer(15, bounds=(0,23)) 
        
    def __init__(self, **params):
        super().__init__(**params)
        self.map = get_map()
        self.folium_pane = pn.pane.plot.Folium(sizing_mode="stretch_both", min_height=500, min_width=900, margin=0) 
        self.view = pn.Row(pn.Column(pn.pane.Markdown("## Settings"), self.param.start_hour,  self.param.end_hour), self.folium_pane)    
        self._update_map()

    @param.depends("start_hour", "end_hour", watch=True)
    def _update_map(self):
        self.map = get_map()
        self.df, self.d = build_df()
        routes, cnt = compute_route_freq(self.df, self.d, start_h=self.start_hour, end_h=self.end_hour)
        render_map(self.map, routes, cnt)
        self.folium_pane.object = self.map



def main():
    ACCENT_COLOR = "#DAA520"
    template = pn.template.FastListTemplate(
        site="Map", 
        title="Most Travelled Route by Hour(s)", 
        accent_base_color=ACCENT_COLOR, header_background=ACCENT_COLOR, theme='default', theme_toggle=False,
        main=["Select the desired time range by slicing to the extreme hours of the interval of your choice", 
        PanelFoliumMap().view]).show()
    return template


if __name__ == "__main__":
    main()
    

