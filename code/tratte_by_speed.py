import pandas as pd
import json
import folium
import branca.colormap as cm
import numpy as np
import param
import panel as pn
import warnings
warnings.filterwarnings("ignore")

'''
Velocità sulle strade battute dai monopattini per fasce orarie.

Esegui usando:

panel serve tratte_by_speed.py
'''


def get_df():
    df = pd.read_parquet('C:/Users/luisa/Desktop/escooter_trento/data/mm_wayID_speed_name.parquet')
    df = df.reset_index(drop=True)

    f = open('C:/Users/luisa/Desktop/escooter_trento/data/ways_id.json')
    d = json.load(f)
    return df, d

def compute_route_speed(df, d, start_h, end_h): 
    '''
    helper function to compute the routes and their average speed for the selected timewindow
    '''
    data_selected = df.loc[(df.start_time.dt.hour >= start_h) & (df.start_time.dt.hour < end_h)].reset_index(drop=True)
    
    # build dictionary with wayID and average speed
    # for the selected hour (slide dataframe according to hour)
    way_speed = {}
    
    for i in range(len(data_selected)):
        sp = data_selected.at[i, 'speed'] # speed of the i-th trip
        # for each edge/segment in the route of the i-th trip, add speed to the segment values
        for ed in data_selected.at[i, 'edges_id']:
            if ed in way_speed.keys():
                way_speed[ed].append(sp)
            else:
                way_speed[ed] = []
                way_speed[ed].append(sp)

    for k in way_speed.keys():
        avg = sum(way_speed[k]) / len(way_speed[k])
        way_speed[k] = avg

    # get coordinates for every street segment from ways_id.json 
    routes = []
    avg_speed = []
    names = []
    for id in way_speed.keys():
        routes.append(d[str(id)]['coords'])
        avg_speed.append(way_speed[id])
        names.append(d[str(id)]['name'])

    return routes, avg_speed, names

def get_map():
    return folium.Map(location=[46.066667,11.133333], tiles='Stamen Toner', zoom_start=14) 

def render_map(map, routes, avg_speed, names):
    '''
    render the map 
    '''
    
    colormap = cm.LinearColormap(colors=['gray', '#E7E56F', '#80C348', '#348538'],  
                                index = np.linspace(min(avg_speed), max(avg_speed), num=4),
                                vmin = min(avg_speed), vmax = max(avg_speed), 
                                caption='Speed at which ways have been passed through on average (in m/s)')
    
    fg = folium.FeatureGroup(name='speed_tratte_forti')    

    for way in range(len(routes)): 
        color = colormap(avg_speed[way])
        w = min(avg_speed[way], 5)
        t = names[way] + ' (average speed: ' + str(round(avg_speed[way], 2)) + 'm/s)'

        iframe = folium.IFrame(names[way] + '<br>Average speed: ' + str(round(avg_speed[way], 2)) + 'm/s')
        p = folium.Popup(iframe, min_width=300, max_width=300)

        folium.vector_layers.PolyLine(routes[way], tooltip=t, popup=p, color=color, weight=w).add_to(fg)

    map.add_child(fg)
    map.add_child(colormap)

    return map



class PanelFoliumMap(param.Parameterized):
    start_hour = param.Integer(0, bounds=(0,23))
    end_hour = param.Integer(1, bounds=(0,23)) 
    button = param.Action(lambda x: x.param.trigger('button'), label='Update Map')
        
    def __init__(self, **params):
        super().__init__(**params)
        self.map = get_map()
        self.folium_pane = pn.pane.plot.Folium(sizing_mode="stretch_both", min_height=500, min_width=900, margin=0) 
        self.view = pn.Row(pn.Column(pn.pane.Markdown("## Settings"), self.param.start_hour,  self.param.end_hour, self.param.button), self.folium_pane)    
        self._update_map()

    @param.depends("button", watch=True) # "start_hour", "end_hour", 
    def _update_map(self):
        self.map = get_map()
        self.df, self.d = get_df()
        routes, avg_speed, names = compute_route_speed(self.df, self.d, start_h=self.start_hour, end_h=self.end_hour)
        render_map(self.map, routes, avg_speed, names)
        self.folium_pane.object = self.map




########################################

ACCENT_COLOR = "#44AD49"
template = pn.template.FastListTemplate(
        site="Map", 
        title="Route by Average Speed for selected Hour(s)", 
        accent_base_color=ACCENT_COLOR, header_background=ACCENT_COLOR, theme='default', theme_toggle=False,
        main=["Select the desired time range using the sliders and then click the button below to update the map.", 
        PanelFoliumMap().view]).servable();
