import pandas as pd
import geohash
from geohash import encode

'''
Use geohash to divide the city into cells of equal size and map each trip starting point to a cell.
Use a precision equal to 7, so that each cell is of dimension 153m x 153m 
Demo and reference: http://www.movable-type.co.uk/scripts/geohash.html
'''

# precision (aka num charachter of the hash)
# highest precision means lower dimension of the cell; 
# e.g., precision = 7, cell dimension ≤ 153m × 153m
# precision = 6, cell dimension ≤ 1.22km × 0.61km
# precision = 5, cells dimensions ≤ 4.89km × 4.89km
precision = 6 

def addToGeohash(m, latitude, longitude):
    '''
    Add point to the mapping function connecting trips origin to the cell hash.
    The origin-point is added to the values associated to the identifier (hash) of its cell.
    '''
    p = (latitude, longitude)
    ph = encode(p[0], p[1], precision)  
    if ph not in m:
        m[ph] = []
    m[ph].append(p)
    return

def getGeohash(m, latitude, longitude):
    '''
    Get geohash identifying the cell to which the passed point belongs to.
    '''
    p = (latitude, longitude)
    ph = encode(p[0], p[1], precision)
    if ph in m:
        return ph
    raise ValueError('Not found')



df = pd.read_parquet('C:/Users/luisa/Desktop/thesis_project/data/tripsv3_augmented.parquet')
m = dict()

gh = []

for pt in range(len(df)):
    addToGeohash(m, df.at[pt, 'trip_origin_latitude'], df.at[pt, 'trip_origin_longitude']) 
    gh.append(getGeohash(m, df.at[pt, 'trip_origin_latitude'], df.at[pt, 'trip_origin_longitude']))

df['geohash_origin'] = gh
df.to_parquet('C:/Users/luisa/Desktop/thesis_project/data/tripsv3_augmented.parquet')

