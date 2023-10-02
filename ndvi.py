import os
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
from scipy.signal import savgol_filter
from datetime import datetime
from satsearch import Search
from shapely.geometry import shape, mapping
from rio_tiler.io import COGReader
from loguru import logger
from joblib import Parallel, delayed

""" Classes of Sentinel-2 L2A Scene Classification (SCL)
    0: No data
    1: Saturated or defective
    2: Dark area pixels
    3: Cloud shadows
    4: Vegetation
    5: Not vegetated
    6: Water
    7: Unclassified
    8: Cloud medium probability
    9: Cloud high probability
    10: Thin cirrus
    11: Snow
    
    function evaluatePixel(samples) {
        const SCL=samples.SCL;
        switch (SCL) {
        // No Data (Missing data) - black   
        case 0: return RGBToColor (0, 0, 0,samples.dataMask);
            
        // Saturated or defective pixel - red 
        case 1: return RGBToColor (255, 0, 0,samples.dataMask);
    
        // Topographic casted shadows ("Dark features/Shadows" for data before 2022-01-25) - very dark grey
        case 2: return RGBToColor (47,  47,  47,samples.dataMask);
            
        // Cloud shadows - dark brown
        case 3: return RGBToColor (100, 50, 0,samples.dataMask);
            
        // Vegetation - green
        case 4: return RGBToColor (0, 160, 0,samples.dataMask);
            
        // Not-vegetated - dark yellow
        case 5: return RGBToColor (255, 230, 90,samples.dataMask);
            
        // Water (dark and bright) - blue
        case 6: return RGBToColor (0, 0, 255,samples.dataMask);
        
        // Unclassified - dark grey
        case 7: return RGBToColor (128, 128, 128,samples.dataMask);
        
        // Cloud medium probability - grey
        case 8: return RGBToColor (192, 192, 192,samples.dataMask);
            
        // Cloud high probability - white
        case 9: return RGBToColor (255, 255, 255,samples.dataMask);
        
        // Thin cirrus - very bright blue
        case 10: return RGBToColor (100, 200, 255,samples.dataMask);
            
        // Snow or ice - very bright pink
        case 11: return RGBToColor (255, 150, 255,samples.dataMask);
    
        default : return RGBToColor (0, 0, 0,samples.dataMask);  
        }
    }
    References: 
    Sentinel Hub. (n.d.). Sentinel-2 L2A scene classification map. Retrieved from https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/
    European Space Agency. (n.d.). Sentinel-2 User Handbook. Retrieved from https://sentinels.copernicus.eu/documents/247904/685211/Sentinel-2_User_Handbook
"""
CLOUD_CLASSES = [3, 8, 9, 10]
BANDS = ['nir08', 'red', 'scl']
NODATA = None

class Constants:
    STAC_API_URL = 'https://earth-search.aws.element84.com/v1'
    COLLECTION = 'sentinel-2-l2a'
    CLOUD_COVER_LIMIT = 5

@logger.catch
def s2cloudless(pixel_data):
    """ Mask the NIR and RED bands based on the SCL band (s2cloudless). """
    # Check if the pixel in the 'scl' band belongs to any cloud class
    is_cloud = pixel_data['scl'] in CLOUD_CLASSES

    if is_cloud:
        pixel_data['nir08'] = NODATA
        pixel_data['red'] = NODATA

    return pixel_data

@logger.catch
def get_pixel_value(urls, lon, lat, datetime):
    band_values = {}
    band_values['datetime'] = datetime

    for band_name, url in zip(BANDS, urls):
        try:
            with COGReader(url) as cog:
                point_data = cog.point(lon, lat)
                value = point_data.array.data[0] if point_data.array is not None else None
                band_values[band_name] = value
        except Exception as e:
            logger.error(f"Failed to get pixel value for band {band_name} in dataset {url}. Error: {e}")
            band_values[band_name] = None

    # Apply the s2cloudless mask
    band_values = s2cloudless(band_values)

    return band_values


@logger.catch
def process_period(id, start_year, end_year, geometry):
    start_date = datetime(start_year, 1, 1, 0, 0, 0).isoformat() + 'Z'
    end_date = datetime(end_year + 1, 1, 1, 0, 0, 0).isoformat() + 'Z'
    date_range = f'{start_date}/{end_date}'

    search = Search(
        url=Constants.STAC_API_URL,
        intersects=mapping(geometry),
        datetime=date_range,
        collections=[Constants.COLLECTION],
        query={'eo:cloud_cover': {'lt': Constants.CLOUD_COVER_LIMIT}},
        limit=1000,
    )

    items = sorted(
        search.items(), key=lambda item: item.properties['eo:cloud_cover']
    )

    # tasks = []
    if items:
        for item in items:
            logger.info(f"{item.assets['scl']}")

        results = Parallel(n_jobs=-1)(delayed(get_pixel_value)([f"{item.assets[b]['href']}" for b in BANDS], geometry.x, geometry.y, item.datetime) for item in items)
        if results:
            # Convert list of dictionaries to list of tuples
            df = pd.DataFrame(results)
            df = df.sort_values(by='datetime')
            plot_ndvi(df)

def plot_ndvi(df):
    # Calculate NDVI
    df['ndvi'] = (df['nir08'] - df['red']) / (df['nir08'] + df['red'])
    df['ndvi'] = df['ndvi'].replace([np.inf, -np.inf], np.nan)
    
    # Savitzky-Golay smoothing
    # window_size = 5 
    # poly_order = 2 
    # df['ndvi_smooth'] = savgol_filter(df['ndvi'], window_size, poly_order)
    
    # Plotting
    fig = px.line(df, x='datetime', y='ndvi', title="NDVI over time")
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='NDVI')
    fig.show()

@logger.catch
def execute(start_year, end_year, geojson_file):
    gdf = gpd.read_file(geojson_file)

    # Take only the first two rows
    gdf = gdf.iloc[:1]

    logger.info(f"CRS: {gdf.crs}")

    for _, row in gdf.iterrows():
        geometry = row['geometry']
        id = row['ID']
        geom = shape(geometry)
        if geom.type != 'Point':
            logger.error('Geometry is not a Point.')
            continue
        process_period(id, start_year, end_year, geom)
      
try:
    start_year = 2017
    end_year = 2023
    geojson_file = os.path.abspath('./points.geojson')
    execute(start_year, end_year, geojson_file)
except Exception as e:
    logger.exception(e)
    pass