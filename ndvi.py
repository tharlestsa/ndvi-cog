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


BANDS = ['nir08', 'red']

class Constants:
    STAC_API_URL = 'https://earth-search.aws.element84.com/v1'
    COLLECTION = 'sentinel-2-l2a'
    CLOUD_COVER_LIMIT = 5

def get_pixel_value(urls, lon, lat, datetime):
    band_values = {}
    band_values['datetime'] = datetime
    for band_name, url in zip(BANDS, urls):
        try:
            with COGReader(url) as cog:
                point_data = cog.point(lon, lat)
                # Extracting value from the masked array
                value = point_data.array.data[0] if point_data.array is not None else None
                band_values[band_name] = value
        except Exception as e:
            logger.error(f"Failed to get pixel value for band {band_name} in dataset {url}. Error: {e}")
            band_values[band_name] = None

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