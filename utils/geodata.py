import geopandas as geopd
import numpy as np
from shapely.geometry import Point
from functools import lru_cache

file_dir = "/Users/mosselveen/dev/extremeweather/.geodata/cb_2018_us_state_20m/"
file_path = file_dir + "cb_2018_us_state_20m.shp"

"""
https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html
"""

excluded_territories = [
    "PR",  # Puerto Rico
    "AS",  # American Samoa
    "VI",  # United States Virgin Islands
    "GU",  # Guam
    "MP",  # Commonwealth of the Northern Mariana Islands
    "HI",  # Hawaii
    "AK",  # Alaska
]


def usa_gdf():
    states_gdf = geopd.read_file(file_path)
    states_gdf["dissolve_field"] = 1

    states_l48 = states_gdf.loc[~states_gdf["STUSPS"].isin(excluded_territories)].copy()
    states_l48 = states_l48.dissolve(by="dissolve_field")
    return states_l48


def usa_polygon():

    gdf = usa_gdf()
    return gdf["geometry"].values[0]


@lru_cache()
def usa_grid(n_width=100, n_height=100):

    polygon = usa_polygon()

    (minx, miny, maxx, maxy) = polygon.bounds

    x_vals = np.linspace(minx, maxx, n_width)
    y_vals = np.linspace(miny, maxy, n_height)

    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)

    x_1d = x_mesh.ravel()
    y_1d = y_mesh.ravel()

    mask = np.array([Point(x, y).within(polygon) for (x, y) in zip(x_1d, y_1d)])
    mask_mesh = mask.reshape(x_mesh.shape)

    return x_mesh, y_mesh, mask_mesh
