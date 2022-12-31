import pandas as pd
import pandas_gbq as pbq
from google.cloud import bigquery

client = bigquery.Client()

cache_path = "~/dev/extremeweather/.cache/"


fields_daily = {
    "PRCP": 10.0,  # Percipitation in mm
    "TMAX": 10.0,  # Max daily temp
    "TMIN": 10.0,  # Min daily temp
}


class NOAAStore:
    def __init__(self):
        self.hdf_path = cache_path + ".noaa_store.h5"

    def _keys(self):
        with pd.HDFStore(self.hdf_path) as store:
            keys = store.keys()
        return keys

    def _stations_get(self):

        qry = """
        SELECT
            stations.id,
            stations.latitude,
            stations.longitude,
            stations.elevation,
            stations.state,
            stations.name
        FROM bigquery-public-data.ghcn_d.ghcnd_stations AS stations
        ORDER BY stations.id
        """

        df = pbq.read_gbq(qry)
        df["country"] = df["id"].str.slice(0, 2)
        df = df.set_index("id")
        return df

    def _stations_inventory_get(self):

        qry = """
        SELECT
            inventory.id,
            inventory.element,
            inventory.firstyear,
            inventory.lastyear
        FROM `bigquery-public-data.ghcn_d.ghcnd_inventory` AS inventory
        ORDER BY id, element
        """

        df = pbq.read_gbq(qry)
        df = df.set_index(["id", "element"])
        df = df[["firstyear", "lastyear"]].astype("int64")
        return df

    def _get_table(self, key, data_getter, overwrite=False):

        with pd.HDFStore(self.hdf_path) as store:
            if (key in store) & (not overwrite):
                df = store.get(key)
            else:
                df = data_getter()
                store.put(key, df, format="table", data_columns=True)
        return df

    def stations(self, overwrite=False):
        return self._get_table("stations", self._stations_get, overwrite=overwrite)

    def stations_inventory(self, element=None, include_info=False, overwrite=False):
        df = self._get_table(
            "inventory", self._stations_inventory_get, overwrite=overwrite
        )

        if element is not None:
            df = df.xs(element, axis="index", level="element")
        if include_info:
            df = df.join(self.stations())

        return df

    def _timeseries_universe_get(self, element):

        qry = f"""
        SELECT data_daily.date, data_daily.id, data_daily.value
        FROM `bigquery-public-data.ghcn_d.ghcnd_*` AS data_daily
        JOIN `bigquery-public-data.ghcn_d.ghcnd_inventory` AS inventory
            ON data_daily.id = inventory.id
            AND inventory.element = '{element}'
            AND inventory.firstyear < 1950
            AND inventory.lastyear >= 2022
        WHERE REGEXP_CONTAINS(_TABLE_SUFFIX, '^(19|20)[0-9]{{2}}$')
            AND data_daily.element = '{element}'
            AND data_daily.date >= '1900-01-01'
            AND data_daily.id LIKE 'US%'
        ORDER BY data_daily.date, data_daily.id
        """

        df = pbq.read_gbq(qry)
        df["value"] /= fields_daily[element]
        df["date"] = df["date"].astype("datetime64[ns]")
        return df.set_index(["date", "id"])["value"]

    def timeseries_universe(self, element, overwrite=False):

        assert element in fields_daily.keys()

        key = element + "_US"

        with pd.HDFStore(self.hdf_path) as store:
            if (key in store) & (not overwrite):
                df = store.get(key)
            else:
                df = self._timeseries_universe_get(element)
                store.put(key, df, format="fixed")

        return df
