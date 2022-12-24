# import pandas as pd
import pandas_gbq as pbq
from functools import lru_cache
from google.cloud import bigquery

client = bigquery.Client()


@lru_cache()
def bigquery2df(query_str):
    query = client.query(query_str)
    results = query.result()
    return results.to_dataframe()


fields_daily = {
    'PRCP': 10.0,  # Percipitation in mm
    'TMAX': 10.0,  # Max daily temp
    'TMIN': 10.0,  # Min daily temp
}


def stations_info():

    query_str = '''
            WITH date_counts AS (
            SELECT id,
                COUNT(hist_data.date) as date_count,
                MIN(hist_data.date) as date_min,
                MAX(hist_data.date) as date_max
            FROM `bigquery-public-data.ghcn_d.ghcnd_*` AS hist_data
            WHERE REGEXP_CONTAINS(_TABLE_SUFFIX, '^[0-9]{4}$')
                AND hist_data.element = 'PRCP'
            GROUP BY 1
            )

            SELECT
                date_counts.id,
                stations.* EXCEPT (id,source_url,etl_timestamp),
                date_counts.date_count,
                date_counts.date_min,
                date_counts.date_max
            FROM date_counts
            JOIN bigquery-public-data.ghcn_d.ghcnd_stations AS stations
                ON stations.id = date_counts.id
            ORDER BY date_counts.date_count DESC
    '''

    df = pbq.read_gbq(query_str)
    df = df.set_index('id')

    date_cols = ['date_min', 'date_max']
    df[date_cols] = df[date_cols].astype("datetime64[ns]")
    return df


def hist_daily(ids: tuple, field: str, date_start=None, date_end=None):

    assert field in fields_daily.keys()

    if date_start is not None and date_end is not None:
        range_str = f"AND date BETWEEN '{date_start}' AND '{date_end}'"
    else:
        range_str = ""

    ids_str = str(tuple(ids))
    daily_qry = f'''
        SELECT id, date, value
        FROM `bigquery-public-data.ghcn_d.ghcnd_*`
        WHERE REGEXP_CONTAINS(_TABLE_SUFFIX, '^[0-9]{{4}}$')
            AND id IN {ids_str}
            AND element = '{field}'
            {range_str}
        ORDER BY date
    '''

    df = pbq.read_gbq(daily_qry)
    df['value'] /= fields_daily['PRCP']
    df['date'] = df['date'].astype("datetime64[ns]")
    return df.set_index(['date', 'id'])['value'].unstack('id')
