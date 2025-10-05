import numpy as np
import pandas as pd
from google.cloud import bigquery

def fetch_sequences_from_bq(project_id, dataset_table, max_rows=10000):
    """Fetch sequence windows from BigQuery and format as NumPy arrays."""
    client = bigquery.Client(project=project_id)
    query = f"SELECT seq_window FROM `{dataset_table}` LIMIT {max_rows}"
    df = client.query(query).to_dataframe()

    sequences, winds, steers = [], [], []
    for _, row in df.iterrows():
        seq = row["seq_window"]
        arr = np.array([
            [
                i.get("lat", 0.0),
                i.get("lon", 0.0),
                i.get("wind_1min_max", 0.0),
                i.get("pressure_msl", 0.0),
                i.get("mean_backscatter", 0.0),
                i.get("sst", 0.0),
                i.get("precip_24h", 0.0),
                i.get("steering_u", 0.0),
                i.get("steering_v", 0.0),
            ]
            for i in seq
        ])
        input_seq = arr[:-1, :]
        target_wind = arr[-1, 2]
        u, v = arr[-1, 7], arr[-1, 8]
        steering_angle = np.degrees(np.arctan2(v, u))
        sequences.append(input_seq)
        winds.append(target_wind)
        steers.append(steering_angle)

    X = np.array(sequences)
    y = {
        "next_max_wind": np.array(winds).reshape(-1, 1),
        "next_steering_angle": np.array(steers).reshape(-1, 1),
    }
    return X, y
