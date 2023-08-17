import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_parquet_data(file_path):
    import pandas as pd

    data_dir = f"{file_path}"

    ride_df = pd.read_parquet(f"{data_dir}/ride_df_agg/")
    alight_df = pd.read_parquet(f"{data_dir}/alight_df_agg/")

    ride_df['lon'] = ride_df['lon'].astype('double')
    ride_df['lat'] = ride_df['lat'].astype('double')

    alight_df['lon'] = alight_df['lon'].astype('double')
    alight_df['lat'] = alight_df['lat'].astype('double')

    ride_df = ride_df.drop(["NODE_ID"], axis = 1)
    alight_df = alight_df.drop(["NODE_ID"], axis = 1)

    return ride_df, alight_df