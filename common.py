import pickle
import os
import numpy as np
import pandas as pd
import sqlite3
from api.model import TripRequest

# project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config.ini')

# Using INI configuration file
from configparser import ConfigParser

config = ConfigParser()
config.read(CONFIG_PATH)
DB_PATH = str(config.get("PATHS", "DB_PATH"))
MODEL_PATH = str(config.get("PATHS", "MODEL_PATH"))
RANDOM_STATE = int(config.get("ML", "RANDOM_STATE"))
TARGET_NAME = str(config.get("ML", "TARGET_NAME"))
TRAIN_TABLE = str(config.get("ML", "TRAIN_TABLE"))
TEST_TABLE = str(config.get("ML", "TEST_TABLE"))
NUM_FEATURES = ['log_distance_haversine', 'hour',
                    'abnormal_period', 'is_high_traffic_trip', 'is_high_speed_trip',
                    'is_rare_pickup_point', 'is_rare_dropoff_point',
                    'vendor_id', 'store_and_fwd_flag']
CAT_FEATURES = ['weekday', 'month']

# # Doing the same with a YAML configuration file
# import yaml
#
# with open("config.yml", "r") as f:
#     config_yaml = yaml.load(f, Loader=yaml.SafeLoader)
#     DB_PATH = str(config_yaml['paths']['db_path'])
#     MODEL_PATH = str(config_yaml['paths']["model_path"])
#     RANDOM_STATE = int(config_yaml["ml"]["random_state"])

# SQLite requires the absolute path
# DB_PATH = os.path.abspath(DB_PATH)
DB_PATH = os.path.join(ROOT_DIR, os.path.normpath(DB_PATH))

# def to_dataframe(trip: TripRequest):
#     param = {
#         "vendor_id": trip.vendor_id,
#         "pickup_datetime": trip.pickup_datetime.to_datetime_str(),
#         "dropoff_datetime": trip.dropoff_datetime.to_datetime_str(),
#         "passenger_count": trip.passenger_count,
#         "pickup_longitude": trip.pickup_longitude,
#         "pickup_latitude": trip.pickup_latitude,
#         "dropoff_longitude": trip.dropoff_longitude,
#         "dropoff_latitude": trip.dropoff_latitude,
#         "store_and_fwd_flag": trip.store_and_fwd_flag
#     }
#     return pd.DataFrame.from_dict(param)

def load_data(_path, _table_name):
    print(f"Reading {_table_name} data from the database: {_path}")
    con = sqlite3.connect(_path)
    _sql = f'SELECT * FROM {_table_name}'
    data_train = pd.read_sql(_sql, con)
    con.close()
    return extract_x_y(data_train)

def extract_x_y(_data):
    X = _data.drop(columns=[TARGET_NAME])
    y = _data[TARGET_NAME]
    y = transform_y(y)
    return X, y

def preprocess_data(X):
    print(f"Preprocessing data")
    X['pickup_datetime'] = pd.to_datetime(X['pickup_datetime'])
    X['pickup_date'] = X['pickup_datetime'].dt.date
    X = step1_add_features(X)
    X = step2_add_features(X)
    X = step3_process_features(X)
    return X

def step3_process_features(X):
  res = X.copy()
  res['vendor_id'] = res['vendor_id'].map({1:0, 2:1})
  res['store_and_fwd_flag'] = res['store_and_fwd_flag'].map({'N':0, 'Y':1})
  res.loc[res['passenger_count'] > 6, 'passenger_count'] = 6

  return res

def step2_add_features(X):
    res = X.copy()
    distance_haversine = haversine_array(res.pickup_latitude, res.pickup_longitude, res.dropoff_latitude, res.dropoff_longitude)
    res['log_distance_haversine'] = np.log1p(distance_haversine)
    res['is_high_traffic_trip'] = is_high_traffic_trip(X).astype(int)
    res['is_high_speed_trip'] = is_high_traffic_trip(X).astype(int)
    res['is_rare_pickup_point'] = is_rare_point(X, "pickup_latitude", "pickup_longitude", 0.01, 0.995, 0, 0.95).astype(int)
    res['is_rare_dropoff_point'] = is_rare_point(X, "dropoff_latitude", "dropoff_longitude", 0.01, 0.995, 0.005, 0.95).astype(int)
    return res

def is_rare_point(X, latitude_column, longitude_column, qmin_lat, qmax_lat, qmin_lon, qmax_lon):
  lat_min = X[latitude_column].quantile(qmin_lat)
  lat_max = X[latitude_column].quantile(qmax_lat)
  lon_min = X[longitude_column].quantile(qmin_lon)
  lon_max = X[longitude_column].quantile(qmax_lon)

  res = (X[latitude_column] < lat_min) | (X[latitude_column] > lat_max) | \
        (X[longitude_column] < lon_min) | (X[longitude_column] > lon_max)
  return res

def is_high_traffic_trip(X):
  return ((X['hour'] >= 8) & (X['hour'] <= 19) & (X['weekday'] >= 0) & (X['weekday'] <= 4)) | \
         ((X['hour'] >= 13) & (X['hour'] <= 20) & (X['weekday'] == 5))

def is_high_speed_trip(X):
  return ((X['hour'] >= 2) & (X['hour'] <= 5) & (X['weekday'] >= 0) & (X['weekday'] <= 4)) | \
         ((X['hour'] >= 4) & (X['hour'] <= 7) & (X['weekday'] >= 5) & (X['weekday'] <= 6))

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def get_abnormal_dates(X):
    df_abnormal_dates = X.groupby('pickup_date').size()
    # abnormal_dates = df_abnormal_dates[df_abnormal_dates < df_abnormal_dates.quantile(0.02)]
    abnormal_dates = df_abnormal_dates[df_abnormal_dates < 6300]
    return abnormal_dates


def step1_add_features(X):
    abnormal_dates = get_abnormal_dates(X)
    res = X.copy()
    res['weekday'] = res['pickup_datetime'].dt.weekday
    res['month'] = res['pickup_datetime'].dt.month
    res['hour'] = res['pickup_datetime'].dt.hour
    res['abnormal_period'] = res['pickup_datetime'].dt.date.isin(abnormal_dates.index).astype(int)
    return res

def transform_y(y):
    return np.log1p(y).rename('log_' + y.name)


def persist_model(model, path):
    print(f"Persisting the model to {path}")
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(path, "wb") as file:
        pickle.dump(model, file)
    print(f"Done")

def load_model(path):
    print(f"Loading the model from {path}")
    with open(path, "rb") as file:
        model = pickle.load(file)
    print(f"Done")
    return model
