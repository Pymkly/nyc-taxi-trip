import sqlite3

from sklearn.compose import ColumnTransformer

import common
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

def load_train_data(path):
    print(f"Reading train data from the database: {path}")
    con = sqlite3.connect(path)
    _sql = f'SELECT * FROM {common.TRAIN_TABLE}'
    data_train = pd.read_sql(_sql, con)
    con.close()
    X = data_train.drop(columns=[common.TARGET_NAME])
    # X = common.preprocess_data(X)
    y = data_train[common.TARGET_NAME]
    y = common.transform_y(y)
    return X, y


def fit_model(X, y):
    num_features = common.NUM_FEATURES
    cat_features = common.CAT_FEATURES
    train_features = num_features + cat_features
    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), cat_features),
        ('scaling', StandardScaler(), num_features)]
    )
    pipeline = Pipeline(steps=[
        ('ohe_and_scaling', column_transformer),
        ('regression', Ridge())
    ])
    print(X[train_features])
    print(y)
    model_ = pipeline.fit(X[train_features], y)
    return model_
    # return None

if __name__ == "__main__":
    X_train, y_train = load_train_data(common.DB_PATH)
    X_train = common.preprocess_data(X_train)
    model = fit_model(X_train, y_train)
    common.persist_model(model, common.MODEL_PATH)