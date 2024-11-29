import common
from api.model import TripRequest
import pandas as pd
import numpy as np

model = common.load_model(common.MODEL_PATH)

def predict(trip: TripRequest):
    trip_d = trip.model_dump()
    X = pd.DataFrame(trip_d, index=[0])

    # print(X)
    X = common.preprocess_data(X)
    X = X[common.NUM_FEATURES+common.CAT_FEATURES]
    print(X.columns)
    print(X)
    print(model.n_features_in_)
    # print(model.features_names_in_)
    # return ""
    return np.expm1(model.predict(X)[0])