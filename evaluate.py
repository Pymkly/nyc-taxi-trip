import sqlite3
import pandas as pd
from sklearn.metrics import mean_squared_error

import common

def load_test_data(_path):
    return common.load_data(_path, common.TEST_TABLE)

def evaluate_model(model_, X, y):
    print(f"Evaluating the model")
    y_pred = model_.predict(X)
    score = mean_squared_error(y, y_pred)
    return score

if __name__ == "__main__":
    X_test, y_test = load_test_data(common.DB_PATH)
    X_test = common.preprocess_data(X_test)
    print(X_test.columns)
    model = common.load_model(common.MODEL_PATH)
    score_test = evaluate_model(model, X_test, y_test)
    print(f"Score on test data {score_test:.2f}")