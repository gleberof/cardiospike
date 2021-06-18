import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from cardiospike import TRAIN_DATA_PATH


def clean_pulse(df):
    df["pulse"] = 60.0 * 1000.0 / df["x"]
    df.loc[(df.pulse < 30) | (df.pulse > 220), "x"] = np.nan
    df["x"] = df["x"].interpolate()
    return df


def scale_ts(vls, save_name=None, quantiles=(25, 75)):
    sc = RobustScaler(quantile_range=quantiles)
    res = sc.fit_transform(vls.reshape(-1, 1)).ravel()
    if save_name is not None:
        joblib.dump(sc, save_name)
    return res, sc


def scale_sts(vls):
    sc = StandardScaler()
    res = sc.fit_transform(vls.reshape(-1, 1)).ravel()
    return res


def scale_time_ts(vls):
    sc = MinMaxScaler()
    return sc.fit_transform(vls.reshape(-1, 1)).ravel()


def load_train():
    df = pd.read_csv(TRAIN_DATA_PATH)
    df = clean_pulse(df)
    df["log_x"] = np.log1p(df["x"].values)

    return df


def scale_train_valid(train_df, valid_df):
    for feat in ["log_x", "x"]:
        sc = RobustScaler(quantile_range=(20, 95))
        sc.fit(train_df[feat].values.reshape(-1, 1))
        train_df[feat] = sc.transform(train_df[feat].values.reshape(-1, 1))
        valid_df[feat] = sc.transform(valid_df[feat].values.reshape(-1, 1))
    return train_df, valid_df
