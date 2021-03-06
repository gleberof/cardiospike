import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


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


def load_df(path):
    df = pd.read_csv(path)
    df = clean_pulse(df)
    df["log_x"] = np.log1p(df["x"].values)

    return df


def scale_train_valid(train_df, val_df, test_df=None):
    for feat in ["log_x", "x"]:
        scaler = RobustScaler(quantile_range=(20, 95))
        scaler.fit(train_df[feat].values.reshape(-1, 1))
        train_df[feat] = scaler.transform(train_df[feat].values.reshape(-1, 1))
        val_df[feat] = scaler.transform(val_df[feat].values.reshape(-1, 1))

        if test_df is not None:
            test_df[feat] = scaler.transform(test_df[feat].values.reshape(-1, 1))

    return train_df, val_df, test_df, scaler
