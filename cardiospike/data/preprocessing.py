from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler


def prepare_data(train_df, val_df, numerical_features=["x"]):  # noqa
    scaler = RobustScaler()

    train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
    val_df[numerical_features] = scaler.transform(val_df[numerical_features])

    return train_df, val_df, scaler


class TrainValGenerator:
    def __init__(self, data, n_splits=5, shuffle=True, random_state=42):
        self.data = data
        self.kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def __iter__(self):
        for train_ids, val_ids in self.kf.split(self.data["id"].unique()):
            data = self.data.copy()
            train_df = data.loc[data["id"].isin(train_ids)].reset_index(drop=True)
            val_df = data.loc[data["id"].isin(val_ids)].reset_index(drop=True)

            train_df, val_df, scaler = prepare_data(train_df=train_df, val_df=val_df)

            yield train_df, val_df, scaler
