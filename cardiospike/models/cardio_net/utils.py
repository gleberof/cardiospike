from sklearn.model_selection import KFold

from cardiospike import DATA_DIR

kf = KFold(n_splits=5, shuffle=True, random_state=42)
inner_kf = KFold(n_splits=5, shuffle=True, random_state=239)

FOLDS_DATA_DIR = DATA_DIR / "folds_data"


class FoldInFoldGenerator:
    def __init__(self, df):
        self.df = df
        self.dataset_ids = self.df["id"].unique()

    def __iter__(self):

        for i, (tr_id, va_id) in enumerate(kf.split(self.dataset_ids)):
            external_val_ids = [self.dataset_ids[x] for x in va_id]
            internal_ids = [self.dataset_ids[x] for x in tr_id]

            for j, (itr_id, iva_id) in enumerate(inner_kf.split(internal_ids)):
                train_ids = [internal_ids[x] for x in itr_id]
                val_ids = [internal_ids[x] for x in iva_id]

                yield train_ids, val_ids, external_val_ids
