import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CardioDataset(Dataset):
    def __init__(self, df, win_size=32):
        self.df = df
        self.win_size = win_size

        self.point_indexes = []
        self.win_lens = []

        dfs = []
        total_len = 0
        for q, qdf in self.df.groupby("id"):
            for i in range(max(1, qdf.shape[0] - win_size + 1)):
                self.point_indexes.append(i + total_len)
                if i + win_size > qdf.shape[0]:
                    self.win_lens.append(qdf.shape[0] - i)
                else:
                    self.win_lens.append(win_size)
            total_len += qdf.shape[0]
            dfs.append(qdf)
        self.df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)

    def __len__(self):
        return len(self.point_indexes)

    def __getitem__(self, idx):
        i0 = self.point_indexes[idx]
        i1 = i0 + self.win_lens[idx]

        x_mat = np.zeros((self.win_size, 2))
        y_mat = np.zeros(self.win_size)
        x_mat[-self.win_lens[idx] :, 1] = self.df.iloc[i0:i1].x.values
        y_mat[-self.win_lens[idx] :] = self.df.iloc[i0:i1].y.values

        return {"x": x_mat, "y": y_mat, "start": i0, "end": i1}
