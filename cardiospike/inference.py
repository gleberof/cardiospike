from typing import List

import joblib
import numpy as np
import pandas as pd


def pop_arr(arr, num):
    s = len(arr)
    return np.resize(np.repeat(np.resize(arr[:s], (1, s)), num, axis=0), (num * s,))


def time_domain(rr, mask):
    results = {}

    rr = rr[mask == 1]

    if len(rr) > 1:
        hr = 60000 / rr

        results["mean_RR"] = np.mean(rr)
        results["std_rr_sdnn"] = np.std(rr)
        results["mean_hr_kubious"] = 60000 / np.mean(rr)
        results["mean_hr"] = np.mean(hr)
        results["std_hr"] = np.std(hr)
        results["min_hr"] = np.min(hr)
        results["max_hr"] = np.max(hr)
        results["rmssd"] = np.sqrt(np.mean(np.square(np.diff(rr))))
        results["nn_xx"] = np.sum(np.abs(np.diff(rr)) > 50) * 1
        results["pnn_xx"] = 100 * np.sum((np.abs(np.diff(rr)) > 50) * 1) / len(rr)
    else:
        results["mean_RR"] = 0
        results["std_rr_sdnn"] = 0
        results["mean_hr_kubious"] = 0
        results["mean_hr"] = 0
        results["std_hr"] = 0
        results["min_hr"] = 0
        results["max_hr"] = 0
        results["rmssd"] = 0
        results["nn_xx"] = 0
        results["pnn_xx"] = 0

    return results


class DummyModel:
    def __init__(self):
        pass

    @staticmethod
    def predict(rr_requence: List[int]):
        return (
            np.random.rand(len(rr_requence)).tolist(),
            0.4,
            np.random.rand(len(rr_requence)).tolist(),
            np.random.rand(),
        )


class SmartModel:
    def __init__(self, path_to_model: str):
        print(path_to_model)
        self.model = joblib.load(path_to_model)
        self.anomaly_thresh = 0.49288617615370894
        self.error_thresh = 0.5
        self.window_size = 17
        self.half_window = int((self.window_size - 1) / 2)
        self.quart_windows = int((self.window_size - 1) / 4)
        self.low_bound = 300
        self.upper_bound = 1400

    def prep_data(self, rr: List[int]):
        rr = np.array(rr)  # type: ignore
        mask = ((rr > self.low_bound) & (rr < self.upper_bound)).astype(int)  # type: ignore

        rr = np.concatenate(
            (
                pop_arr(rr[:2], self.quart_windows),
                rr,
                pop_arr(rr[-2:], self.quart_windows),
            )
        )

        mask = np.concatenate((pop_arr([0], self.half_window), mask, pop_arr([0], self.half_window)))

        result = []
        for i in range(self.half_window + 1, len(rr) - self.half_window + 1):
            local_window = rr[i - self.half_window - 1 : i + self.half_window + 2]
            local_mask = mask[i - self.half_window - 1 : i + self.half_window + 2]
            result.append(
                {
                    **{f"x_{x}": y for x, y in zip(range(-8, 9), local_window)},
                    **{f"mask_{x}": y for x, y in zip(range(-8, 9), local_mask)},
                    **time_domain(local_window, local_mask),
                    **{f"delta_{x}": y for x, y in zip(range(self.window_size - 1), np.diff(local_window))},
                }
            )
        return result

    def predict(self, rr_requence: List[int]):

        df = pd.DataFrame(self.prep_data(rr_requence))

        return (
            self.model.predict_proba(df)[:, 1].tolist(),
            self.anomaly_thresh,
            df["mask_0"].tolist(),
            self.error_thresh,
        )
