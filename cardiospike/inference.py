from typing import List

import numpy as np


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
