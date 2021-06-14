from typing import List

import numpy as np


class DummyModel:
    def __init__(self):
        pass

    @staticmethod
    def predict(rr_requence: List[int]):
        return np.random.randint(0, 2, len(rr_requence)).tolist(), np.random.randint(0, 2, len(rr_requence)).tolist()
