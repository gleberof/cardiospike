import pandas as pd

from cardiospike import TRAIN_DATA_PATH
from cardiospike.data.preprocessing import TrainValGenerator


def get_train_val_generator(data_path=TRAIN_DATA_PATH):
    data = pd.read_csv(data_path)
    return TrainValGenerator(data=data)
