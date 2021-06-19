import multiprocessing
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

NUM_CORES = multiprocessing.cpu_count()

ROOT_DIR = Path(__file__).parents[0].parents[0]
DATA_DIR = ROOT_DIR / "data"
SUBMISSIONS_DIR = DATA_DIR / "submissions"

TRAIN_DATA_PATH = DATA_DIR / "train.csv"

S3_CHECKPOINTS_DIR: str = str(os.environ.get("S3_CHECKPOINTS_DIR"))

assert S3_CHECKPOINTS_DIR != "None"

S3_LOGS_DIR: str = str(os.environ.get("S3_LOGS_DIR"))

assert S3_LOGS_DIR != "None"


DB_COLORS = {
    "violet": "#804bf2",
    "blue": "#1940ff",
    "dirty-blue": "#2d8ca7",
    "turquoise": "#68e4b2",
    "green": "#9fe54a",
    "yellow": "#ffbe00",
    "red": "#ff1c60",
}
