import os
from dataclasses import dataclass

import hydra
import optuna
import sqlalchemy
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback

from cardiospike import DATA_DIR
from cardiospike.models.cardio_net.training import TrainConfig, train

load_dotenv()

OPTUNA_USER = os.getenv("OPTUNA_USER")
OPTUNA_PASSWORD = os.getenv("OPTUNA_PASSWORD")
OPTUNA_HOST = os.getenv("OPTUNA_HOST")
OPTUNA_PORT = os.getenv("OPTUNA_PORT")
OPTUNA_DATABASE = os.getenv("OPTUNA_DATABASE")

YANDEX_CLOUD_CERT_PATH = str(DATA_DIR / "CA.pem")

OPTUNA_STORAGE_URL = str(
    sqlalchemy.engine.url.URL(
        drivername="mysql+pymysql",
        username=OPTUNA_USER,
        password=OPTUNA_PASSWORD,
        host=OPTUNA_HOST,
        port=OPTUNA_PORT,
        database=OPTUNA_DATABASE,
        query={"ssl_ca": YANDEX_CLOUD_CERT_PATH},
    )
)


@dataclass
class SearchConfig:
    study_name: str = MISSING

    n_trials: int = 100
    train: TrainConfig = TrainConfig()

    @classmethod
    def register(cls):
        cs = ConfigStore.instance()
        TrainConfig.register()
        cs.store(node=cls, name="search")


def search(cfg: SearchConfig):
    def objective(trial: optuna.Trial):
        pruning_callback = PyTorchLightningPruningCallback(monitor="Val/loss", trial=trial)

        cfg.train.cardio_system.alpha = trial.suggest_float("alpha", 0.4, 0.6)
        cfg.train.cardio_system.step_ahead = trial.suggest_int("step_ahead", 3, 6)
        cfg.train.cardio_system.lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)

        cfg.train.cardio_system.channels = int(trial.suggest_loguniform("channels", 16, 256))
        cfg.train.cardio_system.top_classifier_units = trial.suggest_int("top_classifier_units", 64, 1024)

        return train(cfg=cfg.train, pruning_callback=pruning_callback)

    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(
        direction="minimize", storage=OPTUNA_STORAGE_URL, pruner=pruner, study_name=f"cardiospike/{cfg.study_name}"
    )
    study.optimize(objective, n_trials=cfg.n_trials)


@hydra.main(config_path=None, config_name="search")
def main(cfg: SearchConfig):
    search(cfg=cfg)


if __name__ == "__main__":
    SearchConfig.register()

    if not os.path.exists(YANDEX_CLOUD_CERT_PATH):
        os.system(f'wget "https://storage.yandexcloud.net/cloud-certs/CA.pem" -O {YANDEX_CLOUD_CERT_PATH}')

    main()
