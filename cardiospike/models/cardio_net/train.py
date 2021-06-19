import os
from dataclasses import dataclass
from uuid import uuid1

import hydra
import numpy as np
import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from joblib import Parallel, delayed, dump, load
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from cardiospike import NUM_CORES, S3_CHECKPOINTS_DIR, S3_LOGS_DIR, TRAIN_DATA_PATH
from cardiospike.data.preprocessing import load_df, scale_train_valid
from cardiospike.models.cardio_net.neural import CardioSystem, CardioSystemConfig
from cardiospike.models.cardio_net.utils import FOLDS_DATA_DIR, FoldInFoldGenerator
from cardiospike.torch_utils import CardioDataset


def dump_folds_data(win_size=17, output_dir=FOLDS_DATA_DIR, num_workers=NUM_CORES):
    """
    Creates train, val, test datasets for all 25 folds in folds splits and dumps them to disk
    :param win_size:
    :param output_dir:
    :param num_workers:
    :return:
    """

    df = load_df(TRAIN_DATA_PATH)

    fold_in_fold_gen = FoldInFoldGenerator(df)

    bar = tqdm(total=25 * df.shape[0])

    def for_delayed(ids):
        train_ids, val_ids, external_val_ids = ids

        train_df = df.loc[df["id"].isin(train_ids)].reset_index(drop=True).copy()
        val_df = df.loc[df["id"].isin(external_val_ids)].reset_index(drop=True).copy()
        test_df = df.loc[df["id"].isin(val_ids)].reset_index(drop=True).copy()

        train_df, val_df, test_df, scaler = scale_train_valid(train_df, val_df, test_df)

        train_ds = CardioDataset(train_df, win_size, bar=bar)
        val_ds = CardioDataset(val_df, win_size, bar=bar)
        test_ds = CardioDataset(test_df, win_size, bar=bar)

        return {"scaler": scaler, "train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds}

    folds_data = Parallel(n_jobs=num_workers, backend="threading")(
        delayed(for_delayed)(ids) for ids in fold_in_fold_gen
    )

    folds_data = {i: fold_data for i, fold_data in enumerate(folds_data)}

    dump(folds_data, output_dir / f"{win_size}.joblib")


@dataclass
class TrainConfig:
    experiment_name: str = f"CardioNet/{uuid1()}"
    win_size: int = 17
    num_workers: int = NUM_CORES - 1
    batch_size: int = 1024
    patience: int = 75
    max_epochs: int = 200
    gpus: int = 1
    cardio_system: CardioSystemConfig = CardioSystemConfig()

    @classmethod
    def register(cls):
        cs = ConfigStore.instance()
        CardioSystemConfig.register()
        cs.store(node=cls, name="train")


def train(cfg: TrainConfig, pruning_callback=None):
    if not os.path.exists(str(FOLDS_DATA_DIR / f"{cfg.win_size}.joblib")):
        dump_folds_data(win_size=cfg.win_size, num_workers=cfg.num_workers)

    folds_data = load(FOLDS_DATA_DIR / f"{cfg.win_size}.joblib")

    test_results = []

    for i in folds_data:
        train_ds = folds_data[i]["train_ds"]
        val_ds = folds_data[i]["val_ds"]
        test_ds = folds_data[i]["test_ds"]

        train_dataloader = DataLoader(
            train_ds, num_workers=cfg.num_workers, pin_memory=False, shuffle=True, batch_size=cfg.batch_size
        )
        val_dataloader = DataLoader(
            val_ds, num_workers=cfg.num_workers, pin_memory=False, shuffle=False, batch_size=cfg.batch_size
        )
        test_dataloader = DataLoader(
            test_ds, num_workers=cfg.num_workers, pin_memory=False, shuffle=False, batch_size=cfg.batch_size
        )

        system: CardioSystem = hydra.utils.instantiate(cfg.cardio_system)

        experiment_checkpoints_dir = f"{S3_CHECKPOINTS_DIR}/{cfg.experiment_name}/{i}"
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=experiment_checkpoints_dir, monitor="Val/loss")
        early_stopping_callback = pl.callbacks.EarlyStopping(patience=cfg.patience, monitor="Val/loss")
        monitor_gpu_callback = pl.callbacks.GPUStatsMonitor()
        callbacks = [checkpoint_callback, early_stopping_callback, monitor_gpu_callback]

        if pruning_callback is not None:
            callbacks.append(pruning_callback)

        logger = TensorBoardLogger(save_dir=S3_LOGS_DIR, name=f"{cfg.experiment_name}/{i}")

        trainer = pl.Trainer(logger=logger, callbacks=callbacks, gpus=cfg.gpus, max_epochs=cfg.max_epochs)
        trainer.fit(system, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
        test_result = trainer.test(test_dataloaders=test_dataloader, ckpt_path=checkpoint_callback.best_model_path)[0][
            "Test/f1_score"
        ]
        logger.finalize(status="success")

        trainer.save_checkpoint(filepath=f"{experiment_checkpoints_dir}/best.ckpt")

        test_results.append(test_result)

    assert len(test_results) == 25

    return np.mean(np.array(test_results))


@hydra.main(config_path=None, config_name="train")
def main(cfg: TrainConfig):
    train(cfg=cfg)


if __name__ == "__main__":
    TrainConfig.register()

    main()
