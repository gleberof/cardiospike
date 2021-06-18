from dataclasses import dataclass
from typing import Any, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from torch.nn import BCEWithLogitsLoss

from cardiospike.torch_utils import Lookahead, Ralamb
from cardiospike.utils.math import threshold_search


class AttentionWeightedAverage(nn.Module):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, hidden_dim: int, return_attention: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.return_attention = return_attention

        self.attention_vector = nn.Parameter(
            torch.empty(self.hidden_dim, dtype=torch.float32),
            requires_grad=True,
        )
        nn.init.xavier_normal_(self.attention_vector.unsqueeze(-1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits = x.matmul(self.attention_vector)
        ai = (logits - logits.max()).exp()

        att_weights = ai / (ai.sum(dim=1, keepdim=True) + 1e-9)
        weighted_input = x * att_weights.unsqueeze(-1)
        output = weighted_input.sum(dim=1)

        if self.return_attention:
            return output, att_weights
        else:
            return output, None


class CardioNet(nn.Module):
    def __init__(self, output_size=1, channels=32, top_classifier_units=512, rnn_units=16):  # noqa
        super().__init__()

        self.kern_sizes = [3, 5, 7, 9]

        self.convs = nn.ModuleDict(
            {
                f"conv_{ks}": nn.Sequential(
                    nn.BatchNorm1d(2),
                    nn.Conv1d(2, channels, ks),
                    nn.Dropout(0.5),
                    nn.GELU(),
                    nn.BatchNorm1d(channels),
                    nn.Conv1d(channels, channels, ks),
                    nn.Dropout(0.5),
                    nn.GELU(),
                )
                for ks in self.kern_sizes
            }
        )
        self.max = nn.AdaptiveMaxPool1d(1)
        self.avg = nn.AdaptiveAvgPool1d(1)

        self._gru = nn.GRU(input_size=2, num_layers=1, hidden_size=rnn_units, batch_first=True, bidirectional=True)

        self._attn = AttentionWeightedAverage(2 * rnn_units)
        self.sum_chans = 305

        self._head = nn.Sequential(
            nn.LayerNorm(self.sum_chans),
            nn.Linear(self.sum_chans, top_classifier_units),
            nn.LayerNorm(top_classifier_units),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.Linear(top_classifier_units, top_classifier_units),
            nn.LayerNorm(top_classifier_units),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(top_classifier_units, output_size),
        )

    def forward(self, x_feats):
        encoded, last_state = self._gru(x_feats)

        conv_outputs = []

        for conv in self.convs:
            conv_result = self.convs[conv](x_feats.transpose(1, 2))
            conv_outputs.append(self.max(conv_result).squeeze(2))
            conv_outputs.append(self.avg(conv_result).squeeze(2))

        encoded_outputs = [self.max(encoded).squeeze(2), self._attn(encoded)[0]]

        all_outputs = conv_outputs + encoded_outputs

        all_outputs_cat = torch.cat(all_outputs, dim=1)

        return self._head(all_outputs_cat)


def Over9000(params, alpha=0.5, k=6, *args, **kwargs):
    ralamb = Ralamb(params, *args, **kwargs)
    return Lookahead(ralamb, alpha, k)


@dataclass
class CardioSystemConfig:
    channels: int = 32
    top_classifier_units: int = 512
    rnn_units: int = 16

    lr: float = 1e-3
    alpha: float = 0.5
    step_ahead: int = 6

    _target_: str = "cardiospike.models.cardio_net.neural.CardioSystem"

    @classmethod
    def register(cls):
        cs = ConfigStore.instance()
        cs.store(node=cls, name="cardio_system")


class CardioSystem(pl.LightningModule):
    def __init__(
        self, output_size=1, channels=32, top_classifier_units=512, rnn_units=16, lr=1e-3, alpha=0.5, step_ahead=6
    ):
        super().__init__()

        self.save_hyperparameters(
            {
                "lr": lr,
                "alpha": alpha,
                "step_ahead": step_ahead,
                "channels": channels,
                "top_classifier_units": top_classifier_units,
                "rnn_units": rnn_units,
                "hp_metric": -1,
            }
        )

        self.model = CardioNet(
            output_size=output_size, channels=channels, top_classifier_units=top_classifier_units, rnn_units=rnn_units
        )
        self.loss = BCEWithLogitsLoss()
        self.thresh = None

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_index):
        return self.common_step(batch, batch_index, "Train")

    def validation_step(self, batch, batch_index):
        return self.common_step(batch, batch_index, "Val")

    def test_step(self, batch, batch_index):
        return self.common_step(batch, batch_index, "Test")

    def validation_epoch_end(self, outputs) -> None:
        self.common_epoch_end(outputs, "Val")

    def test_epoch_end(self, outputs):
        self.hparams.hp_metric = self.common_epoch_end(outputs, "Test")
        self.logger.log_hyperparams(self.hparams)

    def common_step(self, batch, batch_idx, mode):
        inpts = batch["x"]
        actual = batch["y"]
        proba = self.model(inpts).flatten()

        loss = self.loss(proba, actual)

        self.log(f"{mode}/loss", loss)

        return {"loss": loss, "proba": proba, "actual": actual}

    def common_epoch_end(self, outputs, mode):
        actual = torch.cat([out["actual"] for out in outputs])
        proba = torch.cat([out["proba"] for out in outputs])

        pred = torch.sigmoid(proba)

        best_th, best_score = threshold_search(actual.cpu().numpy(), pred.cpu().numpy())

        self.log(f"{mode}/thresh", best_th.item())
        self.log(f"{mode}/f1_score", best_score.item())

        return best_score

    def configure_optimizers(self):
        optimizer = Lookahead(
            torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr), self.hparams.alpha, self.hparams.step_ahead
        )

        return optimizer
