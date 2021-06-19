import math
from collections import defaultdict

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from cardiospike.data.preprocessing import scale_ts


class Lookahead(Optimizer):
    # Lookahead implementation from
    # https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/lookahead.py

    """Lookahead Optimizer Wrapper.
    Implementation modified from: https://github.com/alphadl/lookahead.pytorch
    Paper: `Lookahead Optimizer: k steps forward, 1 step back` - https://arxiv.org/abs/1907.08610
    """

    def __init__(self, base_optimizer, alpha=0.5, k=6):  # noqa
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if not 1 <= k:
            raise ValueError(f"Invalid lookahead steps: {k}")
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if "slow_buffer" not in param_state:
                param_state["slow_buffer"] = torch.empty_like(fast_p.data)
                param_state["slow_buffer"].copy_(fast_p.data)
            slow = param_state["slow_buffer"]
            slow.add_(group["lookahead_alpha"], fast_p.data - slow)
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        # print(self.k)
        # assert id(self.param_groups) == id(self.base_optimizer.param_groups)
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group["lookahead_step"] += 1
            if group["lookahead_step"] % group["lookahead_k"] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {(id(k) if isinstance(k, torch.Tensor) else k): v for k, v in self.state.items()}
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            "state": state_dict["state"],
            "param_groups": state_dict["param_groups"],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        slow_state_new = False
        if "slow_state" not in state_dict:
            print("Loading state_dict from optimizer without Lookahead applied.")
            state_dict["slow_state"] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],  # this is pointless but saves code
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = self.base_optimizer.param_groups  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)


class Ralamb(Optimizer):
    # RAdam + LARS
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(Ralamb, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Ralamb, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("Ralamb does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                state["step"] += 1
                buffered = self.buffer[int(state["step"] % 10)]

                if state["step"] == buffered[0]:
                    N_sma, radam_step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        radam_step_size = math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    else:
                        radam_step_size = 1.0 / (1 - beta1 ** state["step"])
                    buffered[2] = radam_step_size

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)

                # more conservative since it's an approximated value
                radam_step = p_data_fp32.clone()
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    radam_step.addcdiv_(-radam_step_size * group["lr"], exp_avg, denom)
                else:
                    radam_step.add_(-radam_step_size * group["lr"], exp_avg)

                radam_norm = radam_step.pow(2).sum().sqrt()
                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
                if weight_norm == 0 or radam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / radam_norm

                state["weight_norm"] = weight_norm
                state["adam_norm"] = radam_norm
                state["trust_ratio"] = trust_ratio

                if N_sma >= 5:
                    p_data_fp32.addcdiv_(-radam_step_size * group["lr"] * trust_ratio, exp_avg, denom)
                else:
                    p_data_fp32.add_(-radam_step_size * group["lr"] * trust_ratio, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


class CardioDataset(Dataset):
    def __init__(self, df, win_size=33, aug=None, bar=None):
        self.df = df.sort_values(["id", "time"]).reset_index(drop=True).copy()
        self.win_size = win_size

        self.start_index = []
        self.end_index = []
        self.win_start = []
        self.win_end = []

        self.aug = aug

        total_len = 0
        for q, qdf in self.df.groupby("id"):
            for i in range(qdf.shape[0]):
                self.start_index.append(max(total_len, total_len + i - win_size // 2))
                self.win_start.append(i - win_size // 2)
                self.win_end.append(i + win_size // 2 + 1 - qdf.shape[0])
                self.end_index.append(min(total_len + qdf.shape[0], total_len + i + win_size // 2 + 1))
            total_len += qdf.shape[0]

        self.preproc(bar=bar)

    def preproc(self, bar=None):
        bar = bar if bar is not None else tqdm(total=self.df.shape[0])
        N = self.df.shape[0]

        self.X = np.zeros((N, self.win_size, 2))

        for idx in range(N):
            i0 = self.start_index[idx]
            i1 = self.end_index[idx]
            s = i1 - i0

            for i, feat in enumerate(["x", "log_x"]):
                q = self.df.iloc[i0:i1][feat].values
                mq = np.flip(q, axis=0)

                if self.win_start[idx] < 0:
                    sp = -self.win_start[idx]
                    assert sp + s == self.win_size
                    merged = np.concatenate([mq[-sp:], q])

                    self.X[idx, :, i], _ = scale_ts(merged[: self.win_size])  # merged[:self.win_size] #

                elif self.win_end[idx] > 0:
                    sp = self.win_end[idx]
                    assert sp + s == self.win_size
                    merged = np.concatenate([q, mq[:sp]])
                    self.X[idx, :, i], _ = scale_ts(merged[: self.win_size])  # merged[:self.win_size] #
                else:
                    assert i1 - i0 == self.win_size
                    self.X[idx, :, i], _ = scale_ts(q)  # q #

            bar.update(n=1)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]

        if (self.aug is not None) and (np.random.rand() < self.aug):
            a = np.linspace(0, 0.1, self.win_size)
            aa = np.concatenate([a, a]).reshape(self.win_size, 2)
            if np.random.rand() >= 0.5:
                x += aa
            else:
                x -= aa

        y = self.df.y.values[idx].astype("float32")

        return {"x": x.astype("float32"), "y": y}
