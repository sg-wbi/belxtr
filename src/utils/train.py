import json
from typing import Any

import torch
from loguru import logger
from transformers import get_scheduler


class SaveBestCallback:
    def __init__(self):
        self.last_best = 0

    def save_metrics(
        self,
        path: str,
        metrics: dict[str, float],
        epoch: int | None = None,
        step: int | None = None,
    ):
        with open(path, "a") as fp:
            d = json.dumps(metrics | {"epoch": epoch, "step": step})
            fp.write(f"{d}\n")

    def save(self, metric: float, progress_bar: bool = False) -> bool:
        save = False
        if metric > self.last_best:
            log_where("Save best model", condition=progress_bar)
            self.last_best = metric
            save = True

        return save


class IndexRefreshScheduler:
    # def __init__(self, refresh_schedule: str = "-1", num_processes: int = 1):
    #     self.steps_to_rates = self.parse_refresh_schedule_string(
    #         format_str=refresh_schedule, num_processes=num_processes
    #     )

    def __init__(self, refresh_schedule: str = "-1"):
        self.steps_to_rates = self.parse_refresh_schedule_string(
            format_str=refresh_schedule
        )

    def parse_refresh_schedule_string(self, format_str: str):
        """
        format_str: string that specifies the schedule.
            has the format: startstep-endstep:refreshrate,startstep-endstep:refreshrate
            e.g. format_str="0-100:10,100-1000000:500"
            will refresh the index every 10 steps for the first 100 steps
            and then every 500 steps from step 100 to 1M.

            Syntactic Sugar for a fixed schedule: can just pass in a single number
            e.g. format_str="100" will refresh the index every 100 steps

            -1 to never refresh
        """

        parsed = []
        if format_str == "-1":
            parsed = [(0, 2**32, 2**32)]
        elif format_str.isdigit():
            rate = int(format_str)
            parsed = [(0, 2**32, rate)]
        else:
            for piece in format_str.split(","):
                startend, rate = piece.split(":")
                start, end = startend.split("-")
                rate = int(rate)
                parsed.append((int(start), int(end), rate))
        return parsed

    def is_time_to_refresh(self, step: int) -> bool:
        if step > 0:
            # if retriever is not trained only refresh at step 0
            for st, en, rate in self.steps_to_rates:
                if st <= step < en:
                    steps_since_refresh_schedule_change = step - st
                    return (steps_since_refresh_schedule_change % rate) == 0
        return False


def get_num_gradient_updates(
    train_dl: torch.utils.data.DataLoader,
    max_epochs: int,
    gradient_accumulation_steps: int,
):
    return (len(train_dl) * max_epochs) // gradient_accumulation_steps


def log_where(msg: str, *args, level="info", condition: bool = True):
    "Log only on main process"
    if condition:
        getattr(logger, level)(msg, *args)


def batch_to_device(batch: dict[str, Any], device: torch.device):
    """Place to GPU only necessary stuff."""
    batch = {
        k: (
            torch.as_tensor(v, device=device)
            if any(x in k for x in ["input_ids", "attention_mask", "label"])
            else v
        )
        for k, v in batch.items()
    }
    return batch


def get_parameters_groups(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.0,
    no_decay: list[str] | None = None,
) -> dict:
    no_decay = no_decay if no_decay is not None else []
    assert isinstance(no_decay, list), "`no_decay` must be `list[str]`"
    assert all(isinstance(x, str) for x in no_decay), "`no_decay` must be `list[str]`"

    return [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]


def get_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.0,
    amsgrad: bool = False,
) -> torch.optim.Optimizer:
    """
    Instantiate optimizer
    """

    no_decay = ["bias", "LayerNorm.weight"]
    params = get_parameters_groups(
        model=model, lr=lr, weight_decay=weight_decay, no_decay=no_decay
    )
    optimizer = torch.optim.AdamW(
        params=params,
        lr=lr,
        amsgrad=amsgrad,
    )

    return optimizer


def get_lr_scheduler(
    lr_schedule: str,
    optimizer: torch.optim.Optimizer,
    warmup_steps: int | str = "0",
    num_training_steps: int | None = None,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Learning rate scheduler
    """

    try:
        num_warmup_steps = int(warmup_steps)
    except ValueError:
        assert num_training_steps is not None, (
            "Need `num_training_steps` to compute "
            "`warmup_steps` relative to number of steps"
        )
        relative_warmup_steps = float(warmup_steps)
        num_warmup_steps = int(num_training_steps * relative_warmup_steps)

    scheduler = get_scheduler(
        lr_schedule,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return scheduler


def get_gradients_norm(
    model: torch.nn.Module, norm_type: float = 2, skip_bias: bool = True
) -> dict[str, torch.Tensor]:
    """
    Compute norm of the gradients of each model parameters (except biases)
    """

    grads_norm: dict = {}
    all_norms = []

    for name, param in model.named_parameters():
        if skip_bias and ("bias" in name or param.grad is None):
            continue
        param_grad_norm = round(float(param.grad.data.norm(norm_type)), 4)  # type: ignore
        grad_name = f"gn/{name}"
        grads_norm[grad_name] = param_grad_norm
        all_norms.append(param_grad_norm)

    grad_norm_global = round(float(torch.tensor(all_norms).norm(norm_type)), 4)
    grads_norm["gn/ggn"] = grad_norm_global

    return grads_norm
