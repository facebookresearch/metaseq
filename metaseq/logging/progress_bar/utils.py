import logging
from typing import Optional

from metaseq.logging.progress_bar.base_progress_bar import logger
from metaseq.logging.progress_bar.json_progress_bar import JsonProgressBar
from metaseq.logging.progress_bar.tensorboard_progress_bar import TensorboardProgressBarWrapper
from metaseq.logging.progress_bar.wandb_progress_bar import WandBProgressBarWrapper


def get_progress_bar(
    iterator,
    log_format: str = "json",
    log_interval: int = 100,
    log_file: Optional[str] = None,
    epoch: Optional[int] = None,
    prefix: Optional[str] = None,
    tensorboard_logdir: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
):
    if log_file is not None:
        handler = logging.FileHandler(filename=log_file)
        logger.addHandler(handler)

    if log_format == "json":
        bar = JsonProgressBar(iterator, epoch, prefix, log_interval)
    else:
        raise ValueError("Unknown log format: {}".format(log_format))

    if tensorboard_logdir:
        bar = TensorboardProgressBarWrapper(bar, tensorboard_logdir)

    if wandb_project:
        bar = WandBProgressBarWrapper(bar, wandb_project, run_name=wandb_run_name)

    return bar


def get_precise_epoch(epoch: Optional[int], count: int, iterator_size: int) -> float:
    return (
        epoch - 1 + (count + 1) / float(iterator_size)
        if epoch is not None and iterator_size > 0
        else None
    )
