# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Wrapper around various loggers and progress bars (e.g., json).
"""

import logging
from collections import OrderedDict
from contextlib import contextmanager
from numbers import Number
from typing import Optional

import torch

from metaseq.logging.meters import AverageMeter, StopwatchMeter, TimeMeter
from metaseq.logging.progress_bar.json_progress_bar import JsonProgressBar
from metaseq.logging.progress_bar.tensorboard_progress_bar import (
    TensorboardProgressBarWrapper,
)
from metaseq.logging.progress_bar.wandb_progress_bar import WandBProgressBarWrapper

logger = logging.getLogger(__name__)


def progress_bar(
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


def format_stat(stat):
    if isinstance(stat, Number):
        stat = "{:g}".format(stat)
    elif isinstance(stat, AverageMeter):
        stat = "{:.3f}".format(stat.avg)
    elif isinstance(stat, TimeMeter):
        stat = "{:g}".format(round(stat.avg))
    elif isinstance(stat, StopwatchMeter):
        stat = "{:g}".format(round(stat.sum))
    elif torch.is_tensor(stat):
        stat = stat.tolist()
        if isinstance(stat, float):
            stat = f'{float(f"{stat:.3g}"):g}'  # 3 significant figures
    return stat


class BaseProgressBar(object):
    """Abstract class for progress bars."""

    def __init__(self, iterable, epoch=None, prefix=None):
        self.iterable = iterable
        self.n = getattr(iterable, "n", 0)
        self.epoch = epoch
        self.prefix = ""
        if epoch is not None:
            self.prefix += "epoch {:03d}".format(epoch)
        if prefix is not None:
            self.prefix += (" | " if self.prefix != "" else "") + prefix

    def __len__(self):
        return len(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        raise NotImplementedError

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval."""
        raise NotImplementedError

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        raise NotImplementedError

    def update_config(self, config):
        """Log latest configuration."""
        pass

    def _str_commas(self, stats):
        return ", ".join(key + "=" + stats[key].strip() for key in stats.keys())

    def _str_pipes(self, stats):
        return " | ".join(key + " " + stats[key].strip() for key in stats.keys())

    def _format_stats(self, stats):
        postfix = OrderedDict(stats)
        # Preprocess stats according to datatype
        for key in postfix.keys():
            postfix[key] = str(format_stat(postfix[key]))
        return postfix


@contextmanager
def rename_logger(logger, new_name):
    old_name = logger.name
    if new_name is not None:
        logger.name = new_name
    yield logger
    logger.name = old_name


def get_precise_epoch(epoch: Optional[int], count: int, iterator_size: int) -> float:
    return (
        epoch - 1 + (count + 1) / float(iterator_size)
        if epoch is not None and iterator_size > 0
        else None
    )
