# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Wrapper around various loggers and progress bars (e.g., json).
"""

import logging
from collections import OrderedDict
from numbers import Number

import torch

from metaseq.logging.meters import AverageMeter, TimeMeter, StopwatchMeter

logger = logging.getLogger(__name__)


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
