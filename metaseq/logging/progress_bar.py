# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Wrapper around various loggers and progress bars (e.g., json).
"""

import atexit
import json
import logging
import os
import string
import sys
from collections import OrderedDict
from contextlib import contextmanager
from numbers import Number
from typing import Optional

import torch

from .meters import AverageMeter, StopwatchMeter, TimeMeter

logger = logging.getLogger(__name__)


def progress_bar(
    iterator,
    log_format: Optional[str] = None,
    log_interval: int = 100,
    log_file: Optional[str] = None,
    epoch: Optional[int] = None,
    prefix: Optional[str] = None,
    tensorboard_logdir: Optional[str] = None,
    default_log_format: str = "json",
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
):
    if log_format is None:
        log_format = default_log_format

    if log_file is not None:
        handler = logging.FileHandler(filename=log_file)
        logger.addHandler(handler)

    if log_format == "json":
        bar = JsonProgressBar(iterator, epoch, prefix, log_interval)
    elif log_format == "none":
        bar = NoopProgressBar(iterator, epoch, prefix)
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


class JsonProgressBar(BaseProgressBar):
    """Log output in JSON format."""

    def __init__(self, iterable, epoch=None, prefix=None, log_interval=1000):
        super().__init__(iterable, epoch, prefix)
        self.log_interval = log_interval
        self.i = None
        self.size = None

    def __iter__(self):
        self.size = len(self.iterable)
        for i, obj in enumerate(self.iterable, start=self.n):
            self.i = i
            yield obj

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval."""
        step = step or self.i or 0
        if step > 0 and self.log_interval is not None and step % self.log_interval == 0:
            update = get_precise_epoch(self.epoch, self.i, self.size)
            stats = self._format_stats(stats, epoch=self.epoch, update=update)
            with rename_logger(logger, tag):
                logger.info(json.dumps(stats))

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        self.stats = stats
        if tag is not None:
            self.stats = OrderedDict(
                [(tag + "_" + k, v) for k, v in self.stats.items()]
            )
        stats = self._format_stats(self.stats, epoch=self.epoch)
        with rename_logger(logger, tag):
            logger.info(json.dumps(stats))

    def _format_stats(self, stats, epoch=None, update=None):
        postfix = OrderedDict()
        if epoch is not None:
            postfix["epoch"] = epoch
        if update is not None:
            postfix["update"] = round(update, 3)
        # Preprocess stats according to datatype
        for key in stats.keys():
            postfix[key] = format_stat(stats[key])
        return postfix


class NoopProgressBar(BaseProgressBar):
    """No logging."""

    def __init__(self, iterable, epoch=None, prefix=None):
        super().__init__(iterable, epoch, prefix)

    def __iter__(self):
        for obj in self.iterable:
            yield obj

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval."""
        pass

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        pass


try:
    _tensorboard_writers = {}
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        SummaryWriter = None


def _close_writers():
    for w in _tensorboard_writers.values():
        w.close()


atexit.register(_close_writers)


class TensorboardProgressBarWrapper(BaseProgressBar):
    """Log to tensorboard."""

    def __init__(self, wrapped_bar, tensorboard_logdir):
        self.wrapped_bar = wrapped_bar
        self.tensorboard_logdir = tensorboard_logdir

        if SummaryWriter is None:
            logger.warning(
                "tensorboard not found, please install with: pip install tensorboard"
            )

    def _writer(self, key):
        if SummaryWriter is None:
            return None
        _writers = _tensorboard_writers
        if key not in _writers:
            # tensorboard doesn't play well when we clobber it with reruns
            # find an acceptable suffix
            for suffix in [""] + list(string.ascii_uppercase):
                logdir = os.path.join(self.tensorboard_logdir + suffix, key)
                if not os.path.exists(logdir):
                    logger.info(f"Setting tensorboard directory to {logdir}")
                    break
            else:
                # wow we have cycled through a lot of these
                raise RuntimeError(
                    f"Tensorboard logdir {logdir} already exists. "
                    "Ran out of possible suffixes."
                )
            _writers[key] = SummaryWriter(logdir)
            _writers[key].add_text("sys.argv", " ".join(sys.argv))
        return _writers[key]

    def __len__(self):
        return len(self.wrapped_bar)

    def __iter__(self):
        return iter(self.wrapped_bar)

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats to tensorboard."""
        self._log_to_tensorboard(stats, tag, step)
        self.wrapped_bar.log(stats, tag=tag, step=step)

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        self._log_to_tensorboard(stats, tag, step)
        self.wrapped_bar.print(stats, tag=tag, step=step)

    def update_config(self, config):
        """Log latest configuration."""
        # TODO add hparams to Tensorboard
        self.wrapped_bar.update_config(config)

    def _log_to_tensorboard(self, stats, tag=None, step=None):
        writer = self._writer(tag or "")
        if writer is None:
            return
        if step is None:
            step = stats["num_updates"]
        for key in stats.keys() - {"num_updates"}:
            if isinstance(stats[key], AverageMeter):
                writer.add_scalar(key, stats[key].val, step)
            elif isinstance(stats[key], Number):
                writer.add_scalar(key, stats[key], step)
            elif torch.is_tensor(stats[key]) and stats[key].numel() == 1:
                writer.add_scalar(key, stats[key].item(), step)
        writer.flush()


try:
    import wandb
except ImportError:
    wandb = None


class WandBProgressBarWrapper(BaseProgressBar):
    """Log to Weights & Biases."""

    def __init__(self, wrapped_bar, wandb_project, run_name=None):
        super().__init__(
            wrapped_bar, epoch=wrapped_bar.epoch, prefix=wrapped_bar.prefix
        )
        self.wrapped_bar = wrapped_bar
        if wandb is None:
            logger.warning("wandb not found, pip install wandb")
            return

        # reinit=False to ensure if wandb.init() is called multiple times
        # within one process it still references the same run
        wandb.init(project=wandb_project, reinit=False, name=run_name)

    def __len__(self):
        return len(self.wrapped_bar)

    def __iter__(self):
        self.size = len(self.wrapped_bar)
        for i, obj in enumerate(self.wrapped_bar, start=self.n):
            self.i = i
            yield obj

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats to tensorboard."""
        self._log_to_wandb(stats, tag, step)
        self.wrapped_bar.log(stats, tag=tag, step=step)

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        self._log_to_wandb(stats, tag, step)
        self.wrapped_bar.print(stats, tag=tag, step=step)

    def update_config(self, config):
        """Log latest configuration."""
        if wandb is not None:
            wandb.config.update(config, allow_val_change=True)
        self.wrapped_bar.update_config(config)

    def _log_to_wandb(self, stats, tag=None, step=None):
        if wandb is None:
            return
        if step is None:
            step = stats["num_updates"]

        prefix = "" if tag is None else tag + "/"

        epoch = get_precise_epoch(self.epoch, self.i, self.size)
        wandb.log({prefix + "epoch": epoch}, step=step)

        for key in stats.keys() - {"num_updates"}:
            if isinstance(stats[key], AverageMeter):
                wandb.log({prefix + key: stats[key].val}, step=step)
            elif isinstance(stats[key], Number):
                wandb.log({prefix + key: stats[key]}, step=step)
