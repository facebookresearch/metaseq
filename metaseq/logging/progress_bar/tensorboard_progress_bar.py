import os
import string
import sys
from numbers import Number
import atexit

import torch

from metaseq.logging.meters import AverageMeter
from metaseq.logging.progress_bar.base_progress_bar import BaseProgressBar, logger

_tensorboard_writers = {}
SummaryWriter = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        pass


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
