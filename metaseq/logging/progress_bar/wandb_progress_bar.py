from numbers import Number

from metaseq.logging.meters import AverageMeter
from metaseq.logging.progress_bar.base_progress_bar import (
    BaseProgressBar,
    logger,
    get_precise_epoch,
)

wandb = None

try:
    import wandb
except ImportError:
    pass


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
