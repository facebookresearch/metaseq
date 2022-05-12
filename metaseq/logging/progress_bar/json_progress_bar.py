import json
from collections import OrderedDict

from metaseq.logging.progress_bar.base_progress_bar import (
    BaseProgressBar,
    get_precise_epoch,
    rename_logger,
    logger,
    format_stat,
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
