# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import json
import syslog

from metaseq.logging.progress_bar.base_progress_bar import (
    BaseProgressBar,
    format_stat,
)


class SyslogProgressBar(BaseProgressBar):
    """Log to syslog."""

    def __init__(self, wrapped_bar: BaseProgressBar, syslog_tag: str):
        self.wrapped_bar = wrapped_bar
        self.syslog_tag = syslog_tag

    def __len__(self):
        return len(self.wrapped_bar)

    def __iter__(self):
        return iter(self.wrapped_bar)

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats."""
        self._write_to_syslog(stats, tag, step)
        self.wrapped_bar.log(stats, tag=tag, step=step)

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        self._write_to_syslog(stats, tag, step)
        self.wrapped_bar.print(stats, tag=tag, step=step)

    def update_config(self, config):
        """Log latest configuration."""
        self.wrapped_bar.update_config(config)
    
    def _write_to_syslog(self, stats, tag=None, step=None):
        formatted_stats = self._format_stats(stats, epoch=self.epoch)
        msg = f"{self.syslog_tag}: {json.dumps(formatted_stats)}"
        syslog.syslog(msg)
    
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
