# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

logger = logging.getLogger(__name__)


class DynamicLossScaler(object):
    def __init__(
        self,
        init_scale=4.0,
        scale_factor=2.0,
        scale_window=256,
        tolerance=0.0,
        threshold=None,
        min_loss_scale=2**-5,
    ):
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window

        logger.info(
            f"*** SCALE_WINDOW: {self.scale_window}, loss scale: {self.loss_scale} ***"
        )

        self.tolerance = tolerance
        self.threshold = threshold
        self._iter = 0
        self._last_overflow_iter = -1
        self._last_rescale_iter = -1
        self._overflows_since_rescale = 0
        self.min_loss_scale = min_loss_scale

    def scale(self, outputs):
        return self.loss_scale * outputs

    def update(self):
        if (self._iter - self._last_overflow_iter) % self.scale_window == 0:
            self.loss_scale *= self.scale_factor
            self._last_rescale_iter = self._iter
            # When scaling up loss_scale, also scale up the scale_window.
            self.scale_window *= self.scale_factor
        self._iter += 1

    def _decrease_loss_scale(self):
        self.loss_scale /= self.scale_factor
        # also decrease the scale_window (lower loss scale, smaller window)
        self.scale_window = max(int(self.scale_window / self.scale_factor), 1)
        if self.threshold is not None:
            self.loss_scale = max(self.loss_scale, self.threshold)

    def check_overflow(self, grad_norm):
        # detect inf and nan
        if grad_norm == float("inf") or grad_norm != grad_norm:
            # overflow has occurred
            prev_scale = self.loss_scale
            iter_since_rescale = self._iter - self._last_rescale_iter

            self._last_overflow_iter = self._iter
            self._overflows_since_rescale += 1
            pct_overflow = self._overflows_since_rescale / float(iter_since_rescale)
            if pct_overflow >= self.tolerance:
                self._decrease_loss_scale()
                self._last_rescale_iter = self._iter
                self._overflows_since_rescale = 0

            if self.loss_scale < self.min_loss_scale:
                # Don't scale down past min_loss_scale, just continue to skip grad after overflow error is raised.
                self.loss_scale = prev_scale

            self._iter += 1
            raise OverflowError("setting loss scale to: " + str(self.loss_scale))
