# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import logging
import unittest

import torch
from omegaconf import OmegaConf

from metaseq.optim.adam import MetaseqAdam
from metaseq.optim.lr_scheduler.polynomial_decay_schedule import (
    PolynomialDecayLRSchedule,
)


class TestPolynomialLRScheduler(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([2.0])
        weight = 3.0
        bias = 5.0
        error = 1.0
        self.target = torch.tensor([self.x * weight + bias + error])
        self.loss_fn = torch.nn.L1Loss()

        self.model = torch.nn.Linear(1, 1)
        self.model.weight.data = torch.tensor([[weight]])
        self.model.bias.data = torch.tensor([bias])
        self.params = list(self.model.parameters())
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _get_adam(self, starting_lr):
        return MetaseqAdam(
            cfg=OmegaConf.create(
                vars(
                    argparse.Namespace(
                        adam_betas="(0.9, 0.999)",
                        adam_eps=1e-8,
                        weight_decay=0.0,
                        lr=[starting_lr],
                    )
                )
            ),
            params=self.params,
        )

    @staticmethod
    def _get_polynomial_lr_schedule(
        warmup_updates,
        power,
        total_updates,
        starting_lr,
        end_lr,
        zero_lr_warmup_steps,
        optimizer,
    ):
        return PolynomialDecayLRSchedule(
            cfg=OmegaConf.create(
                vars(
                    argparse.Namespace(
                        warmup_updates=warmup_updates,
                        end_learning_rate=end_lr,
                        power=power,
                        total_num_update=total_updates,
                        lr=[starting_lr],
                        zero_lr_warmup_steps=zero_lr_warmup_steps,
                    )
                )
            ),
            optimizer=optimizer,
        )

    def test_polynomial_decay_no_adam_warmup(self):
        starting_lr = 0.1
        total_updates = 50
        warmup_updates = 20
        adam_warmup = 0
        power = 1
        adam_optim = self._get_adam(starting_lr)
        # Test setting end_lr, adam_warmup = 0
        end_lr = starting_lr * 0.1
        lr_sched = self._get_polynomial_lr_schedule(
            warmup_updates,
            power,
            total_updates,
            starting_lr,
            end_lr,
            adam_warmup,
            adam_optim,
        )
        # Init warmup period, halfway mark
        self.assertAlmostEqual(
            lr_sched.step_update(warmup_updates // 2), starting_lr * 0.5
        )
        # Done warming up
        self.assertAlmostEqual(lr_sched.step_update(warmup_updates), starting_lr)
        # Linear decay, halfway mark
        halfway = warmup_updates + (total_updates - warmup_updates) // 2
        self.assertAlmostEqual(
            lr_sched.step_update(halfway), end_lr + (starting_lr - end_lr) * 0.5
        )
        # End of decay
        self.assertAlmostEqual(lr_sched.step_update(total_updates), end_lr)

        # Test power == 2
        power = 2
        end_lr = 0
        lr_sched = self._get_polynomial_lr_schedule(
            warmup_updates,
            power,
            total_updates,
            starting_lr,
            end_lr,
            adam_warmup,
            adam_optim,
        )
        # Init warmup period, halfway mark
        self.assertAlmostEqual(
            lr_sched.step_update(warmup_updates // 2), starting_lr * 0.5
        )
        # Done warming up
        self.assertAlmostEqual(lr_sched.step_update(warmup_updates), starting_lr)
        # Polynomial power == 2 decay, halfway mark
        self.assertAlmostEqual(
            lr_sched.step_update(halfway), end_lr + (starting_lr - end_lr) * 0.5**2
        )

    def test_polynomial_decay_with_adam_warmup(self):
        starting_lr = 0.1
        total_updates = 50
        warmup_updates = 20
        adam_warmup = 10
        power = 1
        adam_optim = self._get_adam(starting_lr)
        end_lr = starting_lr * 0.1
        lr_sched = self._get_polynomial_lr_schedule(
            warmup_updates,
            power,
            total_updates,
            starting_lr,
            end_lr,
            adam_warmup,
            adam_optim,
        )
        # Init warmup period, during adam warmup
        self.assertEqual(lr_sched.step_update(adam_warmup // 2), 0)
        # Init warmup period, past adam warmup
        self.assertAlmostEqual(
            lr_sched.step_update(warmup_updates // 2 + adam_warmup), starting_lr * 0.5
        )
        # Done warming up
        total_warmup = adam_warmup + warmup_updates
        self.assertAlmostEqual(lr_sched.step_update(total_warmup), starting_lr)
        # Linear decay, halfway mark
        halfway = total_warmup + (total_updates - total_warmup) // 2
        self.assertAlmostEqual(
            lr_sched.step_update(halfway), end_lr + (starting_lr - end_lr) * 0.5
        )
        # End of decay
        self.assertAlmostEqual(lr_sched.step_update(total_updates), end_lr)


if __name__ == "__main__":
    unittest.main()
