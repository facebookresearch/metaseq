# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import unittest

import torch
from omegaconf import OmegaConf

from metaseq.optim.fp16_optimizer import FP16Optimizer, MemoryEfficientFP16Optimizer


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestGradientScaling(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([2.0]).cuda().half()
        weight = 3.0
        bias = 5.0
        self.error = 1.0
        self.target = torch.tensor([self.x * weight + bias + self.error]).cuda().half()
        self.loss_fn = torch.nn.L1Loss()

        self.model = torch.nn.Linear(1, 1)
        self.model.weight.data = torch.tensor([[weight]])
        self.model.bias.data = torch.tensor([bias])
        self.model.cuda().half()
        self.params = list(self.model.parameters())

        self.cfg_dls = OmegaConf.create(
            {
                "optimization": {
                    "lr": [0.1],
                },
                "optimizer": {
                    "_name": "adam",
                    "lr": [0.1],
                    "adam_betas": "(0.9, 0.999)",
                    "adam_eps": 1e-8,
                    "weight_decay": 0.0,
                },
                "common": {
                    "bf16": False,
                    "fp16_init_scale": 1,
                    "fp16_scale_window": 1,
                    "fp16_scale_tolerance": 1,
                    "threshold_loss_scale": 1,
                    "min_loss_scale": 1e-4,
                },
            }
        )
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def run_iter(self, model, params, optimizer):
        optimizer.zero_grad()
        y = model(self.x)
        loss = self.loss_fn(y, self.target)
        optimizer.backward(loss)
        self.assertEqual(loss, torch.tensor(1.0, device="cuda:0", dtype=torch.float16))

        grad_norm = optimizer.clip_grad_norm(0)
        self.assertAlmostEqual(grad_norm.item(), 2.2361, 4)

        optimizer.step()
        self.assertEqual(
            model.weight,
            torch.tensor(
                [[3.0996]], device="cuda:0", dtype=torch.float16, requires_grad=True
            ),
        )
        self.assertEqual(
            model.bias,
            torch.tensor(
                [5.1016], device="cuda:0", dtype=torch.float16, requires_grad=True
            ),
        )
        self.assertEqual(optimizer.scaler.loss_scale, 2.0)

    def test_mixed_precision(self):
        model = copy.deepcopy(self.model)
        params = list(model.parameters())
        optimizer = FP16Optimizer.build_optimizer(self.cfg_dls, params)

        self.run_iter(model, params, optimizer)
        self.assertTrue(
            all(
                torch.all(
                    fp32_params.eq(
                        torch.tensor(
                            [3.1000, 5.1000], device="cuda:0", requires_grad=True
                        )
                    )
                )
                for fp32_params in optimizer.fp32_params.values()
            )
        )

    def test_memory_efficient(self):
        model = copy.deepcopy(self.model)
        params = list(model.parameters())
        optimizer = MemoryEfficientFP16Optimizer.build_optimizer(self.cfg_dls, params)

        self.run_iter(model, params, optimizer)


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestBF16(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([2.0]).cuda().bfloat16()
        weight = 3.0
        bias = 5.0
        self.error = 1.0
        self.target = (
            torch.tensor([self.x * weight + bias + self.error]).cuda().bfloat16()
        )
        self.loss_fn = torch.nn.L1Loss()

        self.model = torch.nn.Linear(1, 1)
        self.model.weight.data = torch.tensor([[weight]])
        self.model.bias.data = torch.tensor([bias])
        self.model.cuda().bfloat16()
        self.params = list(self.model.parameters())

        self.cfg_dls = OmegaConf.create(
            {
                "distributed_training": {
                    "distributed_world_size": 1,
                },
                "optimization": {
                    "lr": [0.1],
                    "update_freq": [1],
                },
                "optimizer": {
                    "_name": "adam",
                    "lr": [0.1],
                    "adam_betas": "(0.9, 0.999)",
                    "adam_eps": 1e-8,
                    "weight_decay": 0.0,
                },
                "common": {
                    "model_parallel_size": 1,
                    "bf16": True,
                    "fp16": True,
                },
            }
        )
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def run_iter(self, model, params, optimizer):
        optimizer.zero_grad()
        y = model(self.x)
        loss = self.loss_fn(y, self.target)
        optimizer.backward(loss)
        self.assertEqual(loss, torch.tensor(1.0, device="cuda:0", dtype=torch.bfloat16))

        grad_norm = optimizer.clip_grad_norm(0)
        self.assertAlmostEqual(grad_norm.item(), 2.2361, 4)

        optimizer.step()
        self.assertEqual(
            model.weight,
            torch.tensor(
                [[3.0938]], device="cuda:0", dtype=torch.bfloat16, requires_grad=True
            ),
        )
        self.assertEqual(
            model.bias,
            torch.tensor(
                [5.1016], device="cuda:0", dtype=torch.bfloat16, requires_grad=True
            ),
        )
        self.assertIsNone(optimizer.scaler)

    def test_mixed_precision(self):
        model = copy.deepcopy(self.model)
        params = list(model.parameters())
        optimizer = FP16Optimizer.build_optimizer(self.cfg_dls, params)

        self.run_iter(model, params, optimizer)
        self.assertTrue(
            all(
                torch.all(
                    fp32_params.eq(
                        torch.tensor(
                            [3.1000, 5.1000], device="cuda:0", requires_grad=True
                        )
                    )
                )
                for fp32_params in optimizer.fp32_params.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
