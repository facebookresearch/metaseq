# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, List


@torch.jit.script
def relu_squared(x: torch.Tensor):
    return F.relu(x).pow(2)


@torch.jit.script
def gelu(x):
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


@torch.jit.script
def gelu_back(g, x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    return ff * g


@torch.jit.script
def relu(x):
    return F.relu(x)


@torch.jit.script
def relu_back(g, x):
    return g.masked_fill_(x <= 0, 0)


@torch.jit.script
def swiglu(x: torch.Tensor, gate: torch.Tensor):
    return F.silu(x) * gate


@torch.jit.script
def geglu(x: torch.Tensor, gate: torch.Tensor):
    return gelu(x) * gate


def get_available_activation_fns() -> List:
    return [
        "relu",
        "relu_squared",
        "gelu",
        "tanh",
        "linear",
        "swiglu",
        "geglu",
    ]


class ActivationFn(nn.Module):
    def __init__(self, name, fc1_builder, embed_dim, ffn_dim, **fc1_kwargs):
        super().__init__()
        self.fn = self.__get_fn(name)
        self.gate = None
        if self.fn in self.__get_gated_fns():
            self.gate = fc1_builder(embed_dim, ffn_dim, **fc1_kwargs)

    def forward(self, fc1_in, fc1_out, model_parallel: bool):
        if self.gate is not None:
            if model_parallel:
                g, _ = self.gate(fc1_in)
            else:
                g = self.gate(fc1_in)
            return self.fn(fc1_out, g)
        return self.fn(fc1_out)

    def __get_fn(self, name: str) -> Callable:
        """Returns the activation function corresponding to the arg passed in the run"""

        if name == "relu":
            return F.relu
        elif name == "relu_squared":
            return relu_squared
        elif name == "gelu":
            return gelu
        elif name == "tanh":
            return torch.tanh
        elif name == "linear":
            return lambda x: x
        elif name == "swiglu":
            return swiglu
        elif name == "geglu":
            return geglu
        else:
            raise RuntimeError("--activation-fn {} not supported".format(name))

    def __get_gated_fns(self) -> List:
        return [
            geglu,
            swiglu,
        ]
