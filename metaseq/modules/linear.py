# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter


class Linear(Module):
    """
    Exact same as pytorch nn.Linear but with option to initialize weight and bias directly on GPU
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        initialize_params_on_gpu: bool = False,
        dtype: torch.dtype = None,
    ) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        device = torch.cuda.current_device() if initialize_params_on_gpu else None
        if dtype is None:
            dtype = torch.float
        self.weight = Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
