# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn as nn
from torch.nn import functional as F

from metaseq.modules import Linear, gelu
from metaseq.modules.fused_bias_gelu import has_fused_bias_gelu, fused_bias_gelu


def FeedForwardNetwork(x, fc1, activation_fn, fc2, dropout_module):
    """
    Feedforward network consisting of two linear layers (fc1, fc2), where activation_fn is applied
    between the two layers and dropout_module is applied at the end.
    """
    x_shape = x.shape
    x = x.reshape(-1, x.size(-1))
    # apex fused bias gelu is not yet supported with megatron model parallel
    # TODO [namangoyal]: Find better way to do this
    model_parallel = not isinstance(fc1, nn.Linear) and not isinstance(fc1, Linear)
    if (
        model_parallel
        and activation_fn.fn == gelu
        and has_fused_bias_gelu
        and fc1.bias is not None
    ):
        # here, we do the bias computation outside fc1 and fc2 to take advantage of fused_bias_gelu
        assert fc1.skip_bias_add
        x, bias_fc1 = fc1(x)
        x = fused_bias_gelu(x, bias_fc1)
        x, bias_fc2 = fc2(x)
        x = x + bias_fc2
    elif model_parallel:
        # here, we do the bias computation inside fc1 and fc2 AND gather_output
        x = activation_fn(x, fc1(x)[0], True)
        x, _ = fc2(x)
    elif has_fused_bias_gelu and activation_fn.fn == gelu:
        x = F.linear(x, fc1.weight, None)
        x = fused_bias_gelu(x, fc1.bias)
        x = F.linear(x, fc2.weight, fc2.bias)
    else:
        x = activation_fn(x, fc1(x), False)
        x = fc2(x)
    x = x.view(x_shape)
    x = dropout_module(x)
    return x
