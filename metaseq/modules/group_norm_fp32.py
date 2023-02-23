# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Group norm done in fp32 (for fp16 training)

Reference:
https://github.com/facebookresearch/fairseq/blob/ad0e69cd99e1ff884041fbd8467d1404bd09847a/fairseq/modules/fp32_group_norm.py
"""

import torch.nn as nn
import torch.nn.functional as F


class GroupNormFp32(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)
