# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum, EnumMeta
from typing import List


class StrEnumMeta(EnumMeta):
    # this is workaround for submitit pickling leading to instance checks failing in hydra for StrEnum, see
    # https://github.com/facebookresearch/hydra/issues/1156
    @classmethod
    def __instancecheck__(cls, other):
        return "enum" in str(type(other))


class StrEnum(Enum, metaclass=StrEnumMeta):
    def __str__(self):
        return self.value

    def __eq__(self, other: str):
        return self.value == other

    def __repr__(self):
        return self.value

    def __hash__(self):
        return hash(str(self))


def ChoiceEnum(choices: List[str]):
    """return the Enum class used to enforce list of choices"""
    return StrEnum("Choices", {k: k for k in choices})


LOG_FORMAT_CHOICES = ChoiceEnum(["json", "none"])
DDP_BACKEND_CHOICES = ChoiceEnum(
    [
        "c10d",  # alias for pytorch_ddp
        "fully_sharded",  # FullyShardedDataParallel from fairscale
        "pytorch_ddp",
    ]
)
DATASET_IMPL_CHOICES = ChoiceEnum(["raw", "lazy", "cached", "mmap", "fasta"])
ZERO_SHARDING_CHOICES = ChoiceEnum(["none", "os"])
CLIP_GRAD_NORM_TYPE_CHOICES = ChoiceEnum(["l2", "inf"])
