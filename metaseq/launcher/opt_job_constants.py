# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum


@dataclass
class Size:
    n_layers: int
    emb_size: int
    n_heads: int
    d_head: int
    batch_size: int
    lr: float
    model_parallel: int

    @property
    def ffn_size(self):
        return 4 * self.emb_size


# from appendix b of https://arxiv.org/pdf/2005.14165.pdf
# see table 2.1 in https://arxiv.org/pdf/2005.14165.pdf

# assert all sizes make sense
TOTAL_TRAIN_TOKENS = 300e9
TOTAL_WARMUP_TOKENS = 375e6
M = 1024 * 1024  # 1 million
MODEL_SIZES = {
    "8m": Size(4, 128, 2, 64, int(0.5 * M), 1.0e-3, 2),  # tiny
    "125m": Size(12, 768, 12, 64, int(0.5 * M), 6.0e-4, 2),  # small
    "350m": Size(24, 1024, 16, 64, int(0.5 * M), 3.0e-4, 2),  # medium
    "760m": Size(24, 1536, 16, 96, int(0.5 * M), 2.5e-4, 2),  # large
    "1.3b": Size(24, 2048, 32, 64, int(1.0 * M), 2.0e-4, 2),  # xl
    "2.7b": Size(32, 2560, 32, 80, int(1.0 * M), 1.6e-4, 4),
    "6.7b": Size(32, 4096, 32, 128, int(2.0 * M), 1.2e-4, 2),
    "13b": Size(40, 5120, 40, 128, int(4.0 * M), 1.0e-4, 2),
    "30b": Size(48, 7168, 56, 128, int(4.0 * M), 1.0e-4, 2),
    "66b": Size(64, 9216, 72, 128, int(2.0 * M), 8e-5, 8),
    "175b": Size(96, 12288, 96, 128, int(0.25 * M), 3e-5, 8),
}

# from appendix b of https://arxiv.org/pdf/2005.14165.pdf
# see table 2.1 in https://arxiv.org/pdf/2005.14165.pdf

for name, size in MODEL_SIZES.items():
    assert size.n_heads * size.d_head == size.emb_size, name


class ComputeEnvs(Enum):
    AWS = "aws"
    AZURE = "azure"
    FAIR = "fair"


DATA_LOCATIONS = {
    ComputeEnvs.AZURE: "/data/opt",
}

VALID_SUBSETS = [
    "BookCorpusFair",
    "CommonCrawl",
    "DM_Mathematics",
    "Gutenberg_PG-19",
    "HackerNews",
    "OpenSubtitles",
    "OpenWebText2",
    "USPTO",
    "Wikipedia_en",
    "redditflattened",
    "stories",
]
