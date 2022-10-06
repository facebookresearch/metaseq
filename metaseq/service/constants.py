# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

MAX_SEQ_LEN = 2048
BATCH_SIZE = 2048  # silly high bc we dynamically batch by MAX_BATCH_TOKENS
MAX_BATCH_TOKENS = 3072
DEFAULT_PORT = 6010
MODEL_PARALLEL = 4
TOTAL_WORLD_SIZE = 4


CHECKPOINT_FOLDER = "/data/gpt-z/cm3/models/ablations/causal_one_image/resharded"
# where to store them on SSD for faster loading
CHECKPOINT_LOCAL = os.path.join("/mnt/scratch/", "2.7B", "resharded", "reshard.pt")

LAUNCH_ARGS = [
    f"--model-parallel-size {MODEL_PARALLEL}",
    f"--distributed-world-size {TOTAL_WORLD_SIZE}",
    "--ddp-backend pytorch_ddp",
    "--task language_modeling",
    f"--bpe-merges {BPE_MERGES}",
    f"--bpe-vocab {BPE_VOCAB}",
    "--bpe hf_byte_bpe",
    f"--merges-filename {BPE_MERGES}",  # TODO(susanz): hack for getting interactive_hosted working on public repo
    f"--vocab-filename {BPE_VOCAB}",  # TODO(susanz): hack for getting interactive_hosted working on public repo
    f"--path {MODEL_FILE}",
    "--beam 1 --nbest 1",
    "--bpe hf_cm3_unigram",
    # "--final-vocab-size 65536",
    "--distributed-port 13000",
    "--checkpoint-shard-count 1",
    f"--batch-size {BATCH_SIZE}",
    f"--buffer-size {BATCH_SIZE * MAX_SEQ_LEN}",
    f"--max-tokens {BATCH_SIZE * MAX_SEQ_LEN}",
    "--image-tokens 8192",
    "--speech-tokens 512",
    # "--decoder-layers 40",
    # "--decoder-embed-dim 5120",
    # f"--decoder-ffn-embed-dim {5120*4}",
    # "--decoder-attention-heads 40",
    "/tmp",  # required "data" argument.
]
