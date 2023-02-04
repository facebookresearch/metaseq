# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

MAX_SEQ_LEN = 2048
BATCH_SIZE = 2048  # silly high bc we dynamically batch by MAX_BATCH_TOKENS
MAX_BATCH_TOKENS = 3072
DEFAULT_PORT = 6010
MODEL_PARALLEL = 8
TOTAL_WORLD_SIZE = 8
MAX_BEAM = 16

CHECKPOINT_FOLDER = "/checkpoint/davides/projects/zucchini/models/7b_fmis0_v01/checkpoint_last"

# tokenizer files
BPE_MERGES = os.path.join(CHECKPOINT_FOLDER, "gpt2-merges.txt")
BPE_VOCAB = os.path.join(CHECKPOINT_FOLDER, "gpt2-vocab.json")
HF_TOKENIZER_FILE = "/fsx-mudslide/sviyer/gpt2-unified.json"
MODEL_FILE = os.path.join(CHECKPOINT_FOLDER, "checkpoint.pt")


LAUNCH_ARGS = [
    f"--model-parallel-size {MODEL_PARALLEL}",
    f"--distributed-world-size {TOTAL_WORLD_SIZE}",
    # "--ddp-backend pytorch_ddp",
    # If using FSDP shards, replace ddp-backend and add use-sharded-state
    "--ddp-backend fully_sharded",
    "--use-sharded-state",
    "--task language_modeling",
    # f"--bpe-merges {BPE_MERGES}",
    # f"--bpe-vocab {BPE_VOCAB}",
    f"--hf-tokenizer {HF_TOKENIZER_FILE}",
    "--bpe hf_byte_bpe",
    # f"--merges-filename {BPE_MERGES}",  # TODO(susanz): hack for getting interactive_hosted working on public repo
    # f"--vocab-filename {BPE_VOCAB}",  # TODO(susanz): hack for getting interactive_hosted working on public repo
    f"--path {MODEL_FILE}",
    "--beam 1",
    "--distributed-port 13000",
    "--checkpoint-shard-count 1",
    f"--batch-size {BATCH_SIZE}",
    f"--buffer-size {BATCH_SIZE * MAX_SEQ_LEN}",
    f"--max-tokens {BATCH_SIZE * MAX_SEQ_LEN}",
    "/tmp",  # required "data" argument.
]

# Optional arg overrides which influence model loading during inference
INFERENCE_ARG_OVERRIDES = {}
