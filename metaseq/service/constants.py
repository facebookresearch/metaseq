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

try:
    # internal logic denoting where checkpoints are in meta infrastructure
    from metaseq_internal.constants import CHECKPOINT_FOLDER
except ImportError:
    # CHECKPOINT_FOLDER should point to a shared drive (e.g. NFS) where the
    # checkpoints from S3 are stored. As an example:
    # CHECKPOINT_FOLDER = "/example/175B/reshard_no_os"
    # $ ls /example/175B/reshard_no_os
    # reshard-model_part-0.pt
    # reshard-model_part-1.pt
    # reshard-model_part-2.pt
    # reshard-model_part-3.pt
    # reshard-model_part-4.pt
    # reshard-model_part-5.pt
    # reshard-model_part-6.pt
    # reshard-model_part-7.pt
    CHECKPOINT_FOLDER = "/example/175B/reshard_no_os"

# tokenizer files
BPE_MERGES = os.path.join(CHECKPOINT_FOLDER, "gpt2-merges.txt")
BPE_VOCAB = os.path.join(CHECKPOINT_FOLDER, "gpt2-vocab.json")
MODEL_FILE = os.path.join(CHECKPOINT_FOLDER, "reshard.pt")


LAUNCH_ARGS = {
    "model_parallel_size": MODEL_PARALLEL,
    "distributed_world_size": TOTAL_WORLD_SIZE,
    "task": "language_modeling",
    "bpe_merges": BPE_MERGES,
    "bpe_vocab": BPE_VOCAB,
    "bpe": "hf_byte_bpe",
    "merges_filename": BPE_MERGES,  # TODO(susanz): hack for getting interactive_hosted working on public repo
    "vocab_filename": BPE_VOCAB,  # TODO(susanz): hack for getting interactive_hosted working on public repo
    "path": f"{CHECKPOINT_FOLDER}/reshard_no_os/reshard.pt",
    "beam": 1,
    "nbest": 1,
    "distributed_port": 13000,
    "checkpoint_shard_count": 1,
    "use_sharded_state": True,
    "batch_size": BATCH_SIZE,
    "buffer_size": BATCH_SIZE * MAX_SEQ_LEN,
    "max_tokens": BATCH_SIZE * MAX_SEQ_LEN,
    "data": "/tmp",  # required "data" argument.
}
