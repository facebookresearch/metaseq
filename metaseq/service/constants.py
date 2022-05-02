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


try:
    from metaseq_internal.constants import LOCAL_SSD, MODEL_SHARED_FOLDER
except ImportError:
    # MODEL_SHARED_FOLDER should point to a shared drive (e.g. NFS) where the
    # checkpoints from S3 are stored. As an example:
    # MODEL_SHARED_FOLDER = "/example/175B/reshard_no_os"
    # $ ls /example/175B/reshard_no_os
    # dict.txt
    # reshard-model_part-0.pt
    # reshard-model_part-1.pt
    # reshard-model_part-2.pt
    # reshard-model_part-3.pt
    # reshard-model_part-4.pt
    # reshard-model_part-5.pt
    # reshard-model_part-6.pt
    # reshard-model_part-7.pt
    MODEL_SHARED_FOLDER = ""
    # LOCAL_SSD is optional, but it's assuming you have some sort of local
    # hard disk where we can cache a copy of the weights for faster loading.
    LOCAL_SSD = ""
    if not LOCAL_SSD:
        # don't use local cache
        LOCAL_SSD = MODEL_SHARED_FOLDER
    if not MODEL_SHARED_FOLDER:
        raise RuntimeError(
            "You must set the variables in metaseq.service.constants to launch the API."
        )

# tokenizer files
BPE_MERGES = os.path.join(MODEL_SHARED_FOLDER, "gpt2-merges.txt")
BPE_VOCAB = os.path.join(MODEL_SHARED_FOLDER, "gpt2-vocab.json")

# where to find the raw files on nfs
CHECKPOINT_FOLDER = os.path.join(MODEL_SHARED_FOLDER, "175B", "reshard_no_os")
# where to store them on SSD for faster loading
CHECKPOINT_LOCAL = os.path.join(LOCAL_SSD, "175B", "reshard_no_os", "reshard.pt")

LAUNCH_ARGS = [
    f"--model-parallel-size {MODEL_PARALLEL}",
    f"--distributed-world-size {TOTAL_WORLD_SIZE}",
    "--task language_modeling",
    f"--bpe-merges {BPE_MERGES}",
    f"--bpe-vocab {BPE_VOCAB}",
    "--bpe hf_byte_bpe",
    f"--path {CHECKPOINT_LOCAL}",
    "--beam 1 --nbest 1",
    "--distributed-port 13000",
    "--checkpoint-shard-count 1",
    "--use-sharded-state",
    f"--batch-size {BATCH_SIZE}",
    f"--buffer-size {BATCH_SIZE * MAX_SEQ_LEN}",
    f"--max-tokens {BATCH_SIZE * MAX_SEQ_LEN}",
    "/tmp",  # required "data" argument.
]
