# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

MAX_SEQ_LEN = int(os.getenv("METASEQ_OPT_MAX_SEQ_LEN", 2048))
BATCH_SIZE = int(
    os.getenv("METASEQ_OPT_BATCH_SIZE", 2048)
)  # silly high bc we dynamically batch by MAX_BATCH_TOKENS
MAX_BATCH_TOKENS = int(os.getenv("METASEQ_OPT_MAX_BATCH_TOKENS", 3072))
DEFAULT_PORT = int(os.getenv("METASEQ_OPT_DEFAULT_PORT", 6010))
MODEL_PARALLEL = int(os.getenv("METASEQ_OPT_MODEL_PARALLEL", 8))
TOTAL_WORLD_SIZE = int(os.getenv("METASEQ_OPT_TOTAL_WORLD_SIZE", 8))
DISTRIBUTED_PORT = int(os.getenv("METASEQ_OPT_DISTRIBUTED_PORT", 13000))

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
    CHECKPOINT_FOLDER = os.getenv(
        "METASEQ_OPT_CHECKPOINT_FOLDER", "/example/175B/reshard_no_os"
    )


# tokenizer files
BPE_FOLDER = os.getenv("METASEQ_OPT_BPE_FOLDER", CHECKPOINT_FOLDER)
BPE_MERGES = os.path.join(BPE_FOLDER, "gpt2-merges.txt")
BPE_VOCAB = os.path.join(BPE_FOLDER, "gpt2-vocab.json")
MODEL_FILE = os.path.join(CHECKPOINT_FOLDER, "reshard.pt")


LAUNCH_ARGS = [
    f"--model-parallel-size {MODEL_PARALLEL}",
    f"--distributed-world-size {TOTAL_WORLD_SIZE}",
    "--task language_modeling",
    f"--bpe-merges {BPE_MERGES}",
    f"--bpe-vocab {BPE_VOCAB}",
    "--bpe hf_byte_bpe",
    f"--merges-filename {BPE_MERGES}",  # TODO(susanz): hack for getting interactive_hosted working on public repo
    f"--vocab-filename {BPE_VOCAB}",  # TODO(susanz): hack for getting interactive_hosted working on public repo
    f"--path {CHECKPOINT_FOLDER}/reshard.pt",
    "--beam 1 --nbest 1",
    "--distributed-port {DISTRIBUTED_PORT}",
    "--checkpoint-shard-count 1",
    "--use-sharded-state",
    f"--batch-size {BATCH_SIZE}",
    f"--buffer-size {BATCH_SIZE * MAX_SEQ_LEN}",
    f"--max-tokens {BATCH_SIZE * MAX_SEQ_LEN}",
    "/tmp",  # required "data" argument.
]
