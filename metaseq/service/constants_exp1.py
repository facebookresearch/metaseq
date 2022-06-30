# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

MAX_SEQ_LEN = 2048
BATCH_SIZE = 2048  # silly high bc we dynamically batch by MAX_BATCH_TOKENS
MAX_BATCH_TOKENS = 3072
DEFAULT_PORT = 6010
MODEL_PARALLEL = 2
TOTAL_WORLD_SIZE = 2


CHECKPOINT_FOLDER = "/data/gpt-z/cm3/models/ablations/experiment_1/resharded/"
# where to store them on SSD for faster loading
CHECKPOINT_LOCAL = os.path.join("/mnt/scratch/", "13B", "resharded", "reshard.pt")

LAUNCH_ARGS = [
    f"--model-parallel-size {MODEL_PARALLEL}",
    f"--distributed-world-size {TOTAL_WORLD_SIZE}",
    "--task cm3_language_modeling_inference_for_models_trained_with_streaming",
    f"--spm-path /data/gpt-z/cm3/v1.2/tokenizers/V65536_I8192_S512_M512_R1024.json",
    f"--path {CHECKPOINT_FOLDER}/reshard.pt",
    "--beam 1 --nbest 1",
    "--bpe hf_cm3_unigram",
    # "--final-vocab-size 65536",
    "--distributed-port 13000",
    "--checkpoint-shard-count 1",
    "--use-sharded-state",
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
