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
MAX_BEAM = 16

CHECKPOINT_FOLDER = "/data/gpt-z/cm3/models/ablations/causal_one_image/resharded"
# where to store them on SSD for faster loading
CHECKPOINT_LOCAL = os.path.join("/mnt/scratch/", "2.7B", "resharded", "reshard.pt")

LAUNCH_ARGS = {
    "model_parallel_size": MODEL_PARALLEL,
    "distributed_world_size": TOTAL_WORLD_SIZE,
    "task": "cm3_language_modeling_inference_for_models_trained_with_streaming",
    "spm_path": "/data/gpt-z/cm3/v1.2/tokenizers/V262144_I8192_S512_M512_R1024.json",
    "path": f"{CHECKPOINT_FOLDER}/reshard_no_os/reshard.pt",
    "beam": 1,
    "nbest": 1,
    "bpe": "hf_cm3_unigram",
    "distributed_port": 13000,
    "checkpoint_shard_count": 1,
    "use_sharded_state": True,
    "batch_size": BATCH_SIZE,
    "buffer_size": BATCH_SIZE * MAX_SEQ_LEN,
    "max_tokens": BATCH_SIZE * MAX_SEQ_LEN,
    "image-tokens": 8192,
    "speech-tokens": 512,
    "data": "/tmp",  # required "data" argument.
}
