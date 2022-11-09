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
MAX_BEAM = 32

CHECKPOINT_FOLDER = "/shared/home/liliyu/data/text-speech-models/en_speech_text_c4_2_7B_v3/consolidated_mp2"
# where to store them on SSD for faster loading
CHECKPOINT_LOCAL = os.path.join("/mnt/scratch/", "13B", "resharded", "reshard.pt")

LAUNCH_ARGS = [
    f"--model-parallel-size {MODEL_PARALLEL}",
    f"--distributed-world-size {TOTAL_WORLD_SIZE}",
    "--ddp-backend pytorch_ddp",
    "--task cm3_language_modeling_inference_for_models_trained_with_streaming",
    f"--spm-path /shared/home/liliyu/data/text-speech-models/V262144_I8192_S2000_M512_R1024_1M_V2.json",
    f"--path {CHECKPOINT_FOLDER}/consolidated.pt",
    "--beam 1 --nbest 1",
    "--bpe hf_cm3_unigram",
    "--distributed-port 13000",
    "--checkpoint-shard-count 1",
    f"--batch-size {BATCH_SIZE}",
    f"--buffer-size {BATCH_SIZE * MAX_SEQ_LEN}",
    f"--max-tokens {BATCH_SIZE * MAX_SEQ_LEN}",
    "--image-tokens 8192",
    "--speech-tokens 512",
    "/tmp",  # required "data" argument.
]
