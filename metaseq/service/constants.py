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

CHECKPOINT_FOLDER = "/checkpoint/liliyu/cm3z_saved_models/"
# where to store them on SSD for faster loading
# CHECKPOINT_LOCAL = os.path.join(CHECKPOINT_FOLDER,'cm3_michi_consolidated', 'checkpoint_last_consolidated_inference.pt')
# CHECKPOINT_LOCAL = os.path.join(CHECKPOINT_FOLDER,'cm3_michi_consolidated', 'checkpoint_last_consolidated.pt')
# CHECKPOINT_LOCAL = os.path.join(CHECKPOINT_FOLDER,'cm3_consolidated', 'checkpoint_47_40000_consolidated_inference.pt')
# CHECKPOINT_LOCAL = os.path.join(CHECKPOINT_FOLDER,'cm3_consolidated', 'checkpoint_47_40000_consolidated_inference.pt')
CHECKPOINT_LOCAL = os.path.join(CHECKPOINT_FOLDER,'cm3_consolidated', '47_resharded4' ,'reshard.pt')
SPM_PATH = os.path.join(CHECKPOINT_FOLDER, 'V262144_I8192_S512_M512_R1024.json')

LAUNCH_ARGS = [
    f"--model-parallel-size {MODEL_PARALLEL}",
    f"--distributed-world-size {TOTAL_WORLD_SIZE}",
    "--task cm3_language_modeling_inference_for_models_trained_with_streaming",
    f"--spm-path {SPM_PATH}",
    f"--path {CHECKPOINT_LOCAL}",
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
