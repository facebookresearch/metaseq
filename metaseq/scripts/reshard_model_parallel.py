# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import fire
import logging
import os
import time
from collections import defaultdict
from glob import glob
from pathlib import Path

from typing import List
import torch
from tqdm import tqdm

from metaseq.distributed.stitch_fsdp_ckpt import find_num_parts, reshard_megatron_parts
from metaseq.file_io import load_and_pop_last_optimizer_state

logger = logging.getLogger(__name__)


def _get_model_part_num(filename):
    return int(os.path.basename(filename).replace(".pt", "").split("-")[-1])


def reshard_mp_shards(
    pth_prefix: str,
    new_model_parts: int,
    save_prefix=None,
    new_arch_name=None,
) -> List[str]:

    if pth_prefix.endswith(".pt"):
        logger.info("single file provided, assuming fully consolidated checkpoint")
        pth_prefix = pth_prefix[:-3]
    if save_prefix is None:
        save_prefix = pth_prefix + "_resharded"  # .pt'

    all_ckpt_files = list(glob(f"{pth_prefix}*.pt"))
    if len(all_ckpt_files) > 1:
        for ckpt_file in all_ckpt_files:
            assert "shard" not in os.path.basename(ckpt_file)
            #  "This script should be run only on fsdp consolidated files"
            assert "model_part" in os.path.basename(ckpt_file)

        all_ckpt_files = sorted(all_ckpt_files, key=_get_model_part_num)

    assert all_ckpt_files, f"no paths matched {pth_prefix}*.pt"
    weights = []
    names = []
    t0 = time.time()
    for p in tqdm(all_ckpt_files):
        names.append(Path(p).name)
        ckpt = load_and_pop_last_optimizer_state(p)
        weights.append(ckpt["model"])

    num_parts = find_num_parts(names) if len(all_ckpt_files) > 1 else 1

    model_parts = defaultdict()

    assert len(weights) == num_parts
    for p in range(num_parts):
        model_parts[p] = weights[p]

    resharded_models = reshard_megatron_parts(model_parts, new_model_parts)

    if new_arch_name is not None:
        ckpt["cfg"]["model"]._name = new_arch_name

    def save_checkpoint(weights_to_save, prefix):
        ckpt_resharded = dict(
            model=weights_to_save,
            cfg=ckpt["cfg"],
            extra_state=ckpt["extra_state"],
            optimizer_history=ckpt["optimizer_history"],
            args=ckpt.get("args"),
        )
        save_path = f"{prefix}.pt"
        logger.info(f"Saving to {save_path} ...")
        torch.save(ckpt_resharded, save_path)
        logger.info(f"Done after {time.time()-t0//60} minutes")
        return save_path

    saved_paths = []
    for part_id, part_consolidated_weights in enumerate(resharded_models):
        saved_paths.append(
            save_checkpoint(
                part_consolidated_weights, f"{save_prefix}-model_part-{part_id}"
            )
        )
    return saved_paths


"""
Script to reshard model parallel partitions.
can be used to glue also if --new-model-parts 1

Usage:

python metaseq/scripts/reshard_model_parallel.py checkpoint_last 16  --save-prefix resharded
"""
if __name__ == "__main__":
    fire.Fire(reshard_mp_shards)
