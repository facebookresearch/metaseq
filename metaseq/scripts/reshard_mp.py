# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from fire import Fire
from tqdm import tqdm

from metaseq.checkpoint_utils import (
    get_paths_to_load,
    _merge_flat_fsdp_shards,
    OPT_KEY,
    is_singleton_tensor,
)
from metaseq.file_io import torch_load_cpu

logging.basicConfig(
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("mp_reshard")


def reshard_all_parts(
    save_prefix, save_dir, mpart=16, target_ddp_size=512, no_pad=False
):
    for i in range(mpart):
        try:
            reshard_mp(
                save_prefix,
                save_dir,
                part=i,
                target_ddp_size=target_ddp_size,
                no_pad=no_pad,
            )
        except FileNotFoundError:
            logger.info(f"Resharded {i} model parts")
            return


def _save_shards_to_disk(
    local_state_dicts,
    dummy_model_state,
    state,
    save_dir,
    middle,
    local_opt_states=None,
    target_ddp_size=512,
):
    Path(save_dir).mkdir(exist_ok=True)
    for i, local_state_dict in tqdm(
        enumerate(local_state_dicts),
        desc=f"Saving to {save_dir}/reshard-{middle}-shard[i].pt",
    ):
        if target_ddp_size == 1:
            save_path = f"{save_dir}/reshard-{middle}.pt"
        else:
            save_path = f"{save_dir}/reshard-{middle}-shard{i}.pt"
        local_state_dict.update(dummy_model_state)
        full_state = {"model": local_state_dict}

        full_state.update(state)
        if local_opt_states is not None:
            full_state[OPT_KEY] = local_opt_states[i]
        torch.save(full_state, save_path)


def reshard_mp(
    save_prefix,
    save_dir,
    part=0,
    target_ddp_size=512,
    no_pad=False,
    drop_optimizer_state=False,
):
    middle = f"model_part-{part}"
    do_pad = not no_pad
    if not Path(f"{save_prefix}-{middle}-shard0.pt").exists():
        raise FileNotFoundError(f"{save_prefix}-{middle}-shard0.pt")
    paths_to_load = get_paths_to_load(
        f"{save_prefix}-{middle}-shard0.pt", suffix="-shard"
    )
    logger.info(
        f"Loading {len(paths_to_load)} paths for MP part{part}. Will shard into {target_ddp_size} files."
    )
    state = _merge_flat_fsdp_shards([torch_load_cpu(f) for f in paths_to_load])
    model_state = state.pop("model")

    dummy_model_state = {}  # for decoder.version and other useless keys

    local_state_dicts: List[dict] = [{} for _ in range(target_ddp_size)]
    for k, v in model_state.items():
        if "flat_param" not in k:
            dummy_model_state[k] = v
            continue
        chunks = list(torch.flatten(v).chunk(target_ddp_size))
        assert len(chunks) == target_ddp_size
        num_to_pad = chunks[0].numel() - chunks[-1].numel()

        # Same logic as https://tinyurl.com/fairscale but there is no padding allowed!
        # Notes on padding: https://github.com/fairinternal/fairseq-py/issues/2894
        for rank, param in enumerate(chunks):
            # This clone is essential. Not sure why.
            local_state_dicts[rank][k] = param.clone()
        if num_to_pad > 0 and do_pad:
            local_state_dicts[-1][k] = F.pad(local_state_dicts[-1][k], [0, num_to_pad])
            logger.info(f"Padding {k} with {num_to_pad} zeros")
    state.pop("shard_metadata")  # TODO: update shard metadata to be accurate
    # DO OPT STATE HERE
    if drop_optimizer_state and OPT_KEY in state:
        state.pop(OPT_KEY)

    if OPT_KEY not in state:
        _save_shards_to_disk(
            local_state_dicts, dummy_model_state, state, save_dir, middle
        )
        return

    merged_opt_state = state.pop(OPT_KEY)
    local_opt_states: List[dict] = [{"state": {}} for _ in range(target_ddp_size)]
    for k in merged_opt_state["state"].keys():
        # 0,1,2,3... if each layer wrapped, else 0
        for k2 in merged_opt_state["state"][k].keys():
            for i in range(target_ddp_size):
                if k not in local_opt_states[i]["state"]:
                    local_opt_states[i]["state"][k] = {}
            catted = merged_opt_state["state"][k][k2]
            if not torch.is_tensor(catted) or is_singleton_tensor(catted):
                for i in range(target_ddp_size):

                    local_opt_states[i]["state"][k][k2] = catted
            else:
                chunks = list(torch.flatten(catted).chunk(target_ddp_size))
                assert len(chunks) == target_ddp_size
                num_to_pad = chunks[0].numel() - chunks[-1].numel()
                for rank, param in enumerate(chunks):
                    # This clone is essential. Not sure why.
                    local_opt_states[rank]["state"][k][k2] = param.clone()
                if num_to_pad > 0 and do_pad:
                    local_opt_states[-1]["state"][k][k2] = F.pad(
                        local_opt_states[-1]["state"][k][k2], [0, num_to_pad]
                    )
    # Update Opt keys that arent state
    for k in merged_opt_state.keys():
        if k == "state":
            continue
        for i in range(target_ddp_size):
            local_opt_states[i][k] = merged_opt_state[k]
    _save_shards_to_disk(
        local_state_dicts,
        dummy_model_state,
        state,
        save_dir,
        middle,
        local_opt_states=local_opt_states,
        target_ddp_size=target_ddp_size,
    )


"""
python scripts/reshard_mp.py $model_dir/checkpoint_last  125_mp_reshard --mpart 0
"""

if __name__ == "__main__":
    Fire(reshard_mp)
