#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import fire
import torch
import torch.nn.functional as F


logging.basicConfig(format="%(asctime)s | %(name)s | %(message)s", level=logging.INFO)
logger: logging.Logger = logging.getLogger("metaseq.scripts.reshard_consolidated")


def reshard_consolidated_checkpoint(
    input: str, output: str, num_output_shards: int = 1
) -> None:
    """
    Reshard an FSDP-consolidated checkpoint and write outputs to files. The model weights
    are flattened before the resharding logic applies. The input checkpoint is expected to
    contain unflattened, FSDP-consolidated model weights.

    Args:
        :param input: A path to the input checkpoint (e.g. "opt-2.7b-dp1-mp2/reshard-model_part-0.pt").
        :param output: A string pattern specifying the path names of the output shards.
            Shard indices can be included in the path names if the pattern includes `{i}`.
            (e.g. "opt-2.7b-dp4-mp2/checkpoint_last-model_part-0-shard{i}.pt").
        :param num_output_shards: The number of output shards.
    """
    state_dict = torch.load(input)
    resharded_weights, resharded_metadata = reshard_unflattened_model_weights(
        state_dict["model"], state_dict["shard_metadata"], num_output_shards
    )
    for shard_idx in range(num_output_shards):
        output_file = output.format(i=shard_idx)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        shard_state_dict = {
            "model": resharded_weights[shard_idx],
            "shard_metadata": resharded_metadata[shard_idx],
        }
        for key in state_dict.keys() - shard_state_dict.keys() - {"optimizer_state"}:
            shard_state_dict[key] = state_dict[key]
        logger.info(f"Writing a resharded state dict to {output_file}")
        torch.save(shard_state_dict, output_file)


def reshard_unflattened_model_weights(
    unsharded_weights: Dict[str, torch.Tensor],
    shard_metadata: List[Dict[str, Any]],
    num_output_shards: int = 1,
) -> List[Dict[str, torch.Tensor]]:
    logger.info(f"Reshard unflattened model weights into {num_output_shards} shard(s)")
    resharded_weights = [{} for _ in range(num_output_shards)]

    # We copy the buffer values from the first shard as they are not sharded by FSDP
    for buffer_name in shard_metadata["buffer_names"]:
        if buffer_name not in unsharded_weights:
            raise ValueError(f"No buffer found for buffer name {buffer_name}.")
        for shard_idx in range(num_output_shards):
            resharded_weights[shard_idx][buffer_name] = unsharded_weights[buffer_name]
        unsharded_weights.pop(buffer_name)

    resharded_metadata = [deepcopy(shard_metadata) for _ in range(num_output_shards)]
    for idx, param_metadata in enumerate(shard_metadata["param_metadata"]):
        fsdp_path = param_metadata["fsdp_path"]
        for flat_name, params in param_metadata["params"].items():
            full_key = ".".join([fsdp_path, flat_name]) if fsdp_path else flat_name
            names = [f"{fsdp_path}.{n}" if fsdp_path else n for n in params["names"]]
            flattened = _flatten_tensors([unsharded_weights[k] for k in names])
            # Reshard weights by chunking the unsharded tensor
            weights, paddings = _shard_and_pad_tensor(flattened, num_output_shards)
            for shard_idx, (weight, pad) in enumerate(zip(weights, paddings)):
                resharded_weights[shard_idx][full_key] = weight
                resharded_metadata[shard_idx]["param_metadata"][idx]["params"][
                    flat_name
                ]["padding"] = pad

    return resharded_weights, resharded_metadata


def _flatten_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    output = torch.empty(sum(t.numel() for t in tensors), dtype=tensors[0].dtype)
    offset = 0
    for tensor in tensors:
        output[offset : offset + tensor.numel()].copy_(tensor.view(-1))
        offset += tensor.numel()
    return output


def _shard_and_pad_tensor(
    tensor: torch.Tensor, num_shards: int, dim: int = 0
) -> Tuple[List[torch.Tensor], List[int]]:
    if num_shards == 1:
        return [tensor], [0]
    #  Cloning is needed as torch.save always writes unsliced tensors
    chunks = [chunk.clone() for chunk in tensor.chunk(num_shards, dim=dim)]
    assert len(chunks) == num_shards, len(chunks)
    paddings = [chunks[0].numel() - chunk.numel() for chunk in chunks]
    for idx, (chunk, padding) in enumerate(zip(chunks, paddings)):
        if padding > 0:
            chunks[idx] = F.pad(chunk, [0, padding])
    return chunks, paddings


if __name__ == "__main__":
    """
    Example usage:
        python -m metaseq.scripts.reshard_consolidated \
        --input "opt-2.7b-dp1-mp2/reshard-model_part-0.pt" \
        --output "opt-2.7b-dp4-mp2/checkpoint_last-model_part-0-shard{i}.pt" \
        --num-output-shards 4
    """
    fire.Fire(reshard_consolidated_checkpoint)
