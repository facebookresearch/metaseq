# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import re
from glob import glob

import fire
import torch

logging.basicConfig(format="%(asctime)s | %(name)s | %(message)s", level=logging.INFO)
logger: logging.Logger = logging.getLogger("metaseq.scripts.reshard_mp")


def reshard_model_parallel_parts(
    input: str, output: str, num_output_parts: int = 1, eps: float = 1e-8
) -> None:
    """
    Reshard model parallel (MP) parts and write outputs to files. The model weights
    are merged from the input parts before the resharding logic applies. The model
    parallel parts in the input are expected to contain unflattened, FSDP-consolidated
    model weights (see the script `reshard_fsdp.py` for related information.)

    Args:
        :param input: A glob pattern specifying the path names of the input shards.
            (e.g. "checkpoints/opt-2.7b/reshard_no_os/reshard-model_part-*.pt").
        :param output: A string pattern specifying the path names of the output shards.
            Shard indices can be included in the path names if the pattern includes `{i}`.
            (e.g. "checkpoints/opt-2.7b/reshard_no_os_mp8/reshard-model_part-{i}.pt").
        :param num_output_parts: The number of output model parallel parts.
        :param eps: A tolerance threshold for the maximum discrepancy between MP parts.
    """
    files = glob(input)
    N, M = len(files), num_output_parts
    if N == 0:
        raise ValueError("The glob pattern doesn't match any model parallel parts.")
    files = sorted(files, key=lambda x: list(map(int, re.findall(r"\d+", x))))
    logger.info(f"Found {len(files)} model parallel parts ({files[0]} to {files[-1]})")

    rank0_state_dict = torch.load(files[0], torch.device("cpu"))
    dim0_shard_regex = re.compile("embed_tokens|ffn_layernorm|fc1|(k|q|v)_proj")
    dim1_shard_regex = re.compile("(fc2|out_proj).weight")
    shared_regex = re.compile(
        "embed_positions|layer_norm|(fc2|out_proj).bias|output_projection|version"
    )

    unsharded_dict = {}
    logger.info("Allocating memory for unsharded checkpoint")
    for key, value in rank0_state_dict["model"].items():
        d0, d1 = value.size()[0], value.size()[1:]
        if "qkv" in key:
            unsharded_dict[key] = value.new_zeros(3, N * d0 // 3, *d1)
        elif dim0_shard_regex.search(key):
            unsharded_dict[key] = value.new_zeros(N * d0, *d1)
        elif dim1_shard_regex.search(key):
            assert len(d1) > 0
            unsharded_dict[key] = value.new_zeros(d0, N * d1[0])

    # Iteratively load checkpoints to avoid OOM issues
    for i, file in enumerate(files):
        logger.info(f"Merging {file} into unsharded checkpoint")
        state_dict = torch.load(file, torch.device("cpu"))
        for key, value in state_dict["model"].items():
            d0, d1 = value.size()[0], value.size()[1:]
            if "qkv" in key:
                # Split and copy QKV weights
                unsharded_dict[key][:, i * d0 // 3 : (i + 1) * d0 // 3].copy_(
                    value.view(3, d0 // 3, *d1)
                )
            elif dim0_shard_regex.search(key):
                # Concatenate along dim 0 (e.g. embed_tokens, fc1.weight, fc1.bias)
                unsharded_dict[key][i * d0 : (i + 1) * d0].copy_(value)
            elif dim1_shard_regex.search(key):
                # Concatenate along dim 1 (e.g. fc2.weight, out_proj.weight)
                unsharded_dict[key][:, i * d1[0] : (i + 1) * d1[0]].copy_(value)
            elif shared_regex.search(key):
                # Copy from rank 0 (e.g. embed_positions, final_layer_norm, fc2.bias, out_proj.bias)
                unsharded_dict[key] = value
                diff = _max_diff(rank0_state_dict["model"][key], value)
                if diff > eps:
                    logger.warning(f"Max value discrepancy for key '{key}': {diff:.4e}")

    for i in range(M):
        sharded_dict = {}
        logger.info(f"Resharding state dict for model parallel part {i}")
        for key, value in rank0_state_dict["model"].items():
            if "qkv" in key:
                # Merge QKV weights after chunking unsharded weight
                sharded_dict[key] = unsharded_dict[key].chunk(M, dim=1)[i].flatten(0, 1)
            elif dim0_shard_regex.search(key):
                #  Cloning is needed as torch.save always writes unsliced tensors
                sharded_dict[key] = unsharded_dict[key].chunk(M)[i].clone()
            elif dim1_shard_regex.search(key):
                sharded_dict[key] = unsharded_dict[key].chunk(M, dim=1)[i].clone()
            elif all(p in key for p in ("embed_positions", "_float_tensor")):
                # Assume embed positions are not learned (e.g. sinusoidal)
                sharded_dict[key] = value.new_zeros(1)
            elif shared_regex.search(key):
                sharded_dict[key] = value.clone()

        state_dict = {"model": sharded_dict}
        # Copy other values from rank 0
        for key in ["cfg", "extra_state", "optimizer_history", "args"]:
            state_dict[key] = rank0_state_dict[key]
        state_dict["cfg"]["model"].model_parallel_size = M

        output_file = output.format(i=i)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        logger.info(f"Writing a resharded state dict to {output_file}")
        torch.save(state_dict, output_file)
    logger.info("Done!")


def _max_diff(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    assert tensor1.size() == tensor2.size()
    return (tensor1 - tensor2).abs().max().item()


if __name__ == "__main__":
    """
    Example usage:
        python -m metaseq.scripts.reshard_mp \
        --input "opt-2.7b/reshard_no_os/reshard-model_part-*.pt" \
        --output "opt-2.7b/reshard_no_os_mp8/reshard-model_part-{i}.pt" \
        --num-output-parts 8
    """
    fire.Fire(reshard_model_parallel_parts)
