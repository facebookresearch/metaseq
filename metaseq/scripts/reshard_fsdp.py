# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
from copy import deepcopy
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import fire
import torch
import torch.nn.functional as F

logging.basicConfig(format="%(asctime)s | %(name)s | %(message)s", level=logging.INFO)
logger: logging.Logger = logging.getLogger("metaseq.scripts.reshard_fsdp")


_STRING_TO_DTYPE: Dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def reshard_fsdp_checkpoints(
    input: str,
    output: str,
    num_output_shards: int = 1,
    unflatten_weights: bool = True,
    skip_optimizer_state: bool = False,
    output_dtype: Optional[str] = None,
) -> None:
    """
    Reshard FSDP checkpoints and write outputs to files. The model weights and optimizer states
    are merged from the sharded checkpoints before the resharding logic applies. The sharded
    checkpoints are expected to contain shard metadata.

    Args:
        :param input: A glob pattern specifying the path names of the input shards.
            (e.g. "checkpoints/2.7B/raw/checkpoint_last-model_part-0-shard*.pt").
        :param output: A string pattern specifying the path names of the output shards.
            Shard indices can be included in the path names if the pattern includes `{i}`.
            (e.g. "checkpoints/2.7B/reshard/reshard-model_part-0-shard{i}.pt").
        :param num_output_shards: The number of output shards.
        :param unflatten_weights: Specifies whether the weights in the input shards should be
            unflattened. This option is only applicable when the number of output shards is 1.
        :param skip_optimizer_state: Specifies whether to skip the optimizer states from the input shards.
        :param output_dtype: Specifies the weight dtype of output shards (e.g. "fp32", "fp16", "bf16").
            By default, output_dtype is None and no dtype conversion is applied.
    """
    if output_dtype and output_dtype not in _STRING_TO_DTYPE:
        raise ValueError(f"The specified output dtype {output_dtype} is not supported.")
    files = glob(input)
    if len(files) == 0:
        raise ValueError("The glob pattern doesn't match any sharded checkpoints.")
    files = sorted(files, key=lambda x: list(map(int, re.findall(r"\d+", x))))
    logger.info(f"Found {len(files)} sharded checkpoints ({files[0]} to {files[-1]})")

    logger.info("Loading all sharded checkpoints to CPU")
    shard_state_dicts = [torch.load(path, torch.device("cpu")) for path in files]

    resharded_state_dicts = reshard_fsdp_state_dicts(
        shard_state_dicts,
        num_output_shards=num_output_shards,
        unflatten_weights=unflatten_weights,
        skip_optimizer_state=skip_optimizer_state,
        dtype=_STRING_TO_DTYPE.get(output_dtype, None),
    )
    for shard_idx, state_dict in enumerate(resharded_state_dicts):
        output_file = output.format(i=shard_idx)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        logger.info(f"Writing a resharded state dict to {output_file}")
        torch.save(state_dict, output_file)


def reshard_fsdp_state_dicts(
    shard_state_dicts: List[Dict[str, Any]],
    num_output_shards: int = 1,
    unflatten_weights: bool = True,
    skip_optimizer_state: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> List[Dict[str, Any]]:
    logger.info(f"Resharding state dicts into {num_output_shards} shard(s)")
    # Unshard model weights
    resharded_weights, resharded_metadata = reshard_fsdp_model_weights(
        shard_weights=[s["model"] for s in shard_state_dicts],
        shard_metadata=[s["shard_metadata"] for s in shard_state_dicts],
        num_output_shards=num_output_shards,
        unflatten_weights=unflatten_weights,
        dtype=dtype,
    )
    resharded_state_dicts = [{} for _ in range(num_output_shards)]
    for shard_idx, (weight, metadata) in enumerate(
        zip(resharded_weights, resharded_metadata)
    ):
        resharded_state_dicts[shard_idx]["model"] = weight
        resharded_state_dicts[shard_idx]["shard_metadata"] = metadata

    # Unshard last optimizer state
    if not skip_optimizer_state and "last_optimizer_state" in shard_state_dicts[0]:
        # Assume all optimizer states have same padding as model parameters
        param_padding = [[] for _ in range(len(shard_state_dicts))]
        for shard_idx, shard in enumerate(shard_state_dicts):
            for metadata in shard["shard_metadata"]["param_metadata"]:
                param_padding[shard_idx].extend(
                    param["padding"] for param in metadata["params"].values()
                )
        reshared_optim_states = reshard_fsdp_optim_state(
            shard_optim_states=[s["last_optimizer_state"] for s in shard_state_dicts],
            shard_optim_padding=dict(enumerate(param_padding)),
            num_output_shards=num_output_shards,
            dtype=dtype,
        )
        for shard_idx, optim_state in enumerate(reshared_optim_states):
            resharded_state_dicts[shard_idx]["last_optimizer_state"] = optim_state

    # Copy other state values from the first shard
    for key in shard_state_dicts[0]:
        if key not in {"model", "last_optimizer_state", "shard_metadata"}:
            for shard_idx in range(num_output_shards):
                resharded_state_dicts[shard_idx][key] = shard_state_dicts[0][key]

    return resharded_state_dicts


def reshard_fsdp_model_weights(
    shard_weights: List[Dict[str, torch.Tensor]],
    shard_metadata: List[Dict[str, Any]],
    num_output_shards: int = 1,
    unflatten_weights: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> List[Dict[str, torch.Tensor]]:
    logger.info(f"Resharding model weights into {num_output_shards} shard(s)")
    if len(shard_weights) != len(shard_metadata):
        raise ValueError("Expect shard weights and shard metadata to have same length.")
    if unflatten_weights and num_output_shards > 1:
        raise ValueError("Unflatten weights only if the number of output shards is 1.")

    resharded_weights = [{} for _ in range(num_output_shards)]
    resharded_metadata = [deepcopy(shard_metadata[0]) for _ in range(num_output_shards)]
    for idx, param_metadata in enumerate(shard_metadata[0]["param_metadata"]):
        fsdp_path = param_metadata["fsdp_path"]
        for flat_name, param_info in param_metadata["params"].items():
            full_key = ".".join([fsdp_path, flat_name]) if fsdp_path else flat_name
            if full_key not in shard_weights[0]:
                raise ValueError(f"No weight found for key {full_key} in metadata.")

            # Unshard FSDP tensor weights
            sharded_weights = []
            for weight, metadata in zip(shard_weights, shard_metadata):
                pad = metadata["param_metadata"][idx]["params"][flat_name]["padding"]
                sharded_weight = _maybe_type(weight[full_key], dtype)
                sharded_weights.append(_unpad_tensor(sharded_weight, pad))
            unsharded_weights = torch.cat(sharded_weights, dim=0)

            # For single shard output, tensor weights can be unflattened
            if unflatten_weights:
                names, shapes, numels, _ = param_info.values()
                assert sum(numels) == unsharded_weights.size(0)
                for n, t, s in zip(names, unsharded_weights.split(numels), shapes):
                    param_name = ".".join([fsdp_path, n]) if fsdp_path else n
                    resharded_weights[0][param_name] = t.view(s)
                resharded_metadata = [{}] * num_output_shards
                continue

            # Otherwise, reshard weights by chunking the unsharded tensor
            weights, paddings = _shard_and_pad_tensor(
                unsharded_weights, num_output_shards
            )
            for shard_idx, (weight, pad) in enumerate(zip(weights, paddings)):
                resharded_weights[shard_idx][full_key] = weight
                resharded_metadata[shard_idx]["param_metadata"][idx]["params"][
                    flat_name
                ]["padding"] = pad

        # Copy shared parameters
        if unflatten_weights:
            for src_path, dest_path in metadata.get("shared_param_info", []):
                resharded_weights[0][dest_path] = resharded_weights[src_path]

    # We copy the buffer values from the first shard as they are not sharded by FSDP
    for buffer_name in shard_metadata[0]["buffer_names"]:
        if buffer_name not in shard_weights[0]:
            raise ValueError(f"No buffer found for buffer name {buffer_name}.")
        for shard_idx in range(num_output_shards):
            resharded_weights[shard_idx][buffer_name] = _maybe_type(
                shard_weights[0][buffer_name], dtype
            )

    return resharded_weights, resharded_metadata


def reshard_fsdp_optim_state(
    shard_optim_states: List[Dict[str, Any]],
    shard_optim_padding: Optional[Dict[str, int]] = None,
    num_output_shards: int = 1,
    dtype: Optional[torch.dtype] = None,
) -> List[Dict[str, Any]]:
    logger.info(f"Resharding optimizer states into {num_output_shards} shard(s)")
    if any("state" not in s for s in shard_optim_states):
        raise ValueError(f"Each sharded optimizer state dict should contain `state`.")

    # Reshard optimizer states
    resharded_state_dict = [{"state": {}} for _ in range(num_output_shards)]
    for idx, wrapped_state_dict in shard_optim_states[0]["state"].items():
        for shard_idx in range(num_output_shards):
            resharded_state_dict[shard_idx]["state"][idx] = {}
        for key, value in wrapped_state_dict.items():
            #  Copy non-tensor objects to outputs (e.g. step)
            if not torch.is_tensor(value) or value.dim() == 0:
                sharded_value = _maybe_type(value, dtype)
                for shard_idx in range(num_output_shards):
                    resharded_state_dict[shard_idx]["state"][idx][key] = sharded_value
                continue

            unsharded_value = torch.cat(
                [_maybe_type(s["state"][idx][key], dtype) for s in shard_optim_states]
            )
            unpadded_value = _unpad_tensor(
                shard=unsharded_value,
                pad=shard_optim_padding.get(key, 0) if shard_optim_padding else 0,
            )
            chunks, _ = _shard_and_pad_tensor(unpadded_value, num_output_shards)
            for shard_idx, chunk in enumerate(chunks):
                resharded_state_dict[shard_idx]["state"][idx][key] = chunk

    # Copy unsharded values from the first shard (e.g. loss scale, param groups)
    for key in shard_optim_states[0]:
        for shard_idx in range(num_output_shards):
            if key != "state":
                resharded_state_dict[shard_idx][key] = _maybe_type(
                    shard_optim_states[0][key], dtype
                )
    return resharded_state_dict


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


def _unpad_tensor(shard: torch.Tensor, pad: int) -> torch.Tensor:
    if pad > 0:
        shard = shard[:-pad]
    return shard


def _maybe_type(input: torch.Tensor, dtype: Optional[torch.dtype] = None):
    return input.type(dtype) if dtype is not None else input


if __name__ == "__main__":
    """
    Example usage:
        python -m metaseq.scripts.reshard_fsdp \
        --input "opt-2.7b/raw/checkpoint_last-model_part-0-shard*.pt" \
        --output "opt-2.7b/reshard/reshard-model_part-0.pt" \
        --num-output-shards 1 --skip-optimizer-state True --unflatten-weights True
    """
    fire.Fire(reshard_fsdp_checkpoints)
