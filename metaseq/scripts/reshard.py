# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
import time
import os
import re
import math
from copy import deepcopy
from glob import glob
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import fire
import torch
import torch.nn.functional as F

from metaseq.launcher.opt_job_constants import MODEL_SIZES
from metaseq.logging import get_logger

logger = get_logger(__name__)


def reshard_fsdp_checkpoints(
    input_glob_pattern: str,
    output_pattern: str,
    num_output_shards: int = 1,
    num_output_model_parallel: int = 1,
    unflatten_weights: bool = False,
    skip_optimizer_state: bool = False,
    pretrained_model_size: str = None,
) -> None:
    """
    Reshard FSDP and MP in checkpoints and write outputs to files. The model weights and optimizer states
    are merged from the sharded checkpoints before the resharding logic applies. The sharded
    checkpoints are expected to contain shard metadata.

    Args:
        :param input_glob_pattern: A glob pattern specifying the path names of the input files.
            (e.g. "checkpoints/2.7B/raw/checkpoint_last-*.pt").
        :param output_pattern: A string pattern specifying the path names of the output files.
            Model Parallel indices can be included in the path names if the pattern includes `{mp}`.
            FSDP Shard indices can be included in the path names if the pattern includes `{i}`.
            (e.g. "checkpoints/2.7B/reshard/reshard-model_part-{mp}-shard{i}.pt").
        :param num_output_shards: The number of output fsdp shards.
        :param num_output_model_parallel: The number of model parallel shards.
        :param unflatten_weights: Specifies whether the weights in the input shards should be
            unflattened. Will also flatten the weights if set to False and the input shards are in unflattened state.
        :param skip_optimizer_state: Specifies whether to skip the optimizer states from the input shards.
        :param pretrained_model_size: The model size of the pretrained model. This is required if the input shards
            are flattened but do not contain `shard_metadata`.
    """
    files = glob(input_glob_pattern)
    if len(files) == 0:
        raise ValueError("The glob pattern doesn't match any sharded checkpoints.")
    files = sorted(files, key=lambda x: list(map(int, re.findall(r"\d+", x))))
    checkpoint_first = os.path.basename(files[0])
    checkpoint_last = os.path.basename(files[-1])
    logger.info(f"Found {len(files)} sharded checkpoints ({checkpoint_first} to {checkpoint_last})")

    logger.info("Loading all sharded checkpoints to CPU")
    shard_state_dicts = [torch.load(path, torch.device("cpu")) for path in files]

    mp_factor = get_model_parallel_factor(files)
    fsdp_factor = get_fsdp_factor(files)

    resharded_state_dicts = defaultdict(dict)
    for mp_idx in range(mp_factor):
        for shard_idx in range(fsdp_factor):
            resharded_state_dicts[mp_idx][shard_idx] = shard_state_dicts[(mp_idx * fsdp_factor) + shard_idx]
    shard_state_dicts = resharded_state_dicts

    if ("shard_metadata" not in shard_state_dicts[0][0].keys() or len(shard_state_dicts[0][0]["shard_metadata"].keys()) == 0):
        logger.info("The input checkpoints don't contain shard metadata.")
        shard_state_dicts = create_shard_metadata(mp_factor, fsdp_factor, shard_state_dicts, pretrained_model_size)

    is_current_model_flattened = _is_model_flattened(shard_state_dicts)
    _print_model_state(
        f'Detected Input Model State:',
        model_parallel_factor=mp_factor,
        fsdp_factor=fsdp_factor,
        is_flattened=is_current_model_flattened
    )
    _print_model_state(
        f'Desired Output Model State:',
        model_parallel_factor=num_output_model_parallel,
        fsdp_factor=num_output_shards,
        is_flattened=(not unflatten_weights)
    )

    if mp_factor == num_output_model_parallel and fsdp_factor == num_output_shards and (
        is_current_model_flattened != unflatten_weights
    ):
        logger.error(
            f"You attempted to reshard a model, but the input model's MP ({mp_factor}), FDSP ({fsdp_factor}), and Unflattened state match the desired output MP ({num_output_model_parallel}), FSDP ({num_output_shards}), and Unflattened state. Thus no resharding will happen. Aborting."
        )
        exit(1)

    if num_output_shards > 1 and unflatten_weights:
        logger.error(
            f"You configured the script to output an unflattened model with FSDP factor of {num_output_shards}; however, this is currently not supported. You may only produce unflattened models with FSDP factor of 1. Aborting."
        )
        exit(1)

    if mp_factor != num_output_model_parallel:
        logger.info(f"Resharding model parallel from {mp_factor} to {num_output_model_parallel}")
        if is_current_model_flattened:
            logger.info(
                f"You attempted to change MP from {mp_factor} to {num_output_model_parallel} but the models was flattened. It must be unflattened with FSDP 1 to change model parallel factor. Unflattening and consolidating the weights."
            )
            resharded_state_dicts = defaultdict(dict)
            for mp_idx in range(mp_factor):
                resharded_state_dicts[mp_idx] = reshard_fsdp_state_dicts(
                    shard_state_dicts[mp_idx],
                    num_output_shards=1,
                    unflatten_weights=True,
                    skip_optimizer_state=skip_optimizer_state,
                )
            fsdp_factor = 1

        assert fsdp_factor == 1, (
            f"Resharding model parallel is only supported for FSDP consolidated checkpoints. "
            f"Found {fsdp_factor} shards in the input checkpoints."
        )

        shard_state_list = [fsdp_shard_states[0] for fsdp_shard_states in resharded_state_dicts.values()]
        shard_state_list = reshard_model_parallel_parts(shard_state_list, num_output_parts=num_output_model_parallel)
        assert (
            len(shard_state_list) == num_output_model_parallel
        ), f"Expected {num_output_model_parallel} shards, but got {len(shard_state_list)} shards."
        shard_state_dicts = {mp_idx: {0: shard_state_list[mp_idx]} for mp_idx in range(num_output_model_parallel)}
        mp_factor = num_output_model_parallel

        if not unflatten_weights:
            logger.info("You set the desired output to flattened. The checkpoints will be flattened.")
            resharded_state_dicts = defaultdict(dict)
            for mp_idx in range(mp_factor):
                for shard_idx in range(fsdp_factor):
                    resharded_state_dicts[mp_idx][shard_idx] = flatten_state_dict(shard_state_dicts[mp_idx][shard_idx])
            shard_state_dicts = resharded_state_dicts

    if fsdp_factor != num_output_shards:
        logger.info(f"Resharding FSDP from {fsdp_factor} to {num_output_shards}")
        if not _is_model_flattened(shard_state_dicts):
            assert (
                fsdp_factor == 1
            ), f"You may only change the FSDP factor of unflattened models if the input FSDP is 1, but the FSDP factor is {fsdp_factor}."
            logger.info("The input checkpoints are unflattened. Flattening the weights.")
            resharded_state_dicts = defaultdict(dict)
            for mp_idx in range(mp_factor):
                for shard_idx in range(fsdp_factor):
                    resharded_state_dicts[mp_idx][shard_idx] = flatten_state_dict(shard_state_dicts[mp_idx][shard_idx])
            shard_state_dicts = resharded_state_dicts

        resharded_state_dicts = defaultdict(dict)
        for mp_idx in range(mp_factor):
            resharded_state_dicts[mp_idx] = reshard_fsdp_state_dicts(
                shard_state_dicts[mp_idx],
                num_output_shards=num_output_shards,
                unflatten_weights=unflatten_weights,
                skip_optimizer_state=skip_optimizer_state,
            )
        shard_state_dicts = resharded_state_dicts
        fsdp_factor = num_output_shards

    assert (
        len(shard_state_dicts) == num_output_model_parallel
    ), f"Expected {num_output_model_parallel} shards, but got {len(shard_state_dicts)} shards."

    # If model has still not reached the desired flattened state from above transformations
    # perform a final transformation
    is_current_model_flattened = _is_model_flattened(shard_state_dicts)
    is_desired_model_flattened = not unflatten_weights

    if is_current_model_flattened is not is_desired_model_flattened:
        logger.info(
            f"Current model is {_get_flattened_str(_is_model_flattened(shard_state_dicts))} but desired model is {_get_flattened_str(is_desired_model_flattened)}"
        )

        if (is_current_model_flattened is True) and (is_desired_model_flattened is False):
            logger.info(f"Unflatten model")
            resharded_state_dicts = defaultdict(dict)
            for mp_idx in range(mp_factor):
                resharded_state_dicts[mp_idx] = reshard_fsdp_state_dicts(
                    shard_state_dicts[mp_idx],
                    num_output_shards=num_output_shards,
                    unflatten_weights=unflatten_weights,
                    skip_optimizer_state=skip_optimizer_state,
                )
            shard_state_dicts = resharded_state_dicts
            fsdp_factor = num_output_shards

        if (is_current_model_flattened is False) and (is_desired_model_flattened is True):
            logger.info(f"Flatten model")
            assert fsdp_factor == 1, (
                f"You attempted to flatten the model which requires input FSDP factor of 1 but found FSDP fator of {fsdp_factor}."
            )

            resharded_state_dicts = defaultdict(dict)
            for mp_idx in range(mp_factor):
                for shard_idx in range(fsdp_factor):
                    resharded_state_dicts[mp_idx][shard_idx] = flatten_state_dict(shard_state_dicts[mp_idx][shard_idx])
            shard_state_dicts = resharded_state_dicts

    for mp_idx in range(num_output_model_parallel):
        assert (
            len(shard_state_dicts[mp_idx]) == num_output_shards
        ), f"Expected {num_output_shards} shards, but got {len(shard_state_dicts[mp_idx])} shards."

        for shard_idx in range(num_output_shards):
            output_file = output_pattern.format(mp=mp_idx, i=shard_idx)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            logger.info(f"Writing a resharded state dict to {output_file}")
            torch.save(shard_state_dicts[mp_idx][shard_idx], output_file)


def flatten_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an unflattened state dict to flattened state dict"""
    flat_state_dict = defaultdict(dict)
    for buffer_name in state_dict["shard_metadata"]["buffer_names"]:
        flat_state_dict["model"][buffer_name] = state_dict["model"][buffer_name]

    for param_metadata in state_dict["shard_metadata"]["param_metadata"]:
        fsdp_path = param_metadata["fsdp_path"]
        for param_name in param_metadata["params"]:
            if fsdp_path == "":
                output_key_name = param_name
            else:
                output_key_name = ".".join([fsdp_path, param_name])
            unsharded_weights = []
            for key_name in param_metadata["params"][param_name]["names"]:
                if fsdp_path == "":
                    model_key_name = key_name
                else:
                    model_key_name = ".".join([fsdp_path, key_name])
                unsharded_weights.append(state_dict["model"][model_key_name].flatten())
            unsharded_weights = torch.cat(unsharded_weights, dim=0)
            flat_state_dict["model"][output_key_name] = unsharded_weights

    # Copy other values from the state dict
    for key, value in state_dict.items():
        if key not in ["model"]:
            flat_state_dict[key] = value
    return flat_state_dict


def create_shard_metadata(
    mp_factor: int,
    fsdp_factor: int,
    shard_state_dicts: Dict[int, Dict[int, Any]],
    pretrained_model_size: str = None,
) -> Dict[int, Dict[str, Any]]:
    if "flat_param_0" in shard_state_dicts[0][0]["model"].keys():
        logger.info("The input checkpoints are flattened.")
        assert (
            pretrained_model_size is not None
        ), "pretrained_model_size must be specified if the input checkpoints are flattened."
        assert (pretrained_model_size in MODEL_SIZES), f"pretrained_model_size must be one of {MODEL_SIZES.keys()}."
        logger.info(f"Attempting to create shard metadata based on {MODEL_SIZES[pretrained_model_size]} model config.")
        shard_metadata_dicts = create_shard_metadata_based_on_model_config(
            MODEL_SIZES[pretrained_model_size], mp_factor, fsdp_factor
        )
    else:
        logger.info("Attempting to create shard metadata based on model layers.")
        shard_metadata_dicts = create_shard_metadata_based_on_model_layers(
            shard_state_dicts,
            fsdp_factor,
        )

    assert (len(shard_metadata_dicts) == mp_factor), "The number of shard metadata doesn't match the number of shards."
    for mp_idx in range(mp_factor):
        assert (
            len(shard_metadata_dicts[mp_idx]) == fsdp_factor
        ), "The number of shard metadata doesn't match the number of shards."
        for shard_idx in range(fsdp_factor):
            shard_state_dicts[mp_idx][shard_idx]["shard_metadata"] = shard_metadata_dicts[mp_idx][shard_idx]
    return shard_state_dicts


def get_model_parallel_factor(files):
    """Infer model parallel factor from the file names."""
    return _get_max_index_in_nested_list(_get_string_patterns_from_list(r"-model_part-\d+", files)) + 1


def get_fsdp_factor(files):
    """Infer fsdp factor from the file names."""
    return _get_max_index_in_nested_list(_get_string_patterns_from_list(r"-shard\d+", files)) + 1


def _get_max_index_in_nested_list(items: List[List[str]], default: int = 0):
    """Given a nested list, return the max index of the inner list"""
    flattened = [item for sublist in items for item in sublist]
    indexes = [int(next(iter(idxs), default)) for idxs in _get_string_patterns_from_list(r"\d+", flattened)]
    return max(indexes, default=default)


def _get_string_patterns_from_list(regex: str, items: List[str]):
    return [re.findall(regex, item) for item in items]


def create_shard_metadata_based_on_model_layers(shard_state_dicts: Dict[int, Dict[int, Any]],
                                                fsdp_factor: int = 1) -> Dict[int, Dict[int, Any]]:
    """Create shard metadata based on model layers."""
    n_layers = _get_max_index_in_nested_list(
        _get_string_patterns_from_list(r"decoder.layers.\d+", shard_state_dicts[0][0]["model"].keys())
    ) + 1

    logger.info(f"n_layers: {n_layers}")
    shard_metadata_dicts = defaultdict(dict)

    numels_0_emb = [
        math.prod(list(shard_state_dicts[0][0]["model"]["decoder.embed_tokens.weight"].shape)[:2]),
        math.prod(list(shard_state_dicts[0][0]["model"]["decoder.embed_positions.weight"].shape)[:2]),
        #math.prod(list(shard_state_dicts[0][0]["model"]["decoder.layer_norm.weight"].shape)[:1]),
        #math.prod(list(shard_state_dicts[0][0]["model"]["decoder.layer_norm.bias"].shape)[:1]),
    ]
    paddings_emb = [0 for _ in range(fsdp_factor - 1)]
    paddings_emb += [math.ceil(sum(numels_0_emb) / fsdp_factor) * fsdp_factor - sum(numels_0_emb)]

    paddings_layers = []
    for layer_idx in range(n_layers):
        numels_0_layer = [
            math.prod(
                list(shard_state_dicts[0][0]["model"][f"decoder.layers.{layer_idx}.self_attn.k_proj.weight"].shape)[:2]
            ),
            math.prod(
                list(shard_state_dicts[0][0]["model"][f"decoder.layers.{layer_idx}.self_attn.v_proj.weight"].shape)[:2]
            ),
            math.prod(
                list(shard_state_dicts[0][0]["model"][f"decoder.layers.{layer_idx}.self_attn.q_proj.weight"].shape)[:2]
            ),
            #math.prod(list(shard_state_dicts[0][0]["model"][f"decoder.layers.{layer_idx}.self_attn.qkv_proj.bias"].shape)[:1]),
            math.prod(
                list(shard_state_dicts[0][0]["model"][f"decoder.layers.{layer_idx}.self_attn.out_proj.weight"].shape)[:2]
            ),
            #math.prod(list(shard_state_dicts[0][0]["model"][f"decoder.layers.{layer_idx}.self_attn.out_proj.bias"].shape)[:1]),
            # math.prod(
            #     list(shard_state_dicts[0][0]["model"][f"decoder.layers.{layer_idx}.self_attn_layer_norm.weight"].shape)[:1]
            # ),
            # math.prod(
            #     list(shard_state_dicts[0][0]["model"][f"decoder.layers.{layer_idx}.self_attn_layer_norm.bias"].shape)[:1]
            # ),
            math.prod(list(shard_state_dicts[0][0]["model"][f"decoder.layers.{layer_idx}.fc1.weight"].shape)[:2]),
            #math.prod(list(shard_state_dicts[0][0]["model"][f"decoder.layers.{layer_idx}.fc1.bias"].shape)[:1]),
            math.prod(list(shard_state_dicts[0][0]["model"][f"decoder.layers.{layer_idx}.fc2.weight"].shape)[:2]),
            #math.prod(list(shard_state_dicts[0][0]["model"][f"decoder.layers.{layer_idx}.fc2.bias"].shape)[:1]),
            #math.prod(list(shard_state_dicts[0][0]["model"][f"decoder.layers.{layer_idx}.final_layer_norm.weight"].shape)[:1]),
            #math.prod(list(shard_state_dicts[0][0]["model"][f"decoder.layers.{layer_idx}.final_layer_norm.bias"].shape)[:1]),
        ]
        paddings_layer = [0 for _ in range(fsdp_factor - 1)]
        paddings_layer += [math.ceil(sum(numels_0_layer) / fsdp_factor) * fsdp_factor - sum(numels_0_layer)]
        paddings_layers.append(paddings_layer)

    for mp_idx, mp_shard_state_dicts in shard_state_dicts.items():
        for shard_idx, shard_state_dict in mp_shard_state_dicts.items():
            shard_metadata = defaultdict(dict)
            shard_metadata["param_metadata"] = []
            shapes = [
                list(shard_state_dict["model"]["decoder.embed_tokens.weight"].shape)[:2],
                list(shard_state_dict["model"]["decoder.embed_positions.weight"].shape)[:2],
                #list(shard_state_dict["model"]["decoder.layer_norm.weight"].shape)[:1],
                #list(shard_state_dict["model"]["decoder.layer_norm.bias"].shape)[:1],
            ]
            numels = [
                math.prod(list(shard_state_dict["model"]["decoder.embed_tokens.weight"].shape)[:2]),
                math.prod(list(shard_state_dict["model"]["decoder.embed_positions.weight"].shape)[:2]),
                #math.prod(list(shard_state_dict["model"]["decoder.layer_norm.weight"].shape)[:1]),
                #math.prod(list(shard_state_dict["model"]["decoder.layer_norm.bias"].shape)[:1]),
            ]
            shard_metadata["param_metadata"].append(
                {
                    "fsdp_path": "",
                    "params": {
                        "flat_param_0": {
                            "names": [
                                "decoder.embed_tokens.weight",
                                "decoder.embed_positions.weight",
                                #"decoder.layer_norm.weight",
                                #"decoder.layer_norm.bias",
                            ],
                            "shapes":
                            shapes,
                            "numels":
                            numels,
                            "padding":
                            paddings_emb[shard_idx],
                        },
                    },
                    "no_broadcast_optim_state": False,
                    "shared_param_info": [[
                        "decoder.embed_tokens.weight",
                        "decoder.output_projection.weight",
                    ]],
                }
            )

            for layer_idx in range(n_layers):
                shapes = [
                    list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.self_attn.k_proj.weight"].shape)[:2],
                    list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.self_attn.v_proj.weight"].shape)[:2],
                    list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.self_attn.q_proj.weight"].shape)[:2],
                    #list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.self_attn.qkv_proj.bias"].shape)[:1],
                    list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.self_attn.out_proj.weight"].shape)[:2],
                    #list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.self_attn.out_proj.bias"].shape)[:1],
                    #list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.self_attn_layer_norm.weight"].shape)[:1],
                    #list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.self_attn_layer_norm.bias"].shape)[:1],
                    list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.fc1.weight"].shape)[:2],
                    #list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.fc1.bias"].shape)[:1],
                    list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.fc2.weight"].shape)[:2],
                    #list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.fc2.bias"].shape)[:1],
                    #list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.final_layer_norm.weight"].shape)[:1],
                    #list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.final_layer_norm.bias"].shape)[:1],
                ]
                numels = [
                    math.prod(
                        list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.self_attn.k_proj.weight"].shape)[:2]
                    ),
                    math.prod(
                        list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.self_attn.v_proj.weight"].shape)[:2]
                    ),
                    math.prod(
                        list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.self_attn.q_proj.weight"].shape)[:2]
                    ),
                    # math.prod(
                    #     list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.self_attn.qkv_proj.bias"].shape)[:1]
                    # ),
                    math.prod(
                        list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.self_attn.out_proj.weight"].shape)[:2]
                    ),
                    # math.prod(
                    #     list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.self_attn.out_proj.bias"].shape)[:1]
                    # ),
                    # math.prod(
                    #     list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.self_attn_layer_norm.weight"].shape)[:1]
                    # ),
                    # math.prod(
                    #     list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.self_attn_layer_norm.bias"].shape)[:1]
                    # ),
                    math.prod(list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.fc1.weight"].shape)[:2]),
                    #math.prod(list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.fc1.bias"].shape)[:1]),
                    math.prod(list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.fc2.weight"].shape)[:2]),
                    #math.prod(list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.fc2.bias"].shape)[:1]),
                    # math.prod(
                    #     list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.final_layer_norm.weight"].shape)[:1]
                    # ),
                    # math.prod(list(shard_state_dict["model"][f"decoder.layers.{layer_idx}.final_layer_norm.bias"].shape)[:1]),
                ]
                shard_metadata["param_metadata"].append(
                    {
                        "fsdp_path": f"decoder.layers.{layer_idx}",
                        "params": {
                            "flat_param_0": {
                                "names": [
                                    "self_attn.k_proj.weight",
                                    "self_attn.v_proj.weight",
                                    "self_attn.q_proj.weight",
                                    #"self_attn.qkv_proj.bias",
                                    "self_attn.out_proj.weight",
                                    #"self_attn.out_proj.bias",
                                    #"self_attn_layer_norm.weight",
                                    #"self_attn_layer_norm.bias",
                                    "fc1.weight",
                                    #"fc1.bias",
                                    "fc2.weight",
                                    #"fc2.bias",
                                    #"final_layer_norm.weight",
                                    #"final_layer_norm.bias",
                                ],
                                "shapes":
                                shapes,
                                "numels":
                                numels,
                                "padding":
                                paddings_layers[layer_idx][shard_idx],
                            }
                        },
                        "no_broadcast_optim_state": False,
                        "shared_param_info": [],
                    }
                )
            shard_metadata["buffer_names"] = ["decoder.version"]
            shard_metadata_dicts[mp_idx][shard_idx] = shard_metadata
    return shard_metadata_dicts


def create_shard_metadata_based_on_model_config(model_config: Dict[str, Any],
                                                mp_factor: int = 1,
                                                fsdp_factor: int = 1) -> Dict[int, Dict[int, Any]]:
    """Creates shard metadata based on pretrained model config."""
    logger.info(f"n_layers: {model_config.n_layers}")
    shard_metadata_dicts = defaultdict(dict)

    # Fixed Vocab Size of 50272 and Embedding Positions 2050
    VOCAB_SIZE = 66056
    EMBED_POSITIONS = 4098
    INTERMEDIATE_SIZE = model_config.emb_size #* 4
    QKV_SIZE = model_config.n_heads * model_config.d_head #* 3

    shapes_emb = [
        [
            # Fixed Vocab Size of 50272
            VOCAB_SIZE // mp_factor,
            model_config.emb_size,
        ],
        [
            EMBED_POSITIONS,
            model_config.emb_size,
        ],
        #[model_config.emb_size],
        #[model_config.emb_size],
    ]
    numels_emb = [
        (VOCAB_SIZE // mp_factor) * model_config.emb_size,
        EMBED_POSITIONS * model_config.emb_size,
       # model_config.emb_size,
        #model_config.emb_size,
    ]
    paddings_emb = [0 for _ in range(fsdp_factor - 1)]
    paddings_emb += [math.ceil(sum(numels_emb) / fsdp_factor) * fsdp_factor - sum(numels_emb)]

    shapes_layer = [
        [
            QKV_SIZE // mp_factor,
            model_config.emb_size,
        ],
        [
            QKV_SIZE // mp_factor,
            model_config.emb_size,
        ],
        [
            QKV_SIZE // mp_factor,
            model_config.emb_size,
        ],
        #[QKV_SIZE // mp_factor],
        [
            model_config.emb_size,
            (model_config.n_heads * model_config.d_head) // mp_factor,
        ],
        #[model_config.emb_size],
        #[model_config.emb_size],
        #[model_config.emb_size],
        [(INTERMEDIATE_SIZE) // mp_factor, model_config.emb_size],
        #[(INTERMEDIATE_SIZE) // mp_factor],
        [model_config.emb_size, (INTERMEDIATE_SIZE) // mp_factor],
        #[model_config.emb_size],
        #[model_config.emb_size],
        #[model_config.emb_size],
    ]
    numels_layer = [
        (QKV_SIZE // mp_factor) * model_config.emb_size,
        (QKV_SIZE // mp_factor) * model_config.emb_size,
        (QKV_SIZE // mp_factor) * model_config.emb_size,
        #QKV_SIZE // mp_factor,
        model_config.emb_size * ((model_config.n_heads * model_config.d_head) // mp_factor),
        #model_config.emb_size,
        #model_config.emb_size,
        #model_config.emb_size,
        ((INTERMEDIATE_SIZE) // mp_factor) * model_config.emb_size,
       # (INTERMEDIATE_SIZE) // mp_factor,
        model_config.emb_size * ((INTERMEDIATE_SIZE) // mp_factor),
        #model_config.emb_size,
        #model_config.emb_size,
        #model_config.emb_size,
    ]
    paddings_layer = [0 for _ in range(fsdp_factor - 1)]
    paddings_layer += [math.ceil(sum(numels_layer) / fsdp_factor) * fsdp_factor - sum(numels_layer)]

    for mp_idx in range(mp_factor):
        for shard_idx in range(fsdp_factor):
            shard_metadata = {}
            shard_metadata["param_metadata"] = []
            shard_metadata["param_metadata"].append(
                {
                    "fsdp_path": "",
                    "params": {
                        "flat_param_0": {
                            "names": [
                                "decoder.embed_tokens.weight",
                                "decoder.embed_positions.weight",
                                #"decoder.layer_norm.weight",
                                #"decoder.layer_norm.bias",
                            ],
                            "shapes":
                            shapes_emb,
                            "numels":
                            numels_emb,
                            "padding":
                            paddings_emb[shard_idx],
                        },
                    },
                    "no_broadcast_optim_state": False,
                    "shared_param_info": [[
                        "decoder.embed_tokens.weight",
                        "decoder.output_projection.weight",
                    ]],
                }
            )

            for layer_idx in range(model_config.n_layers):
                shard_metadata["param_metadata"].append(
                    {
                        "fsdp_path": f"decoder.layers.{layer_idx}",
                        "params": {
                            "flat_param_0": {
                                "names": [
                                    "self_attn.k_proj.weight",
                                    "self_attn.v_proj.weight",
                                    "self_attn.q_proj.weight",
                                    #"self_attn.qkv_proj.weight",
                                    "self_attn.out_proj.weight",
                                    #"self_attn.out_proj.bias",
                                   # "self_attn_layer_norm.weight",
                                    #"self_attn_layer_norm.bias",
                                    "fc1.weight",
                                    #"fc1.bias",
                                    "fc2.weight",
                                    #"fc2.bias",
                                    #"final_layer_norm.weight",
                                    #"final_layer_norm.bias",
                                ],
                                "shapes":
                                shapes_layer,
                                "numels":
                                numels_layer,
                                "padding":
                                paddings_layer[shard_idx],
                            }
                        },
                        "no_broadcast_optim_state": False,
                        "shared_param_info": [],
                    }
                )
            shard_metadata["buffer_names"] = ["decoder.version"]
            shard_metadata_dicts[mp_idx][shard_idx] = shard_metadata
    return shard_metadata_dicts


def reshard_fsdp_state_dicts(
    shard_state_dicts: Dict[int, Dict[str, Any]],
    num_output_shards: int = 1,
    unflatten_weights: bool = True,
    skip_optimizer_state: bool = False,
) -> Dict[int, Dict[str, Any]]:
    logger.info(f"Resharding state dicts into {num_output_shards} fsdp shard(s)")
    # Unshard model weights
    resharded_weights, resharded_metadata = reshard_fsdp_model_weights(
        shard_weights=[s["model"] for s in shard_state_dicts.values()],
        shard_metadata=[s["shard_metadata"] for s in shard_state_dicts.values()],
        num_output_shards=num_output_shards,
        unflatten_weights=unflatten_weights,
    )
    resharded_state_dicts = [{} for _ in range(num_output_shards)]
    for shard_idx, (weight, metadata) in enumerate(zip(resharded_weights, resharded_metadata)):
        resharded_state_dicts[shard_idx]["model"] = weight
        resharded_state_dicts[shard_idx]["shard_metadata"] = metadata

    # Unshard last optimizer state
    if not skip_optimizer_state and "last_optimizer_state" in shard_state_dicts[0]:
        # Assume all optimizer states have same padding as model parameters
        param_padding = [[] for _ in range(len(shard_state_dicts))]
        for shard_idx, shard in shard_state_dicts.items():
            for metadata in shard["shard_metadata"]["param_metadata"]:
                param_padding[shard_idx].extend(param["padding"] for param in metadata["params"].values())
        reshared_optim_states = reshard_fsdp_optim_state(
            shard_optim_states=[s["last_optimizer_state"] for s in shard_state_dicts.values()],
            shard_optim_padding=dict(enumerate(param_padding)),
            num_output_shards=num_output_shards,
        )
        for shard_idx, optim_state in enumerate(reshared_optim_states):
            resharded_state_dicts[shard_idx]["last_optimizer_state"] = optim_state

    # Copy other state values from the first shard
    for key in shard_state_dicts[0]:
        if key not in {"model", "last_optimizer_state", "shard_metadata"}:
            for shard_idx in range(num_output_shards):
                resharded_state_dicts[shard_idx][key] = shard_state_dicts[0][key]

    shard_state_dicts = defaultdict(dict)
    for shard_idx, state_dict in enumerate(resharded_state_dicts):
        shard_state_dicts[shard_idx] = state_dict
    return shard_state_dicts


def reshard_fsdp_model_weights(
    shard_weights: List[Dict[str, torch.Tensor]],
    shard_metadata: List[Dict[str, Any]],
    num_output_shards: int = 1,
    unflatten_weights: bool = False,
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
                sharded_weights.append(_unpad_tensor(weight[full_key], pad))
            unsharded_weights = torch.cat(sharded_weights, dim=0)

            # For single shard output, tensor weights can be unflattened
            if unflatten_weights:
                names, shapes, numels, _ = param_info.values()
                assert sum(numels) == unsharded_weights.size(0)
                for n, t, s in zip(names, unsharded_weights.split(numels), shapes):
                    param_name = ".".join([fsdp_path, n]) if fsdp_path else n
                    resharded_weights[0][param_name] = t.view(s)
                continue

            # Otherwise, reshard weights by chunking the unsharded tensor
            weights, paddings = _shard_and_pad_tensor(unsharded_weights, num_output_shards)
            for shard_idx, (weight, pad) in enumerate(zip(weights, paddings)):
                resharded_weights[shard_idx][full_key] = weight
                resharded_metadata[shard_idx]["param_metadata"][idx]["params"][flat_name]["padding"] = pad

        # Copy shared parameters
        if unflatten_weights:
            for (src_path, dest_path) in param_metadata.get("shared_param_info", []):
                resharded_weights[0][dest_path] = resharded_weights[0][src_path]

    # We copy the buffer values from the first shard as they are not sharded by FSDP
    for buffer_name in shard_metadata[0]["buffer_names"]:
        if buffer_name not in shard_weights[0]:
            raise ValueError(f"No buffer found for buffer name {buffer_name}.")
        for shard_idx in range(num_output_shards):
            resharded_weights[shard_idx][buffer_name] = shard_weights[0][buffer_name]

    return resharded_weights, resharded_metadata


def reshard_fsdp_optim_state(
    shard_optim_states: List[Dict[str, Any]],
    shard_optim_padding: Optional[Dict[str, int]] = None,
    num_output_shards: int = 1,
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
                for shard_idx in range(num_output_shards):
                    resharded_state_dict[shard_idx]["state"][idx][key] = value
                continue

            unsharded_value = _unpad_tensor(
                torch.cat([s["state"][idx][key] for s in shard_optim_states]),
                pad=shard_optim_padding.get(key, 0) if shard_optim_padding else 0,
            )
            chunks, _ = _shard_and_pad_tensor(unsharded_value, num_output_shards)
            for shard_idx, chunk in enumerate(chunks):
                resharded_state_dict[shard_idx]["state"][idx][key] = chunk

    # Copy unsharded values from the first shard (e.g. loss scale, param groups)
    for key in shard_optim_states[0]:
        for shard_idx in range(num_output_shards):
            if key != "state":
                resharded_state_dict[shard_idx][key] = shard_optim_states[0][key]

    return resharded_state_dict


def _shard_and_pad_tensor(tensor: torch.Tensor, num_shards: int, dim: int = 0) -> Tuple[List[torch.Tensor], List[int]]:
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


def reshard_model_parallel_parts(state_dicts: List[Dict[str, Any]],
                                 num_output_parts: int = 1,
                                 eps: float = 1e-8) -> List[Dict[str, Any]]:
    """
    Reshard model parallel (MP) parts. The model weights
    are merged from the input parts before the resharding logic applies. The model
    parallel parts in the input are expected to contain unflattened, FSDP-consolidated
    model weights (see the script `reshard_fsdp.py` for related information.)

    Args:
        :param input: A glob pattern specifying the path names of the input shards.
            (e.g. "checkpoints/opt-2.7b/reshard_no_os/reshard-model_part-*.pt").
        :param num_output_parts: The number of output model parallel parts.
        :param eps: A tolerance threshold for the maximum discrepancy between MP parts.
    """
    N, M = len(state_dicts), num_output_parts

    rank0_state_dict = state_dicts[0]
    dim0_shard_regex = re.compile("embed_tokens|output_projection|ffn_layernorm|fc1|(k|q|v)_proj")
    dim1_shard_regex = re.compile("(fc2|out_proj).weight")
    shared_regex = re.compile("embed_positions|layer_norm|(fc2|out_proj).bias|version")

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

    for i, state_dict in enumerate(state_dicts):
        for key, value in state_dict["model"].items():
            d0, d1 = value.size()[0], value.size()[1:]
            if "qkv" in key:
                # Split and copy QKV weights
                unsharded_dict[key][:, i * d0 // 3:(i + 1) * d0 // 3].copy_(value.view(3, d0 // 3, *d1))
            elif dim0_shard_regex.search(key):
                # Concatenate along dim 0 (e.g. embed_tokens, fc1.weight, fc1.bias)
                unsharded_dict[key][i * d0:(i + 1) * d0].copy_(value)
            elif dim1_shard_regex.search(key):
                # Concatenate along dim 1 (e.g. fc2.weight, out_proj.weight)
                unsharded_dict[key][:, i * d1[0]:(i + 1) * d1[0]].copy_(value)
            elif shared_regex.search(key):
                # Copy from rank 0 (e.g. embed_positions, final_layer_norm, fc2.bias, out_proj.bias)
                unsharded_dict[key] = value
                diff = _max_diff(rank0_state_dict["model"][key], value)
                if diff > eps:
                    logger.warning(f"Max value discrepancy for key '{key}': {diff:.4e}")

    shard_state_list = []
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
        for key in rank0_state_dict.keys():
            if key not in ["model", "shard_metadata"]:
                state_dict[key] = rank0_state_dict[key]
        state_dict["cfg"]["model"].model_parallel_size = M
        shard_state_list.append(state_dict)

    shard_metadata_dict = create_shard_metadata_based_on_model_layers(
        {mp_idx: {
            0: shard_state_dict
        }
         for mp_idx, shard_state_dict in enumerate(shard_state_list)},
        fsdp_factor=1,
    )
    shard_metadata_list = [shard_metadata[0] for shard_metadata in shard_metadata_dict.values()]
    assert len(shard_metadata_list
               ) == len(shard_state_list), "The number of shard metadata doesn't match the number of shards."

    for shard_state_dict, shard_metadata in zip(shard_state_list, shard_metadata_list):
        shard_state_dict["shard_metadata"] = shard_metadata
    return shard_state_list


def _max_diff(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    assert tensor1.size() == tensor2.size()
    return (tensor1 - tensor2).abs().max().item()


def _get_flattened_str(is_flattened: bool):
    return "Flattened" if is_flattened else "Unflattened"


def _is_model_flattened(shard_state_dicts: list):
    return "flat_param_0" in shard_state_dicts[0][0]["model"].keys()


def _print_model_state(title: str, model_parallel_factor: int, fsdp_factor: int, is_flattened: bool):
    logger.info(title)
    logger.info(f"- Model Parallel (MP) factor:\t\t\t{model_parallel_factor}")
    logger.info(f"- Fully Sharded Data Parallel (FSDP) factor:\t{fsdp_factor}")
    logger.info(f'- Model Weights:\t\t\t\t{_get_flattened_str(is_flattened)}')


if __name__ == "__main__":
    """
    Example usage:
        python -m metaseq.scripts.reshard \
        --input-glob-pattern "opt-2.7b/raw/checkpoint_last-*.pt" \
        --output-pattern "opt-2.7b/reshard/reshard-model_part-{mp}-shard{i}.pt" \
        --num-output-shards 8 \
        --num-output-model-parallel 4 \
        --skip-optimizer-state True \
        --unflatten-weights True
    """
    fire.Fire(reshard_fsdp_checkpoints)
