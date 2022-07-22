# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gc
import logging
import os
import re
import time
from collections import defaultdict, OrderedDict
from glob import glob
from pathlib import Path

import torch
from tqdm import tqdm

from metaseq.distributed.fully_sharded_data_parallel import FSDP as FSDP
from metaseq.file_io import load_and_pop_last_optimizer_state

logger = logging.getLogger(__name__)


def _get_shard_number(x) -> int:
    match = re.search(r"shard(\d+).pt", x)
    if match is None:
        raise AssertionError(f"{x} did not match shard(\\d+).pt")
    else:
        return int(match.groups()[0])


def consolidate_fsdp_shards(
    pth_prefix: str,
    save_prefix=None,
    strict=False,
    new_arch_name=None,
    no_stitch_megatron=False,
    megatron_part=None,
) -> str:
    if pth_prefix.endswith(".pt"):
        pth_prefix = pth_prefix[:-3]
    if save_prefix is None:
        save_prefix = pth_prefix + "_consolidated"  # .pt'
    all_ckpt_files = list(
        sorted(glob(f"{pth_prefix}*shard*.pt"), key=_get_shard_number)
    )
    if megatron_part is not None:
        no_stitch_megatron = True
        all_ckpt_files = [
            x for x in all_ckpt_files if f"model_part-{megatron_part}" in x
        ]
    assert all_ckpt_files, f"no paths matched {pth_prefix}*shard*.pt"
    weights = []
    metadata = []
    expert_paths = []
    expert_dest_paths = []
    expert_ranks = []
    names = []
    dense = True
    t0 = time.time()
    for p in tqdm(all_ckpt_files):
        names.append(Path(p).name)
        if re.search(r"rank-(\d+)", os.path.basename(p)):  # expert checkpoint
            expert_paths.append(p)
            r = re.search(r"rank-(\d+)", os.path.basename(p)).groups()[0]
            assert r not in expert_ranks
            expert_ranks.append(r)
            expert_dest_paths.append(f"{save_prefix}-rank-{r}.pt")
        else:
            ckpt = load_and_pop_last_optimizer_state(p)
            weights.append(ckpt["model"])
            metadata.append(ckpt["shard_metadata"])
    assert weights, f"all files were considered experts: {all_ckpt_files}"
    do_consolidate = True
    if "decoder.embed_tokens.weight" in weights[0].keys():
        shape = weights[0]["decoder.embed_tokens.weight"].shape
        logger.info(
            f"This ckpt does not seem sharded. I see unflat params! like "
            f"decoder.embed_tokens.weight shaped {shape}. Will just copy files "
            f"and remove optim_state."
        )
        do_consolidate = False
    if do_consolidate:
        num_parts = find_num_parts(names)
        if num_parts:
            logger.info("consolidate_model_parallel")
            consolidated_weights = consolidate_model_parallel(
                metadata,
                names,
                strict,
                weights,
                parts=num_parts,
                no_stitch_megatron=no_stitch_megatron,
            )
        else:
            logger.info("FSDP.consolidate_shard_weights")
            consolidated_weights = FSDP.consolidate_shard_weights(
                shard_weights=weights, shard_metadata=metadata, strict=strict
            )
        del weights, metadata
        gc.collect()
        done_consolidate = time.time()
        logger.info(f"Done consolidating after {done_consolidate-t0//60} minutes")
    else:
        consolidated_weights = weights[0]
    if new_arch_name is not None:
        ckpt["cfg"]["model"]._name = new_arch_name
    if dense:
        logger.info("dense")

        def save_checkpoint(weights_to_save, prefix):
            ckpt_consolidated = dict(
                model=weights_to_save,
                cfg=ckpt["cfg"],
                extra_state=ckpt["extra_state"],
                optimizer_history=ckpt["optimizer_history"],
                args=ckpt.get("args"),
            )
            save_path = f"{prefix}.pt"
            logger.info(f"Saving to {save_path} ...")
            torch.save(ckpt_consolidated, save_path)
            logger.info(f"Done after {time.time()-t0//60} minutes")
            return save_path

        if no_stitch_megatron:
            saved_paths = []
            for part_id, part_consolidated_weights in consolidated_weights.items():
                saved_paths.append(
                    save_checkpoint(
                        part_consolidated_weights, f"{save_prefix}-model_part-{part_id}"
                    )
                )
            return saved_paths
        return save_checkpoint(consolidated_weights, save_prefix)

    ckpt_shared = dict(
        model=consolidated_weights,
        cfg=ckpt["cfg"],
        extra_state=ckpt["extra_state"],
        optimizer_history=ckpt["optimizer_history"],
        args=ckpt["args"],
    )
    logger.info("saving..")
    torch.save(ckpt_shared, f"{save_prefix}-shared.pt")
    logger.info(f"Done saving. Total time: {time.time()-t0//60} minutes,  ")
    # Process experts
    for src, dst in tqdm(
        list(zip(expert_paths, expert_dest_paths)), desc="expert files"
    ):
        ckpt = load_and_pop_last_optimizer_state(src)
        if do_consolidate:
            expert_wt = FSDP.consolidate_shard_weights(
                shard_weights=[ckpt["model"]],
                shard_metadata=[ckpt["shard_metadata"]],
                strict=False,
            )
            ckpt = dict(
                model=expert_wt,
                cfg=ckpt["cfg"],
                extra_state=ckpt["extra_state"],
                optimizer_history=ckpt["optimizer_history"],
                args=ckpt["args"],
            )

        torch.save(ckpt, dst)
    logger.info(f"saved consolidated MoE with prefix {save_prefix}.pt")
    return f"{save_prefix}.pt"


def consolidate_model_parallel(
    metadata, names, strict, weights, parts=2, no_stitch_megatron=False
):
    model_parts = defaultdict(list)
    metadata_parts = defaultdict(list)
    for i, n in enumerate(names):
        for p in range(parts):
            if f"part-{p}" in n:
                model_parts[p].append(weights[i])
                metadata_parts[p].append(metadata[i])
    all_parts_consolidated = defaultdict(list)
    for k, v in model_parts.items():
        part_weights = FSDP.consolidate_shard_weights(
            shard_weights=v, shard_metadata=metadata_parts[k], strict=strict
        )
        all_parts_consolidated[k] = part_weights
    if no_stitch_megatron:
        return all_parts_consolidated
    # glue to be a single megatron mdoel part
    model = reshard_megatron_parts(all_parts_consolidated, new_model_part_count=1)[0]
    return model


def handle_qkv_proj(model_parts, key, new_model_part_count):
    parts = [model_parts[part_id][key] for part_id in range(len(model_parts))]
    ks, vs, qs = [], [], []
    for p in parts:
        k, v, q = torch.split(p, p.shape[0] // 3)
        ks.append(k)
        vs.append(v)
        qs.append(q)
    resharded_ks = torch.chunk(torch.cat(ks, dim=0), new_model_part_count)
    resharded_vs = torch.chunk(torch.cat(vs, dim=0), new_model_part_count)
    resharded_qs = torch.chunk(torch.cat(qs, dim=0), new_model_part_count)
    return resharded_ks, resharded_vs, resharded_qs


def _handle_one(parts, is_weight):
    """Make it look like a normal LayerNorm"""
    n_parts = len(parts)
    err_msg = f"Redundant ModelParallelFusedLayerNorm params have been updated."
    if is_weight:
        init = 1.0
        assert not torch.logical_and(parts[0].ne(1), parts[1].ne(1)).any(), err_msg

    else:
        init = 0.0
        assert not torch.logical_and(parts[0].ne(0), parts[1].ne(0)).any(), err_msg
    ret_val = torch.cat([p.unsqueeze(-1) for p in parts], dim=1).sum(1) - (
        init * (n_parts - 1)
    )
    return ret_val


def get_n_layers(glued_model):
    n_layers = 0
    while True:
        if f"decoder.layers.{n_layers}.fc1.weight" in glued_model:
            n_layers += 1
        else:
            assert (
                n_layers > 0
            ), f"found 0 layers bc no keys matching decoder.layers.0.fc1.weight"
            return n_layers


def reshard_megatron_parts(model_parts, new_model_part_count=1):
    """
    Reshard to different number of model parts.
    When new_model_part_count=1 return glued model
    """
    new_model_parts = [OrderedDict() for _ in range(new_model_part_count)]

    def assert_all_close(key):
        for part_id in range(len(model_parts)):
            if not torch.allclose(model_parts[part_id][key], model_parts[0][key]):
                err = (
                    (model_parts[part_id][key] - model_parts[0][key])
                    .float()
                    .abs()
                    .max()
                    .item()
                )
                logger.info(f"max discrepancy {key}: {err}")

    def _conslidate_and_redshard(key, dim):
        consolidated_tensor = torch.cat(
            [model_parts[part_id][key] for part_id in range(len(model_parts))],
            dim=dim,
        )
        assert consolidated_tensor.size(dim) % new_model_part_count == 0
        newly_resharded_tensors = torch.chunk(
            consolidated_tensor,
            new_model_part_count,
            dim=dim,
        )
        for i in range(new_model_part_count):
            new_model_parts[i][key] = newly_resharded_tensors[i].clone()

    def _copy_key_to_all_parts(key):
        for new_model_part in new_model_parts:
            new_model_part[key] = model_parts[0][key].clone()

    for key in model_parts[0]:
        if "qkv" in key:
            # Bias of CP gets concatenated
            if key.endswith("bias"):
                resharded_ks, resharded_vs, resharded_qs = handle_qkv_proj(
                    model_parts, key, new_model_part_count
                )
            else:
                assert key.endswith("weight")
                resharded_ks, resharded_vs, resharded_qs = handle_qkv_proj(
                    model_parts, key, new_model_part_count
                )

            # Handle the special case when new_model_part_count = 1 (converting to a singleton checkpoint)
            if new_model_part_count == 1:
                new_model_parts[0][key.replace("qkv", "k")] = resharded_ks[0]
                new_model_parts[0][key.replace("qkv", "v")] = resharded_vs[0]
                new_model_parts[0][key.replace("qkv", "q")] = resharded_qs[0]
            else:
                for i in range(new_model_part_count):
                    new_model_parts[i][key] = torch.cat(
                        (resharded_ks[i], resharded_vs[i], resharded_qs[i]), dim=0
                    )

        elif "ffn_layernorm" in key:
            _conslidate_and_redshard(key, dim=0)
        elif "layer_norm" in key:
            assert_all_close(key)
            _copy_key_to_all_parts(key)
        elif "fc1" in key or "k_proj" in key or "q_proj" in key or "v_proj" in key:
            # Bias of CP gets concatenated
            if key.endswith("bias"):
                _conslidate_and_redshard(key, dim=0)
            # weights of CP gets concatenated along dim 0
            else:
                assert key.endswith("weight")
                _conslidate_and_redshard(key, dim=0)
                # FC1 is CP
        # FC2 is RP
        elif "fc2" in key or "out_proj" in key:
            # Bias of RP gets replicated
            if key.endswith("bias"):
                assert_all_close(key)
                _copy_key_to_all_parts(key)
            # weights of RP gets concatenated along dim 1
            else:
                assert key.endswith("weight")
                _conslidate_and_redshard(key, dim=1)
        elif "embed_tokens.weight" in key:
            _conslidate_and_redshard(key, dim=0)
        elif "embed_positions" in key:
            if "_float_tensor" in key:
                # Assume embed positions are non learned ie.e sinusoidal
                for new_model_part in new_model_parts:
                    new_model_part[key] = torch.zeros([1])
            else:
                assert_all_close(key)
                _copy_key_to_all_parts(key)
        elif "version" in key:
            _copy_key_to_all_parts(key)
        else:
            assert_all_close(key)
            _copy_key_to_all_parts(key)

    for new_model_part in new_model_parts:
        assert len(new_model_part.keys()) >= len(model_parts[0].keys())
        assert "decoder.layers.0.ffn_layernorm.lns.0.weight" not in new_model_part

    return new_model_parts


def find_num_parts(names) -> int:
    parts = []
    for n in names:
        part = re.search(r"-model_part-(\d+)", n)
        if part is not None:
            parts.append(int(part.groups()[0]))
    if parts:
        return max(parts) + 1
    else:
        return 0
