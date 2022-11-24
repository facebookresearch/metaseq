# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import collections
import functools
import logging
import os
import re
import traceback
import socket
from typing import Any, Dict, List, Optional, Tuple
import shutil

import torch
from omegaconf import OmegaConf

from metaseq.dataclass.configs import CheckpointConfig
from metaseq.dataclass.utils import overwrite_args_by_name
from metaseq.distributed import utils as distributed_utils
from metaseq.file_io import PathManager, torch_load_cpu
from metaseq.launcher.opt_job_constants import ComputeEnvs

logger = logging.getLogger(__name__)


OPT_KEY = "last_optimizer_state"


def save_checkpoint(
    cfg: CheckpointConfig,
    trainer,
    epoch_itr,
    training_finished=False,
    async_callback_fn=None,
):
    from metaseq import meters

    # only one worker should attempt to create the required dir
    if trainer.data_parallel_rank == 0:
        os.makedirs(cfg.save_dir, exist_ok=True)

    trainer.consolidate_optimizer()

    if not trainer.should_save_checkpoint_on_current_rank:
        return

    write_timer = meters.StopwatchMeter()
    write_timer.start()

    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    logger.info(f"Preparing to save checkpoint for epoch {epoch} @ {updates} updates")

    def is_better(a, b):
        return a >= b if cfg.maximize_best_checkpoint_metric else a <= b

    suffix = trainer.checkpoint_suffix
    checkpoint_conds = collections.OrderedDict()

    save_for_epoch = (
        end_of_epoch
        and cfg.save_interval_epochs > 0
        and epoch % cfg.save_interval_epochs == 0
    )

    save_for_updates = (
        not end_of_epoch
        and cfg.save_interval_updates > 0
        and updates % cfg.save_interval_updates == 0
    )

    checkpoint_conds[f"checkpoint{epoch}{suffix}.pt"] = save_for_epoch
    checkpoint_conds[f"checkpoint_{updates}{suffix}.pt"] = save_for_updates
    checkpoint_conds[f"checkpoint_last{suffix}.pt"] = (
        training_finished and cfg.save_last_checkpoint
    )

    extra_state = {"train_iterator": epoch_itr.state_dict()}

    checkpoints = [
        os.path.join(cfg.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond
    ]
    if len(checkpoints) > 0:
        if PathManager.islink(checkpoints[0]):
            PathManager.rm(checkpoints[0])

        trainer.save_checkpoint(
            checkpoints[0],
            extra_state,
            training_finished=training_finished,
            async_callback_fn=async_callback_fn,
        )

        def _copy_if_not_async(src, dest):
            if cfg.write_checkpoints_asynchronously:
                pass  # TODO[file_io]: Need to implement a delayed asynchronous file copying/moving feature.
            else:
                assert PathManager.copy(
                    src, dest, overwrite=True
                ), f"Failed to copy {src} to {dest}"

        for cp in checkpoints[1:]:
            _copy_if_not_async(src=checkpoints[0], dest=cp)

        write_timer.stop()
        logger.info(
            f"Saved checkpoint {checkpoints[0]} (epoch {epoch} @ {updates} updates) "
            f"(writing took {write_timer.sum} seconds)"
        )

        _delete_old_checkpoint_files(
            cfg,
            end_of_epoch,
            suffix,
        )


def _delete_old_checkpoint_files(
    cfg: CheckpointConfig, end_of_epoch: bool, suffix: str
):
    if not end_of_epoch and cfg.keep_last_updates > 0:
        suffixes = [suffix]

        # remove old checkpoints; checkpoints are sorted in descending order
        for one_suffix in suffixes:
            checkpoints = _checkpoint_paths(
                cfg.save_dir, pattern=r"checkpoint_(\d+){}\.pt".format(one_suffix)
            )
            for old_chk in checkpoints[cfg.keep_last_updates :]:
                if os.path.lexists(old_chk):
                    os.remove(old_chk)
    if cfg.keep_last_epochs > 0:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = _checkpoint_paths(
            cfg.save_dir, pattern=r"checkpoint(\d+){}\.pt".format(suffix)
        )
        for old_chk in checkpoints[cfg.keep_last_epochs :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)


def load_checkpoint(cfg: CheckpointConfig, trainer, **passthrough_args):
    """
    Load a checkpoint and restore the training iterator.

    *passthrough_args* will be passed through to
    ``trainer.get_train_iterator``.
    """

    reset_optimizer = cfg.reset_optimizer
    reset_lr_scheduler = cfg.reset_lr_scheduler
    optimizer_overrides = ast.literal_eval(cfg.optimizer_overrides)
    reset_meters = cfg.reset_meters
    reset_dataloader = cfg.reset_dataloader

    if cfg.finetune_from_model is not None and (
        reset_optimizer or reset_lr_scheduler or reset_meters or reset_dataloader
    ):
        raise ValueError(
            "--finetune-from-model can not be set together with either --reset-optimizer"
            " or reset_lr_scheduler or reset_meters or reset_dataloader"
        )

    suffix = trainer.checkpoint_suffix
    default_restore_file = "checkpoint_last.pt"
    # default to loading from restore file.
    if cfg.restore_file == default_restore_file:
        checkpoint_path_to_load = os.path.join(
            cfg.save_dir, "checkpoint_last{}.pt".format(suffix)
        )
        first_launch = not PathManager.exists(checkpoint_path_to_load)
        if cfg.finetune_from_model is not None and first_launch:
            # if there is no last checkpoint to restore, start the finetune from pretrained model
            # else just use usual logic to load checkpoint, e.g. restart from last checkpoint and etc.
            reset_optimizer = True
            reset_lr_scheduler = True
            reset_meters = True
            reset_dataloader = True
            checkpoint_path_to_load = None
            if PathManager.exists(cfg.finetune_from_model):
                checkpoint_path_to_load = cfg.finetune_from_model
            elif suffix is not None:  # check for sharded version
                sharded_path = cfg.finetune_from_model.replace(".pt", suffix + ".pt")
                if PathManager.exists(sharded_path):
                    checkpoint_path_to_load = sharded_path
            if checkpoint_path_to_load is None:
                raise ValueError(
                    f"--finetune-from-model {cfg.finetune_from_model} does not exist either as is or sharded"
                )

            logger.info(
                f"loading pretrained model from {checkpoint_path_to_load}: "
                "optimizer, lr scheduler, meters, dataloader will be reset"
            )
    elif suffix is not None:
        checkpoint_path_to_load = cfg.restore_file.replace(".pt", suffix + ".pt")
    else:
        checkpoint_path_to_load = cfg.restore_file

    if cfg.restore_file != default_restore_file and cfg.finetune_from_model:
        raise ValueError(
            "--finetune-from-model and --restore-file (non-default value) "
            "can not be specified together: " + str(cfg)
        )

    # Azure logic
    try:
        from metaseq_internal import azure_utils

        has_metaseq_internal = True
    except ImportError:
        has_metaseq_internal = False
        logger.warning(
            "Proceeding without metaseq-internal installed! Please check if you need this!"
        )

    # TODO(susanz): fix all of this spagetti, split out logic by env
    # Note that we compare by value since ComputeEnvs may be imported from metaseq_internal

    if cfg.cloud_upload_path:
        if cfg.cloud_upload_path.startswith("nfs:"):
            checkpoint_path_to_load = os.path.join(
                cfg.save_dir, "checkpoint_last{}.pt".format(suffix)
            )
            nfs_path = cfg.cloud_upload_path[4:]
            filename = None
            specific_restore_file_provided = cfg.restore_file != default_restore_file
            slurm_was_restarted = int(os.environ.get("SLURM_RESTART_COUNT", 0)) > 0
            restart_from_latest = slurm_was_restarted or (
                cfg.finetune_from_model is None and not specific_restore_file_provided
            )
            if restart_from_latest:
                checkpoints = []
                expected_file_count = distributed_utils.get_global_world_size()
                for candidate in os.listdir(nfs_path):
                    if candidate == "checkpoint_last":
                        raise RuntimeError(
                            "trying to restart a job that already wrote checkpoint_last"
                        )
                    m = re.match(r"checkpoint_(\d+)", candidate)
                    if m:
                        checkpoints.append((int(m[1]), candidate))
                for _, candidate in sorted(checkpoints, reverse=True):
                    present_files = len(
                        [
                            f
                            for f in os.listdir(os.path.join(nfs_path, candidate))
                            if not f.startswith("_")
                        ]
                    )
                    if present_files == expected_file_count:
                        filename = os.path.join(
                            nfs_path, candidate, f"checkpoint{suffix}.pt"
                        )
                        break
                    logger.info(
                        f"skipping checkpoint {candidate} because it only has"
                        f" {present_files} files (expected {expected_file_count})"
                    )
            else:
                filename = cfg.restore_file.replace(".pt", suffix + ".pt")
            if filename is not None:
                logger.info(
                    f"Copying checkpoint from nfs {filename} -> {checkpoint_path_to_load}"
                )
                shutil.copyfile(filename, checkpoint_path_to_load)
            else:
                logger.info(f"No NFS checkpoints found")

        elif cfg.cluster_env == ComputeEnvs.AZURE.value and has_metaseq_internal:
            if (
                # --restore-file was not passed, always download latest checkpoint
                (
                    cfg.restore_file == default_restore_file
                    and cfg.finetune_from_model is None
                )
                # --restore-file was passed, but we requeued, so download latest checkpoint
                or int(os.environ.get("SLURM_RESTART_COUNT", 0)) > 0
            ):
                # download checkpoint into local save_dir
                checkpoint_path_to_load = os.path.join(
                    cfg.save_dir, "checkpoint_last{}.pt".format(suffix)
                )
                azure_utils.download_recent_ckpt(
                    cfg.cloud_upload_path, checkpoint_path_to_load, suffix + ".pt"
                )
            elif (
                # --restore-file was passed and is a blob URL, download that checkpoint
                cfg.restore_file != default_restore_file
                and "windows.net" in cfg.restore_file
            ):
                blob_url = cfg.restore_file.replace(".pt", suffix + ".pt")
                # download checkpoint into local save_dir
                checkpoint_path_to_load = os.path.join(
                    cfg.save_dir, "checkpoint_last{}.pt".format(suffix)
                )
                azure_utils.download_specific_ckpt(blob_url, checkpoint_path_to_load)
            else:
                logger.info(
                    f"Using checkpoint {checkpoint_path_to_load} even while on Azure"
                )

    # RSC logic: --restore-file was passed, and we requeued
    elif (
        cfg.restore_file != default_restore_file
        and int(os.environ.get("SLURM_RESTART_COUNT", 0)) > 0
    ):
        # point checkpoint_path to the current checkpoint directory for loading, if it exists.
        save_dir_last = os.path.join(
            cfg.save_dir, "checkpoint_last{}.pt".format(suffix)
        )
        if PathManager.isfile(save_dir_last):
            checkpoint_path_to_load = save_dir_last

    logger.info(f"attempting to load checkpoint from: {checkpoint_path_to_load}")

    # make sure everyone is done downloading their checkpoints before we load
    distributed_utils.global_barrier()

    extra_state = trainer.load_checkpoint(
        checkpoint_path_to_load,
        reset_optimizer,
        reset_lr_scheduler,
        optimizer_overrides,
        reset_meters=reset_meters,
    )

    if reset_dataloader and int(os.environ.get("SLURM_RESTART_COUNT", 0)) > 0:
        logger.info(
            f"Disregarding --reset-dataloader since we are continuing past a requeue"
        )
        reset_dataloader = False
    if extra_state is not None and not reset_dataloader:
        # restore iterator from checkpoint
        itr_state = extra_state["train_iterator"]
        epoch_itr = trainer.get_train_iterator(
            epoch=itr_state["epoch"], **passthrough_args
        )
        epoch_itr.load_state_dict(itr_state)
    else:
        epoch_itr = trainer.get_train_iterator(epoch=1, **passthrough_args)
    trainer.lr_step(epoch_itr.epoch)
    return extra_state, epoch_itr


def _is_checkpoint_sharded(checkpoint_files) -> bool:
    """
    Infer if state is sharded based on recorded configuration in the checkpoint file.
    """
    if not checkpoint_files:
        raise FileNotFoundError(
            "We weren't able to find any checkpoints corresponding to the parameters "
            "you set. This could mean you have a typo, or it could mean you have a "
            "mismatch in distributed training parameters, especially --fsdp or"
            "--model-parallel. If you are working on a new script, it may also mean "
            "you failed to fsdp_wrap or you have an unnecessary fsdp_wrap."
        )
    sd = torch_load_cpu(checkpoint_files[0])
    return sd["cfg"]["distributed_training"]["use_sharded_state"]


def _cache_checkpoint_files(path: str, suffix: str) -> Tuple[str, List[str]]:
    """
    Given a checkpoint path, first try to expand it according to
    a specified filename pattern. If `path` doesn't match the pattern
    we search the remote for it as-is. Then cache all matching files
    and return the local paths.

    Ex. 1:
        Remote: [path://checkpoint_last-shard0.pt, path://checkpoint_last-shard1.pt]
        Input: path=remote://path/checkpoint_last-shard0.pt, suffix=shard
        Output: [/local/checkpoint_last-shard0.pt, /local/checkpoint_last-shard1.pt]

    Ex. 2:
        Remote: [path://checkpoint_last-model_part-0.pt, path://checkpoint_last-model_part-1.pt]
        Input: path=remote://path/checkpoint_last-model_part-0.pt, suffix=shard
        Output: [/local/checkpoint_last-model_part-0.pt]

    Ex. 3:
        Remote: [path://checkpoint_last.pt]
        Input: path=remote://path/checkpoint_last-shard0.pt, suffix=shard
        Output: []
    """

    local_path = PathManager.get_local_path(path, force=True)
    local_name = os.path.basename(local_path)

    # Ex. 2: if `path` doesn't match the pattern suggested by suffix
    # just return the path itself without expansion
    path_prefix = re.sub(rf"{suffix}[0-9]+\.pt$", f"{suffix}*", path)
    if path == path_prefix:
        return local_path, [local_path]

    local_paths = []
    for filepath in PathManager.ls(path_prefix):
        # Ignore non-checkpoint files
        if not filepath.endswith(".pt"):
            continue
        src_name = os.path.basename(filepath)
        if src_name == local_name:
            # Target path is already cached
            local_paths.append(local_path)
        else:
            src_path = os.path.join(os.path.dirname(path), src_name)
            tgt_path = PathManager.get_local_path(src_path, force=True)
            local_paths.append(tgt_path)

    return local_path, local_paths


def get_paths_to_load(path, suffix="rank-"):
    local_path, checkpoint_files = _cache_checkpoint_files(path, suffix)
    checkpoint_files_count = len(checkpoint_files)

    # Check if this looks like a sharded checkpoint
    if not _is_checkpoint_sharded(checkpoint_files):
        return [local_path]

    # Check if we need to merge shards
    world_size = distributed_utils.get_data_parallel_world_size()
    if world_size >= checkpoint_files_count:
        return [local_path]

    # Assign an equal number of shards
    assert checkpoint_files_count % world_size == 0
    n_local_files = int(checkpoint_files_count / world_size)
    rank = distributed_utils.get_data_parallel_rank()
    start_rank = n_local_files * rank
    fnames = []
    for rank_to_load in range(start_rank, start_rank + n_local_files):
        fname = re.sub(
            f"{suffix}[0-9]+",
            f"{suffix}{rank_to_load}",
            checkpoint_files[0],
        )
        fnames.append(fname)
    logger.info(
        f"Loading {checkpoint_files_count} on {world_size} DDP workers: {n_local_files} files per worker."
    )

    return fnames


def load_checkpoint_to_cpu(path, arg_overrides=None, load_on_all_ranks=False) -> dict:
    """Loads a checkpoint to CPU (with upgrading for backward compatibility).

    If doing single-GPU training or if the checkpoint is only being loaded by at
    most one process on each node (current default behavior is for only rank 0
    to read the checkpoint from disk), load_on_all_ranks should be False to
    avoid errors from torch.distributed not having been initialized or
    torch.distributed.barrier() hanging.

    If all processes on each node may be loading the checkpoint
    simultaneously, load_on_all_ranks should be set to True to avoid I/O
    conflicts.

    There's currently no support for > 1 but < all processes loading the
    checkpoint on each node.
    """

    # Expand multi-part checkpoints like "checkpoint_last-shard0.pt"
    paths_to_load = get_paths_to_load(path, suffix="shard")

    try:
        if len(paths_to_load) > 1:
            state = _merge_flat_fsdp_shards([torch_load_cpu(f) for f in paths_to_load])
        else:
            state = torch_load_cpu(paths_to_load[0])
    except Exception as error:
        logger.error(
            f"Got Exception While Trying To Load {path} with Paths to Load {paths_to_load}."
            "If you are not meaning to --restore-file, remove the command explictly."
        )
        raise error

    logger.info("Done reading from disk")

    if "cfg" in state and state["cfg"] is not None:
        # hack to be able to set Namespace in dict config. this should be removed when we update to newer
        # omegaconf version that supports object flags, or when we migrate all existing models
        from omegaconf import _utils

        old_primitive = _utils.is_primitive_type
        _utils.is_primitive_type = lambda _: True

        state["cfg"] = OmegaConf.create(state["cfg"])

        _utils.is_primitive_type = old_primitive

        OmegaConf.set_struct(state["cfg"], True)

        if arg_overrides is not None:
            overwrite_args_by_name(state["cfg"], arg_overrides)

    state = _upgrade_state_dict(state)
    return state


def load_model_ensemble_and_task(
    filenames,
    arg_overrides: Optional[Dict[str, Any]] = None,
    task=None,
    strict=True,
    suffix="",
    num_shards=1,
    state=None,
    build_model_hook=None,
):
    assert state is None or len(filenames) == 1

    from metaseq import tasks

    assert not (
        strict and num_shards > 1
    ), "Cannot load state dict with strict=True and checkpoint shards > 1"
    ensemble = []
    cfg = None

    for filename in filenames:
        orig_filename = filename
        assert num_shards > 0
        for shard_idx in range(num_shards):
            if num_shards == 1:
                filename = filename.replace(".pt", suffix + ".pt")
            else:
                filename = orig_filename[:-3] + f"_part{shard_idx}.pt"
            if state is None:
                state = load_checkpoint_to_cpu(filename, arg_overrides)
            if "cfg" in state and state["cfg"] is not None:
                cfg = state["cfg"]
            else:
                raise RuntimeError(
                    f"!!! cfg does not exist in state keys = {state.keys()} !!!"
                )

            # Load 175B model trained on megatron (model parallel) branch
            # "cfg.common.model_parallel_size == 1" checks if model parallel is
            # enabled at load time. If it's not, fall back to non-MP
            # transformer code path.
            if (
                getattr(cfg.model, "arch", None) == "transformer_lm_megatron"
                and cfg.common.model_parallel_size == 1
            ):
                cfg.model.arch = "transformer_lm_gpt"
                cfg.model._name = "transformer_lm_gpt"

            # We now copy embed_tokens over to output_proj (if its missing) for all arches (only OPT here so far).
            oproj_key = "decoder.output_projection.weight"
            emb_key = "decoder.embed_tokens.weight"
            if emb_key in state["model"] and oproj_key not in state["model"]:
                state["model"][oproj_key] = state["model"][emb_key]

            if task is None:
                task = tasks.setup_task(cfg.task)

            if "task_state" in state:
                task.load_state_dict(state["task_state"])

            if build_model_hook is not None:
                model = build_model_hook(cfg, task)
            else:
                # build model for ensemble
                model = task.build_model(cfg.model)

            model.load_state_dict(state["model"], strict=strict)
            logger.info("Done loading state dict")
            # reset state so it gets loaded for the next model in ensemble
            state = None

        ensemble.append(model)
    return ensemble, cfg, task


def _checkpoint_paths(path, pattern=r"checkpoint(\d+)\.pt"):
    """Retrieves all checkpoints found in `path` directory.

    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order.
    """
    pt_regexp = re.compile(pattern)
    files = os.listdir(path)

    entries = []
    for i, f in enumerate(files):
        m = pt_regexp.fullmatch(f)
        if m is not None:
            idx = float(m.group(1)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]


def torch_persistent_save(
    obj, filename: str, async_write: bool = False, async_callback_fn=None
):
    assert (
        async_callback_fn is None or async_write
    ), "async_callback_fn requires async_write=True (--save-async)"
    if async_write and async_callback_fn is not None:
        callback = functools.partial(async_callback_fn, filename)
    else:
        callback = None
    if async_write:
        with PathManager.opena(filename, "wb", callback_after_file_close=callback) as f:
            _torch_persistent_save(obj, f)
    else:
        if PathManager.supports_rename(filename):
            # do atomic save
            with PathManager.open(filename + ".tmp", "wb") as f:
                _torch_persistent_save(obj, f)
            PathManager.rename(filename + ".tmp", filename)
        else:
            # fallback to non-atomic save
            with PathManager.open(filename, "wb") as f:
                _torch_persistent_save(obj, f)


def _torch_persistent_save(obj, f, num_retries=3):
    if isinstance(f, str):
        with PathManager.open(f, "wb") as h:
            torch_persistent_save(obj, h)
        return
    for i in range(num_retries):
        try:
            return torch.save(obj, f)
        except Exception:
            if i == num_retries - 1:
                logger.error(traceback.format_exc())


def _upgrade_state_dict(state):
    """Helper for upgrading old model checkpoints."""
    # add optimizer_history
    if "optimizer_history" not in state:
        state["optimizer_history"] = [
            {"criterion_name": "CrossEntropyCriterion", "best_loss": state["best_loss"]}
        ]
        state["last_optimizer_state"] = state["optimizer"]
        del state["optimizer"]
        del state["best_loss"]
    # move extra_state into sub-dictionary
    if "epoch" in state and "extra_state" not in state:
        state["extra_state"] = {
            "epoch": state["epoch"],
            "batch_offset": state["batch_offset"],
            "val_loss": state["val_loss"],
        }
        del state["epoch"]
        del state["batch_offset"]
        del state["val_loss"]
    # reduce optimizer history's memory usage (only keep the last state)
    if "optimizer" in state["optimizer_history"][-1]:
        state["last_optimizer_state"] = state["optimizer_history"][-1]["optimizer"]
        for optim_hist in state["optimizer_history"]:
            del optim_hist["optimizer"]
    # move best_loss into lr_scheduler_state
    if "lr_scheduler_state" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["lr_scheduler_state"] = {
            "best": state["optimizer_history"][-1]["best_loss"]
        }
        del state["optimizer_history"][-1]["best_loss"]
    # keep track of number of updates
    if "num_updates" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["num_updates"] = 0
    # use stateful training data iterator
    if "train_iterator" not in state["extra_state"]:
        state["extra_state"]["train_iterator"] = {
            "epoch": state["extra_state"]["epoch"],
            "iterations_in_epoch": state["extra_state"].get("batch_offset", 0),
        }
    return state


def verify_checkpoint_directory(save_dir: str) -> None:
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Unable to create dir {save_dir} on {socket.gethostname()}")
            raise e
    rank = distributed_utils.get_global_rank()
    temp_file_path = os.path.join(save_dir, f"dummy{rank}")
    try:
        with open(temp_file_path, "w"):
            pass
    except OSError as e:
        logger.warning(f"Unable to access checkpoint save directory: {save_dir}")
        raise e
    else:
        try:
            os.remove(temp_file_path)
        except FileNotFoundError:
            pass


def _merge_flat_fsdp_shards(shards_to_load: List[Dict], unpad=False) -> Dict:
    """
    Concatenate tensor entries in a list of local_state_dicts into one
    local_state_dict to allow resumption on a different world size.
    """
    merged_state = {}
    world_size = distributed_utils.get_data_parallel_world_size()
    for key in shards_to_load[0].keys():
        merged_state[key] = shards_to_load[0][key]

    pad_info = _get_pad_info(shards_to_load[-1])
    dtype = torch.float16
    for k in shards_to_load[0]["model"]:
        dtype = shards_to_load[0]["model"][k].dtype
        if "flat_param" in k:
            pad_info_k = pad_info[k]
            catted = torch.cat([x["model"][k] for x in shards_to_load])
            if world_size == 1 and pad_info_k > 0:
                catted = catted[:-pad_info_k]
            elif world_size > 1 and pad_info_k > 0 and not unpad:
                raise NotImplementedError(
                    f"Param {k} padded with {pad_info_k} extra elements. You must use the reshard_mp script."
                )

            merged_state["model"][k] = catted

    # TODO(susanz): Not removing decoder.version due to HF compatibility.
    if "decoder.version" not in merged_state["model"]:
        merged_state["model"]["decoder.version"] = torch.tensor([3.0], dtype=dtype)
    if OPT_KEY in merged_state:
        merged_state[OPT_KEY] = _merge_flat_fsdp_opt_state(shards_to_load)
    return merged_state


def _merge_flat_fsdp_opt_state(shards_to_load: List[Dict]) -> Dict:
    """Logic described here: https://tinyurl.com/2p86zffr"""
    result = shards_to_load[0][OPT_KEY]
    pad_info = _get_pad_info(shards_to_load[-1])
    world_size = distributed_utils.get_data_parallel_world_size()
    os2model_key = dict(
        zip(shards_to_load[0][OPT_KEY]["state"].keys(), pad_info.keys())
    )
    for k in shards_to_load[0][OPT_KEY]["state"].keys():
        # 0,1,2,3... if each layer wrapped, else 0
        for k2 in shards_to_load[0][OPT_KEY]["state"][k].keys():
            # exp_avg, exp_avg_sq, step (for adam32 bit)
            states = [x[OPT_KEY]["state"][k][k2] for x in shards_to_load]
            if not torch.is_tensor(states[0]) or is_singleton_tensor(states[0]):
                result["state"][k][k2] = states[0]
            else:
                catted = torch.cat(states)
                if k in os2model_key:
                    opt_state_key = os2model_key[k]
                    pad_info_k = pad_info[opt_state_key]
                    if world_size == 1 and pad_info_k > 0:  # unpad
                        catted = catted[:-pad_info_k]
                result["state"][k][k2] = catted
    return result


def is_singleton_tensor(x: Any) -> bool:
    """Is x a dimensionless tensor?"""
    return torch.is_tensor(x) and x.dim() == 0


def _get_pad_info(state_dict: Dict) -> Dict[str, int]:
    if "shard_metadata" not in state_dict:
        # Note: comment this out if you have sharded checkpoints that you think can be loaded
        return collections.defaultdict(lambda: 0)
    res = {}
    for m in state_dict["shard_metadata"]["param_metadata"]:
        fsdp_path = m["fsdp_path"]
        for k, v in m["params"].items():
            full_key = f"{fsdp_path}.{k}" if fsdp_path else k
            assert full_key not in res, f"collision: {full_key} already in {res}"
            res[full_key] = v["padding"]
    return res
