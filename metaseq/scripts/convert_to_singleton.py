#!/usr/bin/env python

"""
Script for backing out of the MP-resharded (reshard.pt) files and getting back
a non-flattened state dict.

Particularly useful for converting our models to other repositories.

Usage:
    $ ls 125m
    dict.txt
    gpt2-merges.txt
    gpt2-vocab.json
    reshard-model_part-0.pt
    reshard-model_part-1.pt

    $ python -m metaseq.scripts.convert_to_singleton 125m

    $ ls 125m
    dict.txt
    gpt2-merges.txt
    gpt2-vocab.json
    reshard-model_part-0.pt
    reshard-model_part-1.pt
    restored.pt
"""

import argparse
import glob
import logging
import os
import sys

import torch

from metaseq import options, tasks, checkpoint_utils, utils
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as distributed_utils
from metaseq.distributed import fsdp_enable_wrap, fsdp_wrap
from metaseq.distributed.stitch_fsdp_ckpt import reshard_megatron_parts

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("convert_to_singleton")


def create_generation_config_with_defaults(model_path):
    files = glob.glob(f"{model_path}/reshard*.pt")

    MP = len(files)
    BPE_MERGES = model_path + "/gpt2-merges.txt"
    BPE_VOCAB = model_path + "/gpt2-vocab.json"

    # Skeleton out all the annoying command line args we can infer
    ARGS = [
        "--model-parallel-size",
        str(MP),
        "--distributed-world-size",
        str(MP),
        "--task",
        "language_modeling",
        "--bpe-merges",
        BPE_MERGES,
        "--merges-filename",
        BPE_MERGES,
        "--bpe-vocab",
        BPE_VOCAB,
        "--vocab-filename",
        BPE_VOCAB,
        "--bpe",
        "hf_byte_bpe",
        "--path",
        model_path + "/reshard.pt",
        "--checkpoint-shard-count",
        "1",
        "--use-sharded-state",
        model_path,
    ]
    print(ARGS)

    # build up the config file
    parser = options.get_generation_parser()
    # dumb defaults overriding
    parser.set_defaults(lr_scheduler=None, criterion=None)
    args = options.parse_args_and_arch(parser, input_args=ARGS)
    cfg = convert_namespace_to_omegaconf(args)
    cfg.distributed_training.distributed_world_size = MP

    return cfg


def worker_main(cfg: MetaseqConfig):
    """
    Load up the model on all workers for Model Parallelism, then
    unflatten, move to cpu, and save to "restored.pt".
    """
    task = tasks.setup_task(cfg.task)

    def _build_model(cfg, task):
        cfg.model.tensor_parallel_init_model_on_gpu = True
        model = task.build_model(cfg.model).cuda()
        return fsdp_wrap(model)

    with fsdp_enable_wrap(
        cfg.distributed_training,
        use_sharded_state=cfg.distributed_training.use_sharded_state,
    ):
        models, _model_args, _task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=None,
            task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=True,
            num_shards=cfg.checkpoint.checkpoint_shard_count,
            build_model_hook=_build_model,
        )
        model = models[0]

    # consolidate everything on rank0
    mp_size = distributed_utils.get_model_parallel_world_size()
    model_parts = [{} for _ in range(mp_size)]

    with model.summon_full_params():
        for name, p in model.named_parameters():
            gathered = [torch.zeros_like(p) for _ in range(mp_size)]
            torch.distributed.all_gather(
                gathered, p, group=distributed_utils.get_global_group()
            )
            for r, t in enumerate(gathered):
                model_parts[r][name] = t.cpu()

    glued = reshard_megatron_parts(model_parts, new_model_part_count=1)[0]
    # glued['decoder.output_projection.weight'] = glued['decoder.embed_tokens.weight']

    glued["decoder.version"] = model.state_dict()["decoder.version"].cpu()

    if "decoder.output_projection.weight" in glued:
        del glued["decoder.output_projection.weight"]

    output_sd = checkpoint_utils.load_checkpoint_to_cpu(
        cfg.common_eval.path.replace("reshard.pt", "reshard-model_part-0.pt")
    )
    output_sd["model"] = utils.move_to_cpu(glued)
    output_sd["cfg"]["model"].arch = "transformer_lm"
    output_sd["cfg"]["model"]._name = "transformer_lm"

    if distributed_utils.get_global_rank() == 0:
        with open(cfg.task.data + "/restored.pt", "wb") as f:
            torch.save(output_sd, f)


def main():
    # parser to be used like docstring shows
    real_parser = argparse.ArgumentParser()
    real_parser.add_argument("location")
    args = real_parser.parse_args()

    cfg = create_generation_config_with_defaults(args.location)
    distributed_utils.call_main(cfg, worker_main)


if __name__ == "__main__":
    main()
