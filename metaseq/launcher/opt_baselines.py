#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This sweep script takes some additional optional arguments. See add_extra_options_func
for more details.
"""
import os

from metaseq.launcher.opt_job_constants import (
    TOTAL_TRAIN_TOKENS,
    TOTAL_WARMUP_TOKENS,
    MODEL_SIZES,
    DATA_LOCATIONS,
    VALID_SUBSETS,
)
from metaseq.launcher.sweep import (
    hyperparam,
    get_env_from_args,
    main as sweep_main,
)

# have to do this at the module level, unfortunately; unable to use args.<env>
for _cluster, _folder in DATA_LOCATIONS.items():
    if os.path.exists(_folder):
        try:
            import metaseq_internal  # noqa: F401
            from metaseq_internal.fb_sweep.dependency_checks import *  # noqa
        except ImportError:
            print("\n\nmetaseq_internal not installed! Proceeding...")
            pass
        break


def add_extra_options_func(parser):
    # NOTE we shouldn't add new options here... track changes via git instead
    parser.add_argument(
        "--restore-file", help="load an existing checkpoint for continuing training"
    )
    parser.add_argument(
        "--reset-dataloader",
        action="store_true",
        help="reset the dataloader to epoch 1",
    )
    parser.add_argument("--model-size", choices=MODEL_SIZES.keys(), required=True)
    parser.add_argument(
        "--no-save-dir", action="store_true", help="avoid saving with hparams"
    )

    # Args related to benchmarking and profiling
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="use synthetic data and only train for 50 steps (for benchmarking)",
    )
    parser.add_argument(
        "--profile",
        default=False,
        action="store_true",
    )
    parser.add_argument("--max-update", "--mu", type=int, default=None)


def get_grid(args):
    # Infer data path if not given
    DATA_ROOT = ""
    if args.data is None and not args.benchmark:
        cluster_env = get_env_from_args(args)
        args.data = os.path.join(
            DATA_LOCATIONS[cluster_env], "corpus_dedup_10_10_1_0.05_exp29"
        )
        if os.path.exists(args.data):
            DATA_ROOT = DATA_LOCATIONS[cluster_env]
        else:
            raise RuntimeError("Where are you running this?! Check DATA_LOCATIONS.")

    SEQ_LEN = 2048
    size = MODEL_SIZES[args.model_size]
    # updates = 300B tokens / 2048 seq_len / 1024 batchsize

    total_gpus = args.num_gpus * args.num_nodes

    # TODO: fix training to run with 1 gpu (see Enable sweep scripts to run with a single GPU #176)
    if args.num_gpus < 2:
        raise ValueError("Need at least two gpus to run model parallel code")
    if total_gpus < size.model_parallel:
        raise ValueError(
            "Total gpus (num_gpus * num_nodes) must be great than model parallel factor"
        )
    if total_gpus % size.model_parallel != 0:
        raise ValueError(
            "Total gpus (num_gpus * num_nodes) must be divisible by model parallel factor"
        )

    total_gpus = (args.num_gpus * args.num_nodes) // size.model_parallel
    ddp_bsz = (size.batch_size // total_gpus) // SEQ_LEN
    total_updates = args.max_update
    if total_updates is None:
        total_updates = int(TOTAL_TRAIN_TOKENS) // size.batch_size
    warmup_updates = int(TOTAL_WARMUP_TOKENS) // size.batch_size
    log_interval = 1

    grid = []

    # default streaming_lm task config
    task_config = [
        hyperparam("--task", "streaming_language_modeling"),
        hyperparam(
            "--sample-break-mode",
            "none",
            save_dir_key=lambda val: f"bm_{val}" if not no_save_params else "",
        ),
        hyperparam(
            "--vocab-filename",
            os.path.join(DATA_ROOT, "tokenizers/gpt2-vocab.json"),
            save_dir_key=lambda _: "gpt2" if not no_save_params else "",
        ),
        hyperparam(
            "--merges-filename", os.path.join(DATA_ROOT, "tokenizers/gpt2-merges.txt")
        ),
    ]

    # separate task config for dummy_lm
    if args.benchmark:
        # Overrides for speed benchmarking
        args.data = None
        task_config = [
            hyperparam("--task", "dummy_lm", save_dir_key=lambda val: val),
            hyperparam(
                "--dict-size", 51200 - 4
            ),  # TODO(susan): what is this -4 sorcery? relic of more nmt things?
        ]
        total_updates = 50
        warmup_updates = 50
        log_interval = 5

    grid += task_config

    if args.profile:
        grid += [hyperparam("--new-profiler")]

    no_save_params = args.no_save_dir
    args.snapshot_code = True
    grid += [
        hyperparam("--train-subset", "train"),
        hyperparam("--valid-subset", ",".join(f"valid/{ss}" for ss in VALID_SUBSETS)),
        hyperparam("--ignore-unused-valid-subsets"),
        hyperparam("--num-workers", 8),
        hyperparam("--num-workers-valid", 1),
        hyperparam("--validate-interval-updates", 2000),
        hyperparam("--save-interval-updates", 2000),
        hyperparam(
            "--no-epoch-checkpoints"
        ),  # only save checkpoints based on num steps
        hyperparam("--no-best-checkpoints"),  # don't save checkpoint_best.pt
        hyperparam(
            "--memory-efficient-fp16",
            save_dir_key=lambda val: "me_fp16" if not no_save_params else "",
        ),
        hyperparam("--fp16-init-scale", 4),
        # we set this for the main run but it's probably nt needed here
        # hyperparam("--threshold-loss-scale", 0.25, save_dir_key=lambda val: f"minscale{val}"),
        hyperparam(
            "--ddp-backend",
            "fully_sharded",
            save_dir_key=lambda val: "fsdp" if not no_save_params else "",
        ),
        hyperparam("--no-reshard-after-forward", save_dir_key=lambda _: "zero2"),
        hyperparam("--use-sharded-state"),
        hyperparam("--checkpoint-activations"),
        hyperparam("--model-parallel-size", size.model_parallel),
        hyperparam("--criterion", "vocab_parallel_cross_entropy"),
        hyperparam("--distribute-checkpointed-activations"),
        hyperparam("--tensor-parallel-init-model-on-gpu"),
        # Flags to match exact same initialization of Megatron code for exp 12.00
        hyperparam("--full-megatron-init"),
        hyperparam("--megatron-init-sigma", 0.006),
        hyperparam(
            "--activation-fn",
            "relu",
            save_dir_key=lambda x: x if not no_save_params else "",
        ),
        hyperparam(
            "--arch",
            "transformer_lm_megatron",
            save_dir_key=lambda val: val if not no_save_params else "",
        ),
        hyperparam("--share-decoder-input-output-embed"),
        hyperparam(
            "--decoder-layers",
            size.n_layers,
            save_dir_key=lambda val: f"nlay{val}" if not no_save_params else "",
        ),
        hyperparam(
            "--decoder-embed-dim",
            size.emb_size,
            save_dir_key=lambda val: f"emb{val}" if not no_save_params else "",
        ),
        hyperparam("--decoder-ffn-embed-dim", size.ffn_size),
        hyperparam("--decoder-attention-heads", size.n_heads),
        # Switch to learned position embeddings for exp 12.00, without scaling
        hyperparam(
            "--decoder-learned-pos",
            save_dir_key=lambda _: "lrnpos" if not no_save_params else "",
        ),
        hyperparam(
            "--no-scale-embedding",
            save_dir_key=lambda _: "0emb_scale" if not no_save_params else "",
        ),
        hyperparam(
            "--tokens-per-sample",
            SEQ_LEN,
            save_dir_key=lambda val: f"tps{val}" if not no_save_params else "",
        ),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        # GPT-3 uses "(0.9, 0.95)"
        hyperparam(
            "--adam-betas",
            f"(0.9, 0.95)",
            save_dir_key=lambda val: "b2_{}".format(eval(val)[1])
            if not no_save_params
            else "",
        ),
        # Sometimes lowering --adam-eps to 1e-6 can stabilize training
        hyperparam(
            "--adam-eps",
            1e-8,
            save_dir_key=lambda val: f"eps{val}" if not no_save_params else "",
        ),
        # GPT-3 used --clip-norm=1.0
        hyperparam(
            "--clip-norm",
            1.0,
            save_dir_key=lambda val: f"cl{val}" if not no_save_params else "",
        ),
        hyperparam("--clip-norm-type", "l2"),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam(
            "--lr",
            size.lr,
            save_dir_key=lambda val: f"lr{val:.3g}" if not no_save_params else "",
        ),
        hyperparam(
            "--end-learning-rate",
            size.lr * 0.1,
            save_dir_key=lambda val: f"endlr{val:.3g}" if not no_save_params else "",
        ),
        hyperparam(
            "--warmup-updates",
            warmup_updates,
            save_dir_key=lambda val: f"wu{val}" if not no_save_params else "",
        ),
        hyperparam("--total-num-update", total_updates),
        hyperparam(
            "--dropout",
            0.1,
            save_dir_key=lambda val: f"dr{val}" if not no_save_params else "",
        ),
        hyperparam(
            "--attention-dropout",
            0.1,
            save_dir_key=lambda val: f"atdr{val}" if not no_save_params else "",
        ),
        hyperparam(
            "--no-emb-dropout",
            save_dir_key=lambda _: "0emb_dr" if not no_save_params else "",
        ),
        hyperparam(
            "--weight-decay",
            0.1,
            save_dir_key=lambda val: f"wd{val}" if not no_save_params else "",
        ),
        hyperparam(
            "--batch-size",
            ddp_bsz,
            save_dir_key=lambda val: f"ms{val}" if not no_save_params else "",
        ),
        hyperparam(
            "--update-freq",
            1,
            save_dir_key=lambda val: f"uf{val}" if not no_save_params else "",
        ),
        hyperparam(
            "--max-update",
            total_updates,
            save_dir_key=lambda val: f"mu{val}" if not no_save_params else "",
        ),
        hyperparam(
            "--seed",
            1,
            save_dir_key=lambda val: f"s{val}" if not no_save_params else "",
        ),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", log_interval),
        hyperparam("--required-batch-size-multiple", 1),
    ]
    if args.restore_file:
        grid += [hyperparam("--restore-file", args.restore_file)]
    if args.reset_dataloader:
        grid += [hyperparam("--reset-dataloader")]

    return grid


def postprocess_hyperparams(args, config):
    pass


def cli_main():
    sweep_main(
        get_grid, postprocess_hyperparams, add_extra_options_func=add_extra_options_func
    )


if __name__ == "__main__":
    cli_main()
