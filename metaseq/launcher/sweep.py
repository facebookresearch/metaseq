# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import os
import subprocess
from typing import Optional, List, Callable, MutableMapping
from urllib.parse import urlparse

from metaseq.launcher.opt_job_constants import ComputeEnvs


class hyperparam(object):
    """Base class for defining hyperparameters."""

    def __init__(
        self,
        name,
        values=None,
        binary_flag=False,
        save_dir_key=None,
        positional_arg=False,
    ):
        """
        Arguments:
        - name : the name of the hyperparameter (e.g., `--dropout`)
        - values : the set of values to sweep over (e.g., `[0.0, 0.1, 0.2]`)
        - binary_flag : whether the hyperparameter uses a boolean flag (e.g., `--no-save`)
        - save_dir_key : function that takes the hyperparameter value and returns the "key"
                         to be appended to the output directory name
        - positional_arg : whether the hyperparameter is a positional argument
        """
        self.name = name
        if values is None:  # syntactic sugar for binary flags
            self.values = [True]
            self.binary_flag = True
        else:
            self.values = values if isinstance(values, list) else [values]
            self.binary_flag = binary_flag
        self.save_dir_key = save_dir_key
        self.positional_arg = positional_arg
        self.current_value = None

        if positional_arg and name.startswith("-"):
            raise ValueError(
                f"positional arguments must not start with a dash ({name})"
            )

        if len(self.values) > 1 and self.save_dir_key is None:
            raise ValueError(
                f"{name} has more than one value but is missing a save_dir_key!"
            )

    def get_cli_args(self):
        if self.binary_flag:
            return [self.name] if self.current_value else []
        elif self.positional_arg:
            return [self.current_value]
        else:
            return [self.name, self.current_value]

    def get_save_dir_key(self):
        if self.save_dir_key is None:
            return None
        if self.binary_flag:
            return self.save_dir_key(1) if self.current_value else None
        return self.save_dir_key(self.current_value)


def get_env_from_args(args):
    # Returns a ComputeEnvs enum.
    if args.azure:
        return ComputeEnvs.AZURE
    elif args.aws:
        return ComputeEnvs.AWS
    elif args.fair:
        return ComputeEnvs.FAIR
    else:
        raise NotImplementedError(
            "Env not passed in! Please pass in one of: --azure, --aws, --fair"
        )


def _get_args(add_extra_options_func=None, input_args: Optional[List[str]] = None):
    """
    input_args (List[str]): strings to parse, defaults to sys.argv
    """
    parser = argparse.ArgumentParser("Script for launching hyperparameter sweeps ")
    parser.add_argument("--grid", help="grid function we used", default=None)

    parser.add_argument("-d", "--data", help="path to data directory")
    parser.add_argument(
        "-p",
        "--prefix",
        required=True,
        help="save checkpoints and logs in <checkpoints-dir>/<prefix>.<save_dir_key>",
    )
    parser.add_argument(
        "-t",
        "--num-trials",
        default=-1,
        type=int,
        help="number of random hyperparam configurations to try (-1 for grid search)",
    )
    parser.add_argument(
        "-g", "--num-gpus", type=int, required=True, help="number of GPUs per node"
    )
    parser.add_argument(
        "-n",
        "--num-nodes",
        type=int,
        default=1,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--update-freq",
        type=int,
        default=0,
        help="update freq",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--resume-failed",
        action="store_true",
        help="resume any runs that failed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="output only a list of actions to perform without performing them",
    )
    parser.add_argument("--local", action="store_true", help="run job locally")
    parser.add_argument("--debug", action="store_true", help="debug")
    parser.add_argument(
        "--script", default="metaseq_cli/train.py", help="script to launch"
    )
    parser.add_argument(
        "--python", default="python", help="path to nonstandard python binary"
    )

    # Slurm params
    parser.add_argument(
        "--salloc", action="store_true", help="run agaist current allocation"
    )
    parser.add_argument("--reservation", help="reservation to run on")
    parser.add_argument(
        "--exclusive", action="store_true", help="if set, get exclusive host"
    )
    parser.add_argument(
        "--time", default="4320", help="expected job duration in minutes"
    )
    parser.add_argument("--mem", "--mem", help="memory to request")
    parser.add_argument(
        "--constraint",
        metavar="CONSTRAINT",
        help='gpu constraint, if any. e.g. "volta"',
    )
    parser.add_argument("--comment", help="comment string")
    parser.add_argument(
        "--snapshot-code",
        action="store_true",
        default=False,
        help="Flag for creating a snapshot of training code while creating slurm job,"
        ' path is "./slurm_snapshot_code/<TIME_ISO_FORMAT/>:", '
        "can find time from comment of slurm job.",
    )
    parser.add_argument(
        "--snapshot-root",
        type=str,
        default=".",
        help="root path for saving the snapshot code.",
    )
    parser.add_argument(
        "--snapshot-recurse-dirs-oss",
        default="metaseq,metaseq_cli",
        help="comma-separated directories from where to recursively copy *.py, *.so and *.yaml files",
    )
    parser.add_argument(
        "--no-tensorboard", action="store_true", help="disable tensorboard logging"
    )
    parser.add_argument("--no-wandb", action="store_true", help="disable WandB logging")

    # Env flags
    parser.add_argument("--azure", action="store_true", help="running on azure")
    parser.add_argument("--aws", action="store_true", help="running on aws")
    parser.add_argument("--fair", action="store_true", help="running on fair")

    # Azure specific flag
    parser.add_argument(
        "--full-azure-upload-path",
        default=None,
        help="Azure blob storage SAS URL",
    )

    parser.add_argument(
        "--azure-folder-auto-name",
        action="store_true",
        help="Automatically name azure folder",
    )

    # Following args have env specific defaults.
    parser.add_argument(
        "--partition",
        help="slurm partition to run on",
    )
    parser.add_argument(
        "--checkpoints-dir",
        help="save checkpoints and logs in <checkpoints-dir>/<prefix>.<save_dir_key>",
    )
    parser.add_argument("--cpus-per-task", type=str)
    parser.add_argument(
        "--cpu-bind", help="configured to improve all-to-all perf, especially on A100s"
    )
    parser.add_argument(
        "--local-checkpoints-dir",
        help="node-local directory for saving checkpoints",
    )
    parser.add_argument(
        "--tensorboard-logdir",
        default=None,  # None will default to save_dir/tb
        help="save tensorboard logs in <tensorboard-logdir>/<prefix>.<save_dir_key>",
    )

    if add_extra_options_func is not None:  # mutates parser
        add_extra_options_func(parser)
    args = parser.parse_args(input_args)

    # Env check
    assert (
        sum([args.azure, args.aws, args.fair]) == 1
    ), "Must pass an env, and only one env (--azure, --aws, --fair)!"

    # Set defaults based on env
    env = get_env_from_args(args)
    _modify_arg_defaults_based_on_env(env, args)
    return args


def _modify_arg_defaults_based_on_env(env, args):
    # TODO(susan): move all this default logic into separate config file
    default_partition = None
    if env == ComputeEnvs.FAIR:
        default_partition = "learnfair"

    default_prefix = ""
    if env == ComputeEnvs.AZURE:
        default_prefix = "/shared/home"
    elif env == ComputeEnvs.AWS:
        default_prefix = "/checkpoints"
    elif env == ComputeEnvs.FAIR:
        default_prefix = "/checkpoint"

    if env == ComputeEnvs.FAIR:
        default_checkpoint_dir = os.path.join(
            default_prefix, os.environ["USER"], str(datetime.date.today())
        )
    else:
        default_checkpoint_dir = os.path.join(
            default_prefix,
            os.environ["USER"],
            "checkpoints",
            str(datetime.date.today()),
        )

    default_cpu_per_task = None
    if env == ComputeEnvs.AZURE or env == ComputeEnvs.AWS:
        default_cpu_per_task = 12
    elif env == ComputeEnvs.FAIR:
        default_cpu_per_task = 10

    default_cpu_bind = "none"
    if env == ComputeEnvs.AZURE:
        default_cpu_bind = (
            "mask_cpu:ffffff000000,ffffff000000,ffffff,ffffff,"
            "ffffff000000000000000000,ffffff000000000000000000,"
            "ffffff000000000000,ffffff000000000000"
        )
    elif env == ComputeEnvs.AWS:
        default_cpu_bind = (
            "mask_cpu:000000ffffff000000ffffff,000000ffffff000000ffffff,"
            "000000ffffff000000ffffff,000000ffffff000000ffffff,"
            "ffffff000000ffffff000000,ffffff000000ffffff000000,"
            "ffffff000000ffffff000000,ffffff000000ffffff000000"
        )
    elif env == ComputeEnvs.FAIR:
        default_cpu_bind = "map_ldom:0,0,0,0,1,1,1,1"

    default_local_checkpoints_dir = None
    if env == ComputeEnvs.AZURE:
        azure_upload_path = os.environ.get("AZURE_BLOB_SAS_URL", "")
        if azure_upload_path != "":
            # write checkpoints to local scratch storage on each node
            default_local_checkpoints_dir = os.path.join(
                "/mnt/scratch",
                os.environ["USER"],
                "checkpoints",
                str(datetime.date.today()),
            )

            # then copy them to Azure blob storage
            o = urlparse(azure_upload_path)
            o = o._replace(
                path=os.path.join(
                    o.path, os.environ["USER"], str(datetime.date.today())
                )
            )
            azure_upload_path = o.geturl()

            # set upload path if not specified
            if args.full_azure_upload_path is None:
                args.full_azure_upload_path = azure_upload_path

            # if needed, create a container for this user on the Azure blob account
            cmd = [
                "azcopy",  # TODO(susanz): requires azcopy to be installed.
                "make",
                o._replace(path=os.path.dirname(o.path)).geturl(),
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # assign default slurm partition
    if args.partition is None:
        args.partition = default_partition

    # assign default checkpoint directory
    if args.checkpoints_dir is None:
        args.checkpoints_dir = default_checkpoint_dir

    # assign default # cpus per task
    if args.cpus_per_task is None:
        args.cpus_per_task = str(default_cpu_per_task)

    # assign default cpu bind
    if args.cpu_bind is None:
        args.cpu_bind = default_cpu_bind

    # assign default local checkpoint dir
    if args.local_checkpoints_dir is None:
        args.local_checkpoints_dir = default_local_checkpoints_dir


def main(
    get_grid: Callable[[argparse.Namespace], List[hyperparam]],
    postprocess_hyperparams: Callable[
        [argparse.Namespace, MutableMapping[str, hyperparam]], None
    ],
    add_extra_options_func: Optional[Callable[[argparse.ArgumentParser], None]] = None,
    scheduler_args: Optional[List[str]] = None,
) -> None:
    """Do a grid search.

    Parameters:
        get_grid: A unary callable which returns the grid to search over.
            The callable is passed the parsed sweep arguments including the extra
            arguments defined by `add_extra_options_func`. See also `get_args`.
            The returned list represents the dimensions of the grid. That is, a list of
            length n represents a grid of dimension n. Let v_i denote the number of
            possible values for dimension i. Then the total number of configurations
            is given by v_1 * ... * v_n.
        postprocess_hyperparams: A 2-ary callable to post-process hyperparameter
            configurations before running the job. The first argument is the parsed
            sweep arguments including the extra arguments defined by
            `add_extra_options_func`. The second argument is a realized hyperparameter
            configuration as a mutable mapping of hyperparameter name to `hyperparam`
            instance with a `current_value` set.
        add_extra_options_func: A unary callable which adds extra arguments to the
            sweep CLI. It is passed the parser used to define the sweep script's CLI.
        scheduler_args: A list of unprocessed arguments to parse. If None, then
            `sys.argv[1:]`.
    """
    args = _get_args(add_extra_options_func, scheduler_args)
    from .slurm import main as backend_main

    get_grid = get_grid[args.grid] if args.grid is not None else get_grid
    backend_main(get_grid, postprocess_hyperparams, args)
