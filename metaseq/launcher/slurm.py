# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import fnmatch
import hashlib
import itertools
import os
import random
import shlex
import shutil
import subprocess
import textwrap
from collections import OrderedDict
from pathlib import Path

import metaseq
from metaseq.launcher.sweep import get_env_from_args
from metaseq.utils import get_random_port


def main(get_grid, postprocess_hyperparams, args):
    def dry_run(msg):
        if args.dry_run:
            print(f"| dry-run:  {msg}")
        return args.dry_run

    if args.local:
        args.num_nodes = 1

    # compute all possible hyperparameter configurations
    grid = get_grid(args)
    grid_product = list(itertools.product(*[hp.values for hp in grid]))

    # randomly shuffle configurations
    random.seed(args.seed)
    random.shuffle(grid_product)

    launch_train(args, grid, grid_product, dry_run, postprocess_hyperparams)


def copy_all_python_files(
    source,
    snapshot_main_dir,
    code_snapshot_hash,
    recurse_dirs="metaseq,metaseq_cli,scripts",
):
    """
    Copies following files from source to destination:
        a) all *.py files at direct source location.
        b) all metaseq/*.py recursively (default); recurse through comma-separated recurse_dirs
    """

    def include_patterns(*patterns):
        """Factory function that can be used with copytree() ignore parameter.

        Arguments define a sequence of glob-style patterns
        that are used to specify what files to NOT ignore.
        Creates and returns a function that determines this for each directory
        in the file hierarchy rooted at the source directory when used with
        shutil.copytree().
        from: https://stackoverflow.com/questions/52071642/python-copying-the-files-with-include-pattern
        """

        def _ignore_patterns(path, names):
            keep = set(
                name for pattern in patterns for name in fnmatch.filter(names, pattern)
            )
            ignore = set(
                name
                for name in names
                if name not in keep and not os.path.isdir(os.path.join(path, name))
            )
            return ignore

        return _ignore_patterns

    def pys_but_no_dirs(path, names):
        pys = set(fnmatch.filter(names, "*.py"))
        return [name for name in names if name not in pys]

    destination = os.path.join(snapshot_main_dir, code_snapshot_hash)
    # copy root files:
    shutil.copytree(source, destination, ignore=pys_but_no_dirs)
    # copy folders
    for d in recurse_dirs.split(","):
        shutil.copytree(
            os.path.join(source, d),
            os.path.join(destination, d),
            ignore=include_patterns("*.py", "*.so", "*.yaml"),
        )
    return destination


def run_setup(args, config, dry_run):
    # compute save_dir
    save_dir_key = ".".join(
        filter(
            lambda save_dir_key: save_dir_key is not None and len(save_dir_key) > 0,
            [hp.get_save_dir_key() for hp in config.values()],
        )
    )
    save_dir_key = save_dir_key.replace(",", "_")
    num_total_gpus = args.num_nodes * args.num_gpus
    save_dir = os.path.join(
        args.checkpoints_dir, f"{args.prefix}.{save_dir_key}.ngpu{num_total_gpus}"
    )

    # create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        if not dry_run(f"create directory: {save_dir}"):
            os.makedirs(save_dir)

    return save_dir_key, save_dir


def is_job_valid(args, save_dir, dry_run):
    if has_finished(save_dir):
        return False
    elif has_failed(save_dir):
        if args.resume_failed:
            dry_run(f"resume failed run: {save_dir}")
        else:
            print(f"skip failed run (override with --resume-failed): {save_dir}")
            return False
    elif has_started(save_dir):
        print(f"skip in progress run: {save_dir}")
        return False
    return True


DEFAULT_NCCL_DEBUG = os.getenv("NCCL_DEBUG", "INFO")
DEFAULT_NCCL_DEBUG_LOCAL = os.getenv("NCCL_DEBUG", "")


def set_env(args, env, dry_run):
    if "OMP_NUM_THREADS" not in env:
        env["OMP_NUM_THREADS"] = "2"
    if args.local:
        if not dry_run("start training locally"):
            if "CUDA_VISIBLE_DEVICES" not in env:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args.num_gpus)))
            env["NCCL_DEBUG"] = DEFAULT_NCCL_DEBUG_LOCAL
    else:
        if args.num_nodes > 1:
            env["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
            env["NCCL_DEBUG"] = DEFAULT_NCCL_DEBUG


def gen_train_command(args, env, config, oss_destination, save_dir, save_dir_key):
    # generate train command
    train_cmd = [args.python, os.path.join(oss_destination, args.script)]
    train_cmd.extend(["--distributed-world-size", str(args.num_nodes * args.num_gpus)])
    if args.num_nodes > 1 or (args.num_gpus > 1 and not args.local):
        train_cmd.extend(
            [
                "--distributed-port",
                str(get_random_port()),
            ]
        )
    if args.data is not None:
        train_cmd.extend([args.data])
    if args.local_checkpoints_dir is None:
        train_cmd.extend(["--save-dir", save_dir])
    else:
        num_total_gpus = args.num_nodes * args.num_gpus
        local_save_dir = os.path.join(
            args.local_checkpoints_dir,
            f"{args.prefix}.{save_dir_key}.ngpu{num_total_gpus}",
        )
        train_cmd.extend(["--save-dir", local_save_dir])
    if getattr(args, "full_azure_upload_path", None) is not None:
        if args.azure_folder_auto_name:
            from urllib.parse import urlparse

            o = urlparse(args.full_azure_upload_path)
            o = o._replace(
                path=os.path.join(
                    o.path, f"{args.prefix}.{save_dir_key}.ngpu{num_total_gpus}"
                )
                + "/"
            )
            train_cmd.extend(["--save-async", "--cloud-upload-path", o.geturl()])
        else:
            train_cmd.extend(
                ["--save-async", "--cloud-upload-path", args.full_azure_upload_path]
            )

    if not args.no_wandb:
        try:
            import wandb
        except ImportError:
            wandb = None
        if wandb or ("WANDB_API_KEY" in env and "WANDB_BASE_URL" in env):
            if "--wandb-project" not in config:
                project = f"{args.prefix}.{save_dir_key}"
                train_cmd.extend(["--wandb-project", project])
            if "WANDB_RUN_GROUP" not in env:
                env["WANDB_RUN_GROUP"] = args.prefix
            if "WANDB_RUN_ID" not in env:
                env["WANDB_RUN_ID"] = hashlib.md5(save_dir.encode("utf-8")).hexdigest()
            if "WANDB_RESUME" not in env:
                env["WANDB_RESUME"] = "allow"

    if not args.no_tensorboard:
        if args.tensorboard_logdir is None:
            tensorboard_logdir = os.path.join(save_dir, "tb")
        else:
            tensorboard_logdir = os.path.join(
                args.tensorboard_logdir,
                f"{args.prefix}.{save_dir_key}.ngpu{str(args.num_nodes * args.num_gpus)}",
            )
        train_cmd.extend(["--tensorboard-logdir", tensorboard_logdir])
    cluster_env = get_env_from_args(args)
    train_cmd.extend(["--cluster-env", cluster_env.value])

    for hp in config.values():
        train_cmd.extend(map(str, hp.get_cli_args()))
    return train_cmd


def gen_srun_command_and_str(args, save_dir_key, train_log, train_stderr, train_cmd):
    base_srun_cmd = [
        "srun",
        "--job-name",
        f"{args.prefix}.{save_dir_key}",
        "--output",
        train_log,
        "--error",
        train_stderr,
        "--open-mode",
        "append",
        "--unbuffered",
    ]
    if args.cpu_bind:
        base_srun_cmd += [f"--cpu-bind={args.cpu_bind}"]
    if args.salloc:
        excluded_hosts = os.environ.get("EXCLUDED_HOSTS", None)
        included_hosts = os.environ.get("INCLUDED_HOSTS", None)
        base_srun_cmd += [
            "--nodes",
            str(args.num_nodes),
            "--ntasks-per-node",
            str(args.num_gpus),
            "--ntasks",
            str(args.num_gpus * args.num_nodes),
            "--cpus-per-task",
            args.cpus_per_task,
        ]
        base_srun_cmd += ["-x", excluded_hosts] if excluded_hosts is not None else []
        base_srun_cmd += ["-w", included_hosts] if included_hosts is not None else []

    srun_cmd = base_srun_cmd + train_cmd
    srun_cmd_str = " ".join(map(shlex.quote, srun_cmd))
    if getattr(args, "requeue_on_fail", False):
        # sometimes we want the job to just be requeued magically if it exit codes
        # i.e. in the case of very very large models with long runtimes.
        srun_cmd_str = f"( {srun_cmd_str} || scontrol requeue $SLURM_JOB_ID )"
    return srun_cmd, srun_cmd_str


def gen_sbatch_command_and_str(
    args, job_name, train_log, train_stderr, oss_destination, srun_cmd_str
):
    excluded_hosts = os.environ.get("EXCLUDED_HOSTS", None)
    included_hosts = os.environ.get("INCLUDED_HOSTS", None)
    sbatch_cmd = [
        "sbatch",
        "--job-name",
        job_name,
        "--gpus-per-node",
        str(args.num_gpus),
        "--nodes",
        str(args.num_nodes),
        "--ntasks-per-node",
        str(args.num_gpus),
        "--cpus-per-task",
        args.cpus_per_task,
        "--output",
        train_log,
        "--error",
        train_stderr,
        "--open-mode",
        "append",
        # '--no-requeue',
        "--signal",
        "B:USR1@180",
    ]

    if args.constraint:
        sbatch_cmd += ["--constraint", args.constraint]

    if args.partition:
        sbatch_cmd += ["--partition", args.partition]

    if args.reservation:
        sbatch_cmd += ["--reservation", args.reservation]

    if args.exclusive:
        sbatch_cmd += ["--exclusive"]

    comment = ""
    if args.comment:
        comment = args.comment

    if args.snapshot_code:
        comment += (
            f", OSS Code Location: {oss_destination}"
            if comment
            else f"OSS Code Location: {oss_destination}"
        )
        sbatch_cmd += ["--comment", comment]

    if args.time is not None:
        sbatch_cmd.extend(["--time", args.time])

    if args.mem is not None:
        sbatch_cmd += ["--mem", args.mem]
    else:
        sbatch_cmd += ["--mem", "0"]

    sbatch_cmd += ["-x", excluded_hosts] if excluded_hosts is not None else []
    sbatch_cmd += ["-w", included_hosts] if included_hosts is not None else []

    wrapped_cmd = requeue_support()
    if args.azure:
        wrapped_cmd += "\n" + azure_support()

    wrapped_cmd += "\n" + srun_cmd_str + " \n wait $! \n sleep 610 & \n wait $!"

    sbatch_cmd += ["--wrap", wrapped_cmd]
    sbatch_cmd_str = " ".join(map(shlex.quote, sbatch_cmd))
    return sbatch_cmd, sbatch_cmd_str


def local_run(args, env, train_cmd, dry_run):
    assert args.num_nodes == 1, "distributed training cannot be combined with --local"
    if not dry_run("start training locally"):
        if "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args.num_gpus)))
        env["NCCL_DEBUG"] = DEFAULT_NCCL_DEBUG_LOCAL
        train_proc = subprocess.Popen(train_cmd, env=env)
        train_proc.wait()


def run_batch(env, sbatch_cmd_str, sbatch_cmd):
    print(f"running command: {sbatch_cmd_str}\n")
    with subprocess.Popen(sbatch_cmd, stdout=subprocess.PIPE, env=env) as train_proc:
        stdout = train_proc.stdout.read().decode("utf-8")
        try:
            job_id = int(stdout.rstrip().split()[-1])
            print(f"Launched job {job_id}")
        except IndexError:
            job_id = None
    return job_id, stdout


def write_git_commit(train_log):
    with open(train_log, "a") as train_log_h:
        # log most recent git commit
        git_commit = subprocess.check_output(
            "git log | head -n 1", shell=True, encoding="utf-8"
        )
        print(git_commit.rstrip(), file=train_log_h)


def dry_run_batch(env, train_log, train_stderr, sbatch_cmd_str, sbatch_cmd, dry_run):
    dry_run("start remote training")
    dry_run(f"- log stdout to: {train_log}")
    dry_run(f"- log stderr to: {train_stderr}")
    dry_run(f"- run command: {sbatch_cmd_str}")
    sbatch_cmd += ["--test-only"]
    with subprocess.Popen(sbatch_cmd, stdout=subprocess.PIPE, env=env) as train_proc:
        stdout = train_proc.stdout.read().decode("utf-8")
        print(stdout)


def launch_train(args, grid, grid_product, dry_run, postprocess_hyperparams):
    oss_destination = ""
    if args.snapshot_code:
        # Currently hash is just the current time in ISO format.
        # Remove colons since they cannot be escaped in POSIX PATH env vars.
        code_snapshot_hash = datetime.datetime.now().isoformat().replace(":", "_")
        # Copy metaseq OSS code
        metaseq_oss_path = str(Path(metaseq.__file__).parents[1])
        oss_destination = copy_all_python_files(
            metaseq_oss_path,
            os.path.join(args.snapshot_root, "slurm_snapshot_code_oss"),
            code_snapshot_hash,
            args.snapshot_recurse_dirs_oss,
        )
        os.environ["PYTHONPATH"] = (
            oss_destination + ":" + os.environ.get("PYTHONPATH", "")
        )

    # set environment
    base_env = os.environ.copy()
    set_env(args, base_env, dry_run)

    # start training
    for i, hp_values in enumerate(grid_product):
        if i == args.num_trials:
            break
        config = OrderedDict()
        for hp, value in zip(grid, hp_values):
            config[hp.name] = hp
            config[hp.name].current_value = value

        # postprocess hyperparams
        postprocess_hyperparams(args, config)

        save_dir_key, save_dir = run_setup(args, config, dry_run)

        # check if job failed, exists, finished
        if not is_job_valid(args, save_dir, dry_run):
            continue

        # clone base env and update for this job, e.g., we set WANDB_RUN_ID
        # based on the save_dir, which is based on the current hyperparam values
        env = base_env.copy()

        # generate train command
        train_cmd = gen_train_command(
            args, env, config, oss_destination, save_dir, save_dir_key
        )

        train_log = os.path.join(save_dir, "train.log")
        train_stderr = os.path.join(save_dir, "train.stderr.%j")  # %j = slurm job id
        srun_cmd, srun_cmd_str = gen_srun_command_and_str(
            args, save_dir_key, train_log, train_stderr, train_cmd
        )

        job_id = None
        if args.dry_run:
            train_cmd_str = " ".join(train_cmd)
            dry_run(f"train command: {train_cmd_str}")

        if args.local:
            local_run(args, env, train_cmd, dry_run)
        else:
            srun_cmd_str = srun_cmd_str + " &"
            # build command
            if not args.salloc:
                job_name = f"{args.prefix}.{save_dir_key}"
                sbatch_cmd, sbatch_cmd_str = gen_sbatch_command_and_str(
                    args,
                    job_name,
                    train_log,
                    train_stderr,
                    oss_destination,
                    srun_cmd_str,
                )
            else:
                sbatch_cmd = srun_cmd
                sbatch_cmd_str = srun_cmd_str
            if args.dry_run:
                dry_run_batch(
                    env, train_log, train_stderr, sbatch_cmd_str, sbatch_cmd, dry_run
                )
            else:
                write_git_commit(train_log)
                with open(train_log, "a") as train_log_h:
                    job_id, stdout = run_batch(env, sbatch_cmd_str, sbatch_cmd)
                    print(stdout, file=train_log_h)
        if job_id is not None:
            print("Launched {}".format(job_id))


def has_finished(save_dir):
    train_log = os.path.join(save_dir, "train.log")
    if not os.path.exists(train_log):
        return False
    with open(train_log, "r") as h:
        lines = h.readlines()
        if len(lines) == 0:
            return False
        if "done training" in lines[-1]:
            return True
    return False


def has_failed(save_dir):
    if not os.path.exists(save_dir):
        return False

    # find max job id
    job_ids = []
    for fn in os.listdir(save_dir):
        if fn.startswith("train.stderr."):
            job_ids.append(int(fn.split(".")[-1]))
    if len(job_ids) == 0:
        return False
    max_job_id = max(job_ids)

    def _has_failed(stderr_fn):
        with open(stderr_fn, "r") as h:
            for line in h:
                if len(line.strip()) > 0:
                    # assume that any output in stderr indicates an error
                    return True
        return False

    return _has_failed(os.path.join(save_dir, f"train.stderr.{max_job_id}"))


def has_started(save_dir):
    train_log = os.path.join(save_dir, "train.log")
    if not os.path.exists(train_log):
        return False
    return True


def requeue_support():
    return textwrap.dedent(
        """
        trap_handler () {
           echo "Caught signal: " $1
           # SIGTERM must be bypassed
           if [ "$1" = "TERM" ]; then
               echo "bypass sigterm"
           else
             # Submit a new job to the queue
             echo "Requeuing " $SLURM_JOB_ID
             scontrol requeue $SLURM_JOB_ID
           fi
        }


        # Install signal handler
        trap 'trap_handler USR1' USR1
        trap 'trap_handler TERM' TERM
    """
    )


def azure_support():
    return textwrap.dedent(
        """
        export NCCL_TOPO_FILE=/opt/microsoft/ndv4-topo.xml
        export NCCL_IB_PCI_RELAXED_ORDERING=1
        export UCX_IB_PCI_RELAXED_ORDERING=on
        export NCCL_SOCKET_IFNAME=eth0
        export UCX_NET_DEVICES=eth0
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        export OMPI_MCA_COLL_HCOLL_ENABLE=0
        if [ -e "/etc/profile.d/modules.sh" ]; then
            . /etc/profile.d/modules.sh
            module load mpi/hpcx
        fi
        """
    )
