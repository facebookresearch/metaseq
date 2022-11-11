# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import multiprocessing as mp
import datetime
import signal
import os
import time
from pathlib import Path
import logging

JOB_STATE_CODES = [
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "CONFIGURING",
    "COMPLETING",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PENDING",
    "PREEMPTED",
    "RUNNING",
    "RESV_DEL_HOLD",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "REQUEUED",
    "RESIZING",
    "REVOKED",
    "SIGNALING",
    "SPECIAL_EXIT",
    "STAGE_OUT",
    "STOPPED",
    "SUSPENDED",
    "TIMEOUT",
]


def tombstones_procedure(
    job_id,
    dirstones,
    period_before_tombstone_detected=datetime.timedelta(seconds=60),
    period_after_tombstone_detected=datetime.timedelta(seconds=3),
    logger=logging.getLogger(),
):

    tombstone_detected = False
    period = period_before_tombstone_detected

    while True:
        sacct_result = os.popen(f"squeue -j {job_id} -O State -h ").read()
        status = sacct_result.strip()
        if tombstone_detected:
            logger.info(f".. scanceling the job and its current squeue.state is {status}")
        if status not in JOB_STATE_CODES:
            logger.info(f"Done scanceling the job. Its squeue.state now is: {status}")
            return
        if not tombstone_detected:
            for tombstone_name in dirstones["scancel"]:
                if os.path.exists(tombstone_name):
                    logger.info(
                        f"tombstones_procedure has detected file {tombstone_name}. "
                        f"scancel {job_id} will be called every {period_after_tombstone_detected} "
                        f"until the job is dead "
                    )
                    tombstone_detected = True
                    period = period_after_tombstone_detected
            for tombstone_name in dirstones["requeuehold"]:
                if os.path.exists(tombstone_name):
                    logger.info(
                        f"tombstones_procedure has detected file {tombstone_name}. "
                        f"scontrol requeuehold {job_id} will be called once. "
                        f"remove the file {tombstone_name} within the next {period_before_tombstone_detected} "
                        f"for it not to trigger the same command again "
                    )
                    _ = os.popen(f"scontrol requeuehold {job_id}").read()
            for tombstone_name in dirstones["requeuehold"]:
                if os.path.exists(tombstone_name):
                    logger.info(
                        f"tombstones_procedure has detected file {tombstone_name}. "
                        f"scontrol release {job_id} will be called once. "
                        f"remove the file {tombstone_name} within the next {period_before_tombstone_detected} "
                        f"for it not to trigger the same command again "
                    )
                    _ = os.popen(f"scontrol release {job_id} ").read()
        if tombstone_detected:
            _ = os.popen(f"scancel {job_id} ").read()
        time.sleep(period.total_seconds())


def tombstones(job_id, base_dir, period=datetime.timedelta(seconds=60), dirstones=None):
    if dirstones is None:
        dirstones = {"scancel": [], "requeuehold": [], "release": []}
        for userdir in os.listdir(base_dir):
            dirstones["scancel"].append(
                os.path.join(base_dir, userdir, f"scancel_{job_id}")
            )
            dirstones["requeuehold"].append(
                os.path.join(base_dir, userdir, f"requeuehold_{job_id}")
            )
            dirstones["release"].append(
                os.path.join(base_dir, userdir, f"release_{job_id}")
            )

    for directive in ["scancel", "requeuehold", "release"]:
        if directive not in dirstones.keys():
            dirstones[directive] = []

    # start a process that monitors
    ctx = mp.get_context("spawn")
    heartbeat_proc = ctx.Process(
        target=tombstones_procedure, args=(job_id, dirstones, period), daemon=False
    )
    heartbeat_proc.start()



if __name__ == "__main__":
    logger = logging.getLogger(f"touchstoning process: PID {os.getpid()}")
    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser("Tombstoning script")
    parser.add_argument(
        "-bd",
        "--base-dir",
        type=str,
        required=True,
        help="number of random hyperparam configurations to try (-1 for grid search)",
    )
    parser.add_argument(
        "-j",
        "--job-id",
        type=int,
        required=True,
        help="number of random hyperparam configurations to try (-1 for grid search)",
    )
    args = parser.parse_args()
    tombstones(str(args.job_id), args.base_dir, logger=logger)
