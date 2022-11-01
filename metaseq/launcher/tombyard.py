# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import datetime
import signal
import os
import time
from pathlib import Path


def tombstones_procedure(
    job_id,
    dirstones,
    period_before_tombstone_detected=datetime.timedelta(seconds=60),
    period_after_tombstone_detected=datetime.timedelta(seconds=3),
):

    job_state_codes = [
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

    tombstone_detected = False
    period = period_before_tombstone_detected

    while True:
        sacct_result = os.popen(f"squeue -j {job_id} -O State -h ").read()
        status = sacct_result.strip()
        if tombstone_detected:
            print(f".. scanceling the job and its current squeue.state is {status}")
        if status not in job_state_codes:
            print(f"Done scanceling the job. Its squeue.state now is: {status}")
            return

        if not tombstone_detected:
            for tombstone_name in dirstones["scancel"]:
                if os.path.exists(tombstone_name):
                    print(
                        f"""tombstones_procedure has detected file {tombstone_name}. 
                         scancel {job_id} will be called every {period_after_tombstone_detected}
                          until the job is dead"""
                    )
                    tombstone_detected = True
                    period = period_after_tombstone_detected

            for tombstone_name in dirstones["requeuehold"]:
                if os.path.exists(tombstone_name):
                    print(
                        f"""tombstones_procedure has detected file {tombstone_name}. 
                         scontrol requeuehold {job_id} will be called once.
                         remove the file {tombstone_name} within the next {period_before_tombstone_detected}
                         for it not to trigger the same command again"""
                    )
                    report = os.popen(
                        f"""
                        scontrol requeuehold {job_id} 
                        """
                    ).read()

            for tombstone_name in dirstones["requeuehold"]:
                if os.path.exists(tombstone_name):
                    print(
                        f"""tombstones_procedure has detected file {tombstone_name}. 
                         scontrol release {job_id} will be called once.
                         remove the file {tombstone_name} within the next {period_before_tombstone_detected}
                         for it not to trigger the same command again"""
                    )
                    report = os.popen(
                        f"""
                        scontrol release {job_id} 
                        """
                    ).read()

        if tombstone_detected:
            report = os.popen(
                f"""
                    scancel {job_id}
                    """
            ).read()

        time.sleep(period.total_seconds())


def tombstones(job_id, period=datetime.timedelta(seconds=60), dirstones=None):

    if dirstones is None:
        dirstones = {"scancel": [], "requeuehold": [], "release": []}
        azure_base = "/shared/home"
        for userdir in os.listdir(azure_base):
            dirstones["scancel"].append(
                os.path.join(azure_base, userdir, f"scancel_{job_id}")
            )
            dirstones["requeuehold"].append(
                os.path.join(azure_base, userdir, f"requeuehold_{job_id}")
            )
            dirstones["release"].append(
                os.path.join(azure_base, userdir, f"release_{job_id}")
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
