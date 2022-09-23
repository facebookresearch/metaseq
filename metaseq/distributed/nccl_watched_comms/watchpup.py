import datetime
import signal

import torch.distributed as dist
import os
import contextlib
import logging

from metaseq.distributed.nccl_watched_comms.utils import ALL_TO_ALL, ALL_TO_ONE, loginfo, init_comm
from metaseq.distributed.nccl_watched_comms.checking_ports import check_required_ports


@contextlib.contextmanager
def init_watched_comm(
        rank, 
        world_size, 
        watch_device,
        backend="nccl", 
        nccl_timeout=datetime.timedelta(seconds=90), 
        communicator_timeout=datetime.timedelta(seconds=30), 
        communicator_check_period=datetime.timedelta(seconds=1), 
        mode=ALL_TO_ALL,
        verbose=False
        ):
    
    if verbose:
        logging.basicConfig(
                    format=f'slurm_procid {os.environ["SLURM_PROCID"]}:: %(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

    loginfo(f"Checking that the required ports are not busy..")

    check_required_ports(world_size, mode=mode)

    loginfo(f"Setting up NCCL_ASYNC_ERROR_HANDLING..")

    if "NCCL_ASYNC_ERROR_HANDLING" in os.environ.keys():
        previous_NCCL_ASYNC_ERROR_HANDLING_flag = os.environ["NCCL_ASYNC_ERROR_HANDLING"]
    else:
        previous_NCCL_ASYNC_ERROR_HANDLING_flag = "0"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "2"
    

    partners = set(range(world_size)).difference({rank})

    signal_to_send_at_timeout = signal.SIGALRM
    loginfo(f"Describing partner communication processes..")

    if mode == ALL_TO_ALL:
        from metaseq.distributed.nccl_watched_comms.heartbeat.all_to_all import get_heartbeat_queues_and_procs, get_signal_handler, clean_heartbeat_queues_and_procs
    elif mode == ALL_TO_ONE:
        from metaseq.distributed.nccl_watched_comms.heartbeat.all_to_one import get_heartbeat_queues_and_procs, get_signal_handler, clean_heartbeat_queues_and_procs
    else:
        raise NotImplementedError(f"only {ALL_TO_ALL} or {ALL_TO_ONE} modes are permitted")

    heartbeat_queues, heartbeat_procs = get_heartbeat_queues_and_procs(
                            partners=partners,
                            rank=rank, 
                            world_size=world_size, 
                            backend=backend, 
                            watch_device=watch_device, 
                            communicator_check_period=communicator_check_period, 
                            communicator_timeout=communicator_timeout, 
                            signal_to_send_at_timeout=signal_to_send_at_timeout, 
                            # verbose=verbose,
                            )

    loginfo(f"Modifying signal handeling routine..")

    former_sigalrm_handler = signal.getsignal(signal_to_send_at_timeout)
    sigalrm_handler = get_signal_handler(heartbeat_queues, former_sigalrm_handler, communicator_timeout)
    signal.signal(signal_to_send_at_timeout, sigalrm_handler)

    loginfo(f"Initializing main communication channel..")

    import math
    signal.alarm(math.ceil(communicator_timeout.total_seconds()))
    init_comm(rank, world_size, backend=backend, timeout=nccl_timeout)
    signal.alarm(0)

    loginfo(f"Launching partner communication processes..")

    for heartbeat_proc in heartbeat_procs.values():
        heartbeat_proc.start()

    # signal.alarm(math.ceil(internal_timeout.total_seconds()))

    def reinit_fn(alive_comp_units):
        # gather the dead
        loginfo(f"Reinitialization of the main communication channel. Our survived computation units are {alive_comp_units}. Cleaning the dead partner comms..")
        
        clean_heartbeat_queues_and_procs(heartbeat_queues, heartbeat_procs, alive_comp_units)
    
        loginfo(f"Obtaining the process group..")
        pg = dist.distributed_c10d._get_default_group()
        loginfo(f"Destroying the process group..")
        dist.distributed_c10d.destroy_process_group(pg)

        new_world_size = len(alive_comp_units)
        new_rank = alive_comp_units.index(rank)
        os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + len(alive_comp_units))
        loginfo(f"New world size is {new_world_size}. Reinitializing communication..")
        init_comm(new_rank, new_world_size, backend=backend, timeout=nccl_timeout)
        loginfo(f"Initialization completed.")




    yield reinit_fn

    # signal.alarm(0)
    for heartbeat_proc in heartbeat_procs.values():
        heartbeat_proc.terminate()

    for heartbeat_proc in heartbeat_procs.values():
        heartbeat_proc.join()
    
    signal.signal(signal_to_send_at_timeout, former_sigalrm_handler)

    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = previous_NCCL_ASYNC_ERROR_HANDLING_flag


