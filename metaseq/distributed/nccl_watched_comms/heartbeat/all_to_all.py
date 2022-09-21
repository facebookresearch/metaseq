
import logging
from metaseq.distributed.nccl_watched_comms.utils import loginfo, init_comm, NCCLtimeout, NotGPUfailRelatedTimeout, SENDER, RECEIVER

import torch
import datetime
import queue as queue_module
import time
import signal
import os
import torch.distributed as dist
import torch.multiprocessing as mp


def heartbeat(comm_fn, queue, ppid, device, period, timeout, signal_to_send_at_timeout):
    """
        comm_fn = lambda tensor: dist.isend(tensor, rcv=receiver_rank)
        comm_fn = lambda tensor: dist.irecv(tensor, rcv=receiver_rank)
        comm_fn must return a work object (be async)

        period and timeout are float number of seconds

        device is "cuda:%N%"
    """

    last_comm_time = datetime.datetime.utcnow()

    tensor = torch.tensor(1, device=device, dtype=torch.bool)

    work = comm_fn(tensor)
    n_missed_beats = 0
    period_in_seconds = period.total_seconds()
    
    fired_flag = False

    while True:
        if work.is_completed():
            last_comm_time = datetime.datetime.utcnow()
            loginfo(f"Successful heartbeat! The time is {last_comm_time}")
            n_missed_beats = 0
            work = comm_fn(tensor)
        elif not fired_flag:
            n_missed_beats += 1
            loginfo(f"Heartbeat miss! This is {n_missed_beats}-th time in a row!")
            if n_missed_beats*period > timeout:
                loginfo(f"Timeout exceeded! Sending signal {signal_to_send_at_timeout} to {ppid}")
                fired_flag = True
                os.kill(ppid, signal_to_send_at_timeout)
        try:
            put_request = queue["to_heartbeat"].get_nowait()
            if put_request:
                timediff = datetime.datetime.utcnow() - last_comm_time
                loginfo(f"Report requested, sending {timediff}")
                queue["from_heartbeat"].put_nowait(timediff)
        except queue_module.Empty:
            loginfo(f"No report requested, going to sleep for {period_in_seconds} seconds")
            pass
        time.sleep(period_in_seconds)

def get_lags(dict_of_timediff_queues):
    lags = {}
    for q in dict_of_timediff_queues.values():
        q["to_heartbeat"].put_nowait(True)
    for comm_name, q in dict_of_timediff_queues.items():
        loginfo(f"Requesting a lag from {comm_name}..")
        lags[comm_name] = q["from_heartbeat"].get()
        loginfo(f"Reported lag of {comm_name} is {lags[comm_name]}")
    return lags

def get_signal_handler(dict_of_timediff_queues, former_signal_handler, timeout):

    def handler(signum, stack):
        current_handler = signal.signal(signum, signal.SIG_IGN)
        loginfo(f"Signal {signum} is being handled..")
        lags = get_lags(dict_of_timediff_queues)
        timeouted_comms = [comm_name for comm_name in lags.keys() if lags[comm_name] > timeout]
        signal.signal(signum, current_handler)
        if len(timeouted_comms) > 0:
            loginfo(f"Timeouted comms are {timeouted_comms}. Raising NCCLtimeout")
            raise NCCLtimeout(timeouted_comms)
        else:
            loginfo(f"No timeouted comms. Falling back to standard handler and then raising the NotGPUfailRelatedTimeout")
            signal.signal(signum, former_signal_handler)
            os.kill(os.getpid(), signum)
            raise NotGPUfailRelatedTimeout()

    return handler



def heartbeat_fn(role, rank, partner, world_size, backend, queue, watch_device, communicator_check_period, communicator_timeout, signal_to_send_at_timeout, verbose=False):
    if verbose:
        logging.basicConfig(
                    format=f'{role} heartbeat {min(rank, partner)}->{max(rank, partner)}: %(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + world_size*(min(rank, partner)+1) + max(rank, partner))
    loginfo(f"Using port {os.environ['MASTER_PORT']} for communication")
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    if role==SENDER:
        init_comm(rank=0, world_size=2, backend=backend) 
        comm_fn = lambda tensor: dist.isend(tensor, dst=1) 
    else: 
        init_comm(rank=1, world_size=2, backend=backend)
        comm_fn = lambda tensor: dist.irecv(tensor, src=0)
    heartbeat(comm_fn=comm_fn, queue=queue, ppid=os.getppid(), device=watch_device, period=communicator_check_period, timeout=communicator_timeout,
                signal_to_send_at_timeout=signal_to_send_at_timeout)


def get_heartbeat_queues_and_procs(
                            partners,
                            rank, 
                            world_size, 
                            backend, 
                            watch_device, 
                            communicator_check_period, 
                            communicator_timeout, 
                            signal_to_send_at_timeout, 
                            verbose):
    heartbeat_procs = {}
    heartbeat_queues = {}

    ctx = mp.get_context('spawn')
    for partner in partners:
        role = SENDER if rank < partner else RECEIVER
        queue_to_heartbeat = ctx.Queue()
        queue_from_heartbeat = ctx.Queue()
        queue = {"to_heartbeat": queue_to_heartbeat, "from_heartbeat": queue_from_heartbeat}
        heartbeat_queues[partner] = queue
        heartbeat_procs[partner] = ctx.Process(target=heartbeat_fn, 
                    args=(role, rank, partner, world_size, backend, queue, watch_device, communicator_check_period, communicator_timeout, signal_to_send_at_timeout, verbose), 
                    daemon=True)
    
    return heartbeat_queues, heartbeat_procs

def clean_heartbeat_queues_and_procs(heartbeat_queues, heartbeat_procs, alive_comp_units):
    for partner in list(heartbeat_queues):
        if partner not in alive_comp_units:
            heartbeat_queues[partner]["to_heartbeat"].close()
            heartbeat_queues[partner]["from_heartbeat"].close()
            heartbeat_procs[partner].terminate()
            del heartbeat_queues[partner]
            del heartbeat_procs[partner]