import datetime
import signal
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys
import logging
import time
import contextlib
import queue as queue_module
from collections import deque

def loginfo(msg):
    logging.info(msg)
    sys.stdout.flush()


def init_comm(rank, world_size, backend="nccl", timeout=datetime.timedelta(seconds=40)):

    dist.init_process_group(backend=backend,
                                    world_size=world_size, rank=rank,
                                    timeout=timeout)
    dist.barrier() # Initialize communication! (otherwise the first async op will be blocking)


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


class NCCLtimeout(Exception):
    def __init__(self, timeouted_comms, *args: object) -> None:
        super().__init__(*args)
        self.timeouted_comms = timeouted_comms


class NotGPUfailRelatedTimeout(Exception):
    pass


def get_lags(dict_of_timediff_queues):
    lags = {}
    for q in dict_of_timediff_queues.values():
        q["to_heartbeat"].put_nowait(True)
    for comm_name, q in dict_of_timediff_queues.items():
        loginfo(f"Requesting a lag from {comm_name}..")
        lags[comm_name] = q["from_heartbeat"].get()
        loginfo(f"Reported lag of {comm_name} is {lags[comm_name]}")
    return lags

def timeout_check(dict_of_timediff_queues, former_signal_handler, timeout):

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

RECEIVER = "recv"
SENDER = "send"

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

class PortBusyException(Exception):
    pass

def check_a_port(addr, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((addr, port))
    # if result == 0:
    #     print("Port is open")
    # else:
    #     print("Port is not open")
    sock.close()
    return result

def check_required_ports(world_size, n_attempts=3):
    main_master_port = int(os.environ["MASTER_PORT"])
    master_addr = os.environ["MASTER_ADDR"] 
    required_ports = deque(range(main_master_port+1, main_master_port+world_size*(world_size+1)))
    for _ in range(n_attempts):
        l = len(required_ports)
        for _ in range(l):
            port = required_ports.popleft()
            if check_a_port(master_addr, port) == 0:
                required_ports.append(port)
    if len(required_ports) > 0:
        raise PortBusyException(required_ports)
    

@contextlib.contextmanager
def init_watched_comm(
        rank, 
        world_size, 
        watch_device,
        backend="nccl", 
        nccl_timeout=datetime.timedelta(seconds=90), 
        communicator_timeout=datetime.timedelta(seconds=30), 
        communicator_check_period=datetime.timedelta(seconds=1), 
        # internal_timeout=datetime.timedelta(seconds=30)
        ):

    loginfo(f"Checking that the required ports are not busy..")

    check_required_ports(world_size)

    loginfo(f"Setting up NCCL_ASYNC_ERROR_HANDLING..")

    if "NCCL_ASYNC_ERROR_HANDLING" in os.environ.keys():
        previous_NCCL_ASYNC_ERROR_HANDLING_flag = os.environ["NCCL_ASYNC_ERROR_HANDLING"]
    else:
        previous_NCCL_ASYNC_ERROR_HANDLING_flag = "0"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "2"

    heartbeat_procs = {}
    heartbeat_queues = {}
    ctx = mp.get_context('spawn')

    partners = set(range(world_size)).difference({rank})

    signal_to_send_at_timeout = signal.SIGALRM
    loginfo(f"Describing partner communication processes..")

    for partner in partners:
        role = SENDER if rank < partner else RECEIVER
        queue_to_heartbeat = ctx.Queue()
        queue_from_heartbeat = ctx.Queue()
        queue = {"to_heartbeat": queue_to_heartbeat, "from_heartbeat": queue_from_heartbeat}
        heartbeat_queues[partner] = queue
        heartbeat_procs[partner] = ctx.Process(target=heartbeat_fn, 
                    args=(role, rank, partner, world_size, backend, queue, watch_device, communicator_check_period, communicator_timeout, signal_to_send_at_timeout), 
                    daemon=True)

    loginfo(f"Modifying signal handeling routine..")

    former_sigalrm_handler = signal.getsignal(signal_to_send_at_timeout)
    sigalrm_handler = timeout_check(heartbeat_queues, former_sigalrm_handler, communicator_timeout)
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
        
        for partner in heartbeat_queues.keys():
            if partner not in alive_comp_units:
                heartbeat_queues[partner]["to_heartbeat"].close()
                heartbeat_queues[partner]["from_heartbeat"].close()
                heartbeat_procs[partner].terminate()
                del heartbeat_queues[partner]
                del heartbeat_procs[partner]
    
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
    
    signal.signal(signal_to_send_at_timeout, former_sigalrm_handler)

    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = previous_NCCL_ASYNC_ERROR_HANDLING_flag


