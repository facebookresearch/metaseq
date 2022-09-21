from turtle import shape
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import signal
import os
import queue as queue_module
import time
import logging
import datetime

from metaseq.distributed.nccl_watched_comms.utils import init_comm, loginfo, NCCLtimeout, NotGPUfailRelatedTimeout, SENDER, RECEIVER
from metaseq.distributed.nccl_watched_comms.heartbeat.datetime_converter import DatetimeTensorConverter

CURATOR = 'curator'

# This code relies significantly on the assumption that the devices in the network have synchronized clocks

def curator_starter(rank, partners, world_size, heartbeat_queues, communicator_check_period, communicator_timeout, signal_to_send_at_timeout, watch_device, signal_pid, backend, verbose):
    if verbose:
        logging.basicConfig(
                    format=f'{CURATOR} heartbeat on {rank}: %(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
    
    queues_to_heartbeats = {}
    heartbeat_procs = {}
    role = SENDER

    ctx = mp.get_context('spawn')
    for partner in partners:
        curator_to_hearbeat = ctx.Queue()
        curator_from_heartbeat = ctx.Queue()
        queue = {"to_heartbeat": curator_to_hearbeat, "from_heartbeat": curator_from_heartbeat}
        queues_to_heartbeats[partner] = queue
        heartbeat_procs[partner] = ctx.Process(target=heartbeat_starter, 
                    args=(role, rank, partner, world_size, backend, queue, watch_device, signal_pid, communicator_check_period, communicator_timeout, signal_to_send_at_timeout, verbose), 
                    daemon=True)
    
    for heartbeat_proc in heartbeat_procs.values():
        heartbeat_proc.start()

    heartbeat_procedure(rank=rank, all_partners=partners, queues_to_host=heartbeat_queues[CURATOR], queues_to_heartbeats=queues_to_heartbeats, heartbeat_procs=heartbeat_procs, device=watch_device, period=communicator_check_period)

def curator_procedure(rank, all_partners, queues_to_host, queues_to_heartbeats, heartbeat_procs, device, period):
    all_partners = sorted(all_partners)
    alive_comp_units = all_partners
    converter = DatetimeTensorConverter(device=device)
    period_in_seconds = period.total_seconds()

    last_comm_time_encodings = []
    for partner in all_partners:
        last_comm_time_encodings.append(queues_to_heartbeats[partner]["from_heartbeat"].get())
    last_comm_time_encodings.insert(rank, converter.utc_tensor())
    times_tensor = converter.join_tensors(last_comm_time_encodings)

    while True:
        for partner in alive_comp_units:
            try:
                last_comm_time_encodings[partner] = queues_to_heartbeats[partner]["from_heartbeat"].get_nowait()
            except queue_module.Empty:
                pass
        last_comm_time_encodings[rank] = converter.utc_tensor()
        times_tensor = converter.join_tensors(last_comm_time_encodings)
        for partner in alive_comp_units:
            try:
                queues_to_heartbeats[partner]["to_heartbeat"].get_nowait()
            except queue_module.Empty:
                pass
            queues_to_heartbeats[partner]["to_heartbeat"].put_nowait(times_tensor)

        try:
            message = queues_to_host["host_to_curator"].get_nowait()
            if message['type'] == 'put_request':
                now_datetime = datetime.datetime.utcnow()
                timediffs = [now_datetime - last_comm_time for last_comm_time in converter.matrix_to_datetimes(converter.decode_tensor(times_tensor))]
                loginfo(f"Report requested, sending {timediffs}")
                queues_to_host["host_from_curator"].put_nowait(timediffs)
            if message['type'] == 'clear_request':
                alive_comp_units = message['content']
                for partner in list(queues_to_heartbeats):
                    if partner not in alive_comp_units:
                        queues_to_heartbeats[partner]["to_heartbeat"].close()
                        queues_to_heartbeats[partner]["from_heartbeat"].close()
                        heartbeat_procs[partner].terminate()
                        del queues_to_heartbeats[partner]
                        del heartbeat_procs[partner]

        except queue_module.Empty:
            loginfo(f"No report requested, going to sleep for {period_in_seconds} seconds")
            pass
        time.sleep(period_in_seconds)




def heartbeat_starter(role, rank, partner, world_size, backend, queue, watch_device, signal_pid, communicator_check_period, communicator_timeout, signal_to_send_at_timeout, verbose):
    if verbose:
        logging.basicConfig(
                    format=f'{role} heartbeat {min(rank, partner)}->{max(rank, partner)}: %(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + world_size + max(rank, partner))
    loginfo(f"Using port {os.environ['MASTER_PORT']} for communication")
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    if role==SENDER:
        init_comm(rank=0, world_size=2, backend=backend) 
        comm_fn = lambda tensor: dist.isend(tensor, dst=1) 
    else: 
        init_comm(rank=1, world_size=2, backend=backend)
        comm_fn = lambda tensor: dist.irecv(tensor, src=0)
    heartbeat_procedure(initial_world_size=world_size, role=role, comm_fn=comm_fn, queue=queue, ppid=signal_pid, device=watch_device, period=communicator_check_period, timeout=communicator_timeout,
                signal_to_send_at_timeout=signal_to_send_at_timeout)



def heartbeat_procedure(initial_world_size, role, comm_fn, queue, ppid, device, period, timeout,
                signal_to_send_at_timeout):
    """
        comm_fn = lambda tensor: dist.isend(tensor, rcv=receiver_rank)
        comm_fn = lambda tensor: dist.irecv(tensor, rcv=receiver_rank)
        comm_fn must return a work object (be async)

        period and timeout are float number of seconds

        device is "cuda:%N%"
    """

    last_comm_time = datetime.datetime.utcnow()
    alive_comm_units = set(range(initial_world_size))
    converter = DatetimeTensorConverter(device=device)
    n_missed_beats = 0
    period_in_seconds = period.total_seconds()

    if role == RECEIVER:
        times_tensor = torch.zeros(shape=(initial_world_size,), device=device, dtype=torch.int64)
        work = comm_fn(times_tensor)

        while True:
            if work.is_completed():
                last_comm_time = datetime.datetime.utcnow()
                loginfo(f"Successful heartbeat! The time is {last_comm_time}")

                comm_datetimes = converter.matrix_to_datetimes(converter.decode_tensor(times_tensor))
                died_comms = {}
                for comm_i, comm_datetime in enumerate(comm_datetimes):
                    if comm_i in alive_comm_units and last_comm_time - comm_datetime > timeout:
                        died_comms.add(comm_i)
                if len(died_comms) > 0: 
                    loginfo(f"Timeout on comms {died_comms} exceeded! Sending signal {signal_to_send_at_timeout} to {ppid}")
                    os.kill(ppid, signal_to_send_at_timeout)
                    alive_comm_units = alive_comm_units.difference(died_comms)


                n_missed_beats = 0
                work = comm_fn(times_tensor)
            else:
                n_missed_beats += 1
                loginfo(f"Heartbeat miss! This is {n_missed_beats}-th time in a row!")
                if n_missed_beats*period > timeout:
                    loginfo(f"Timeout exceeded! Sending signal {signal_to_send_at_timeout} to {ppid}")
                    os.kill(ppid, signal_to_send_at_timeout)
            try:
                put_request = queue["to_heartbeat"].get_nowait()
                if put_request:
                    now_datetime = datetime.datetime.utcnow()
                    timediffs = [now_datetime - last_comm_time for last_comm_time in converter.matrix_to_datetimes(converter.decode_tensor(times_tensor))]
                    loginfo(f"Report requested, sending {timediffs}")
                    queue["from_heartbeat"].put_nowait(timediffs)
            except queue_module.Empty:
                loginfo(f"No report requested, going to sleep for {period_in_seconds} seconds")
                pass
            time.sleep(period_in_seconds)
        

    elif role == SENDER:
        fired_flag = False

        queue["from_heartbeat"].put_nowait(last_comm_time)
        times_tensor = queue["to_heartbeat"].get()
        work = comm_fn(times_tensor)


        while True:
            if work.is_completed():
                last_comm_time = datetime.datetime.utcnow()
                try:
                    queue["from_heartbeat"].get_nowait()
                except queue_module.Empty:
                    pass
                queue["from_heartbeat"].put_nowait(last_comm_time)

                try:
                    times_tensor = queue["to_heartbeat"].get_nowait()
                except queue_module.Empty:
                    pass
                
                loginfo(f"Successful heartbeat! The time is {last_comm_time}")
                n_missed_beats = 0
                work = comm_fn(times_tensor)
            elif not fired_flag:
                n_missed_beats += 1
                loginfo(f"Heartbeat miss! This is {n_missed_beats}-th time in a row!")
                if n_missed_beats*period > timeout:
                    loginfo(f"Timeout exceeded! Sending signal {signal_to_send_at_timeout} to {ppid}")
                    fired_flag = True
                    os.kill(ppid, signal_to_send_at_timeout)
            time.sleep(period_in_seconds)
    else:
        raise NotImplementedError()



   


def get_heartbeat_queues_and_procs(
                            alive_comp_units,
                            rank, 
                            world_size, 
                            backend, 
                            watch_device, 
                            communicator_check_period, 
                            communicator_timeout, 
                            signal_to_send_at_timeout, 
                            verbose,
                            central_rank=0):
    heartbeat_procs = {}
    heartbeat_queues = {}

    ctx = mp.get_context('spawn')
    signal_pid = os.getpid()

    if rank == central_rank:
        

        host_to_curator = ctx.Queue()
        host_from_curator = ctx.Queue()
        queue = {"host_to_curator": host_to_curator, "host_from_curator": host_from_curator}
        heartbeat_queues[CURATOR] = queue
        heartbeat_procs[CURATOR] = ctx.Process(target=curator_starter, 
                    args=(rank, alive_comp_units, world_size, heartbeat_queues, communicator_check_period, communicator_timeout, signal_to_send_at_timeout, watch_device, signal_pid, backend, verbose), 
                    daemon=False)


    else:
        role = RECEIVER
        partner = central_rank
        to_hearbeat = ctx.Queue()
        from_heartbeat = ctx.Queue()
        queue = {"to_hearbeat": to_hearbeat, "from_heartbeat": from_heartbeat}
        heartbeat_queues[partner] = queue
        heartbeat_procs[partner] = ctx.Process(target=heartbeat_starter, 
                    args=(role, rank, partner, world_size, backend, queue, watch_device, signal_pid, communicator_check_period, communicator_timeout, signal_to_send_at_timeout, verbose), 
                    daemon=True)


    return heartbeat_queues, heartbeat_procs


def clean_heartbeat_queues_and_procs(heartbeat_queues, heartbeat_procs, alive_comp_units):
    if CURATOR in heartbeat_procs.keys():
        heartbeat_queues[CURATOR]["host_to_curator"].put_nowait({"type": "clear_request", "content": alive_comp_units})
    else: 
        for partner in list(heartbeat_queues):
            if partner not in alive_comp_units and partner != CURATOR:
                heartbeat_queues[partner]["to_heartbeat"].close()
                heartbeat_queues[partner]["from_heartbeat"].close()
                heartbeat_procs[partner].terminate()
                del heartbeat_queues[partner]
                del heartbeat_procs[partner]

def get_lags(dict_of_timediff_queues):
    if CURATOR in dict_of_timediff_queues.keys():
        dict_of_timediff_queues[CURATOR]["host_to_curator"].put_nowait({"type": "put_request", "content": True})
        timediffs = dict_of_timediff_queues[CURATOR]["host_from_curator"].get()
        

    else:
        lags = {}
        for q in dict_of_timediff_queues.values():
            q["to_heartbeat"].put_nowait(True)
        for comm_name, q in dict_of_timediff_queues.items():
            loginfo(f"Requesting a lag from {comm_name}..")
            lags[comm_name] = q["from_heartbeat"].get()
            loginfo(f"Reported lag of {comm_name} is {lags[comm_name]}")
        return lags

def get_signal_handler(dict_of_queues, former_signal_handler, timeout):
    def handler(signum, stack):
        current_handler = signal.signal(signum, signal.SIG_IGN)
        loginfo(f"Signal {signum} is being handled..")
        lags = get_lags(dict_of_queues)
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