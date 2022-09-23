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
ALIVE = 'alive'
DEAD = 'dead'

class EmptyWorldException(Exception):
    pass

class Partner:
    def __init__(self, id, role, status = ALIVE, heartbeat_process=None, heartbeat_queues = None) -> None:
        self.comp_unit_id = id
        self.status = status
        self.role = role
        self.heartbeat_process = heartbeat_process
        self.heartbeat_queues = heartbeat_queues

# This code relies significantly on the assumption that the devices in the network have synchronized clocks

def curator_starter(self_id, 
                    partners, 
                    world_size, 
                    heartbeat_queues, 
                    communicator_check_period, 
                    communicator_timeout, 
                    signal_to_send_at_timeout, 
                    watch_device, 
                    signal_pid, 
                    backend, 
                    verbose=False):
    if verbose:
        logging.basicConfig(
                    format=f'{CURATOR} heartbeat on {self_id}: %(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

    ctx = mp.get_context('spawn')

    for partner_id, partner in partners.items():
        if partner.role is not None:
            curator_to_hearbeat = ctx.Queue()
            curator_from_heartbeat = ctx.Queue()
            queue = {"to_heartbeat": curator_to_hearbeat, "from_heartbeat": curator_from_heartbeat}
            heartbeat_process = ctx.Process(target=heartbeat_starter, 
                    args=(partner.role, self_id, partner_id, world_size, backend, queue, watch_device, communicator_check_period, verbose), 
                    daemon=True)
            partner.heartbeat_process = heartbeat_process
            partner.heartbeat_queues = queue

    for partner in partners.values():
        if partner.role is not None:
            partner.heartbeat_process.start()

    curator_procedure(self_id=self_id, all_partners=partners, queues_to_host=heartbeat_queues[CURATOR], device=watch_device, 
                        period=communicator_check_period, 
                        signal_pid = signal_pid,
                        communicator_timeout=communicator_timeout, 
                        signal_to_send_at_timeout=signal_to_send_at_timeout)

def try_to_get_from(queue):
    try:
        return queue.get_nowait()
    except queue_module.Empty:
        pass

def multiclass_one_hot(inp, num_classes, device):
    """
    test:
    print(multiclass_one_hot([1,2,5,6], num_classes=7))
    print(multiclass_one_hot([[1,2,5,6], [4,3,5]], num_classes=7))

    out:
    tensor([0, 1, 1, 0, 0, 1, 1])
    tensor([[0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 0]])

    """
    if len(inp) > 0:
        try:
            len(inp[0])
            result = []
            for item in inp:
                onehot = torch.nn.functional.one_hot(torch.tensor(item, dtype=torch.int64, device=device), num_classes=num_classes)
                onehot = onehot.sum(dim=0)
                result.append(onehot)
            return torch.stack(result)
        except TypeError:
            onehot = torch.nn.functional.one_hot(torch.tensor(inp, dtype=torch.int64, device=device), num_classes=num_classes)
            onehot = onehot.sum(dim=0)
            return onehot
    else:
        return None

def clear_timeouted_partners(partners_dict, timeouted_partner_ids):
    for partner in partners_dict.values():
        if partner.comp_unit_id in timeouted_partner_ids:
            partner.status = DEAD
            if partner.role is not None:
                partner.heartbeat_process.terminate()
                partner.heartbeat_process.join()
                partner.heartbeat_queues["to_heartbeat"].close()
                partner.heartbeat_queues["from_heartbeat"].close()

def put_replace(queue, obj):
    try_to_get_from(queue)
    if isinstance(obj, torch.Tensor):
        obj = obj.to('cpu')
    queue.put_nowait(obj)

def try_fmax_update(queue, tensor):
    tensor_update = try_to_get_from(queue)
    if tensor_update is not None:
        tensor_update = tensor_update.to(tensor.device)
        torch.fmax(tensor, tensor_update, out=tensor)


def curator_procedure(self_id, all_partners, queues_to_host, device, period, signal_pid, communicator_timeout, signal_to_send_at_timeout):
    alive_comp_units = sorted(list(all_partners.keys()))
    initial_world_size = len(alive_comp_units) + 1
    alive_units_mask = multiclass_one_hot(alive_comp_units, num_classes=initial_world_size, device=device)
    converter = DatetimeTensorConverter(device=device)
    period_in_seconds = period.total_seconds()
    timeout = torch.tensor(communicator_timeout.total_seconds(), device=device)*(10**6)

    times_tensor = converter.encode_tensor(converter.join_tensors([converter.utc_tensor() for _ in range(initial_world_size)]))

    while True:
        for partner in all_partners.values():
            if partner.status == ALIVE and partner.role is not None:
                try_fmax_update(partner.heartbeat_queues["from_heartbeat"], times_tensor)
                put_replace(partner.heartbeat_queues["to_heartbeat"], times_tensor)
                    

        now_tensor = converter.encode_tensor(converter.utc_tensor())
        timediff_tensor = torch.sub(now_tensor, times_tensor)
        torch.mul(timediff_tensor, alive_units_mask, out=timediff_tensor)
        timeouted_partners = (timediff_tensor > timeout).nonzero()
        if len(timeouted_partners) > 0:
            timeouted_partners = set(timeouted_partners.flatten().tolist())
            loginfo(f"Timeout exceeded! Timeouted partners are {timeouted_partners}. The lags in us are {timediff_tensor}")
            os.kill(signal_pid, signal_to_send_at_timeout)
            alive_comp_units = sorted(list(set(alive_comp_units).difference(timeouted_partners)))
            if len(alive_comp_units) == 0:
                raise EmptyWorldException()
            alive_units_mask = multiclass_one_hot(alive_comp_units, num_classes=initial_world_size, device=device)
            clear_timeouted_partners(partners_dict=all_partners, timeouted_partner_ids=timeouted_partners)

        loginfo(f"Putting alive_comp_units: {alive_comp_units}")

        put_replace(queues_to_host["host_from_curator"], alive_comp_units)
        time.sleep(period_in_seconds)


def heartbeat_starter(role, self_id, partner_id, world_size, backend, queue, watch_device, communicator_check_period, verbose=False):
    if verbose:
        logging.basicConfig(
                    format=f'{role} heartbeat {min(self_id, partner_id)}->{max(self_id, partner_id)}: %(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + world_size + max(self_id, partner_id))
    loginfo(f"Using port {os.environ['MASTER_PORT']} for communication")
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    def inner_comm_fn(tensor):
        try_fmax_update(queue["to_heartbeat"], tensor)
        put_replace(queue["from_heartbeat"], tensor)
    if role==SENDER:
        init_comm(rank=0, world_size=2, backend=backend) 
        outer_comm_fn = lambda tensor: dist.isend(tensor, dst=1) 
    else: 
        init_comm(rank=1, world_size=2, backend=backend)
        outer_comm_fn = lambda tensor: dist.irecv(tensor, src=0)

    self_id_indicator = torch.nn.functional.one_hot(torch.tensor(self_id, dtype=torch.int64, device=watch_device), num_classes=world_size)
    partner_indicator = torch.nn.functional.one_hot(torch.tensor(partner_id, dtype=torch.int64, device=watch_device), num_classes=world_size)
    last_comm_time_multiplier = self_id_indicator + partner_indicator
    
    heartbeat_procedure(initial_world_size=world_size, last_comm_time_multiplier=last_comm_time_multiplier, 
                        inner_comm_fn=inner_comm_fn, outer_comm_fn=outer_comm_fn, 
                        device=watch_device, 
                        period=communicator_check_period)


def heartbeat_procedure(initial_world_size, last_comm_time_multiplier, inner_comm_fn, outer_comm_fn, device, period):
    """
        outer_comm_fn = lambda tensor: dist.isend(tensor, rcv=receiver_rank)
        outer_comm_fn = lambda tensor: dist.irecv(tensor, rcv=receiver_rank)
        outer_comm_fn must return a work object (be async)

        period and timeout are float number of seconds

        device is "cuda:%N%"
    """

    converter = DatetimeTensorConverter(device=device)
    period_in_seconds = period.total_seconds()

    times_tensor = converter.encode_tensor(converter.join_tensors([converter.utc_tensor() for _ in range(initial_world_size)]))

    work = outer_comm_fn(times_tensor)

    while True:
        if work.is_completed():
            last_comm_time = converter.encode_tensor(converter.utc_tensor())
            loginfo(f"Successful heartbeat! The time is {last_comm_time}")
            torch.fmax(times_tensor, last_comm_time_multiplier*last_comm_time, out=times_tensor)
            work = outer_comm_fn(times_tensor)

        inner_comm_fn(times_tensor)

        time.sleep(period_in_seconds)

def get_heartbeat_queues_and_procs(
                            partners,
                            rank, 
                            world_size, 
                            backend, 
                            watch_device, 
                            communicator_check_period, 
                            communicator_timeout, 
                            signal_to_send_at_timeout, 
                            verbose=False,
                            central_comp_unit_id=0):
    alive_comp_units = partners
    self_id = rank

    heartbeat_procs = {}
    heartbeat_queues = {}

    ctx = mp.get_context('spawn')
    signal_pid = os.getpid()

    partners = {} 
    for partner_id in alive_comp_units:
        if self_id == central_comp_unit_id:
            role = SENDER
        else:
            if partner_id == central_comp_unit_id:
                role = RECEIVER
            else:
                role = None
        partners[partner_id] = Partner(partner_id, role)

    host_to_curator = ctx.Queue()
    host_from_curator = ctx.Queue()
    queue = {"host_to_curator": host_to_curator, "host_from_curator": host_from_curator}
    heartbeat_queues[CURATOR] = queue
    heartbeat_procs[CURATOR] = ctx.Process(target=curator_starter, 
                    args=(self_id, 
                    partners, 
                    world_size, 
                    heartbeat_queues, 
                    communicator_check_period, 
                    communicator_timeout, 
                    signal_to_send_at_timeout, 
                    watch_device, 
                    signal_pid, 
                    backend, 
                    verbose), 
                    daemon=False)

    return heartbeat_queues, heartbeat_procs


def clean_heartbeat_queues_and_procs(heartbeat_queues, heartbeat_procs, alive_comp_units):
    pass

def get_signal_handler(dict_of_queues, former_signal_handler, timeout):
    def handler(signum, stack):
        current_handler = signal.signal(signum, signal.SIG_IGN)
        loginfo(f"Signal {signum} is being handled..")
        alive_comp_units = dict_of_queues[CURATOR]["host_from_curator"].get()
        signal.signal(signum, current_handler)
        loginfo(f"raising NCCLtimeout with alive_comp_units: {alive_comp_units}")
        raise NCCLtimeout(alive_comp_units=alive_comp_units)

    return handler