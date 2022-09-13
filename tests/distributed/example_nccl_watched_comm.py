from metaseq.distributed import nccl_watched_comms as watchpup
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
import logging
import datetime
import sys

def loginfo(msg):
    logging.info(msg)
    sys.stdout.flush()

def main_computation_program(rank, world_size, device):

    tensor = torch.ones((2,2), dtype=torch.int32, device=device)*rank
    loginfo(f"preparing the tensor {tensor} to the broadcast")
    for src in range(world_size):
        time.sleep(4)
        loginfo(f"Initializing broadcast from {src}..")
        work = dist.broadcast(tensor, src=src, async_op=True)
        time.sleep(2)
        loginfo(f"Enabling wait..")
        work.wait()
        time.sleep(2)
        loginfo(f"Synchronization with the host..")
        torch.cuda.synchronize()


def slurm_process_program(SLURM_PROCID, SLURM_LOCALID):
    os.environ["SLURM_PROCID"] = str(SLURM_PROCID)
    os.environ["SLURM_LOCALID"] = str(SLURM_LOCALID)

    n_comp_units = int(os.environ["WORLD_SIZE"])
    comp_unit_id = int(os.environ["SLURM_PROCID"])
    local_id = int(os.environ["SLURM_LOCALID"])

    logging.basicConfig(
        filename=os.environ["LOGFILE_NAME"],
        format=f'slurm_procid {comp_unit_id}: %(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    
    device = f"cuda:{local_id}"

    loginfo(f"Initializing communication..")

    alive_comp_units = list(range(n_comp_units))
    world_size = len(alive_comp_units)
    rank = alive_comp_units.index(comp_unit_id)
    start_time = datetime.datetime.utcnow()
    with watchpup.init_watched_comm(comp_unit_id, n_comp_units, device) as reinit_fn:
        loginfo(f"Communication initialized..")
        while (datetime.datetime.utcnow() - start_time).total_seconds() < 120:
            try:
                loginfo(f"Trying a round of communication..")
                main_computation_program(rank, world_size, device)
            except watchpup.NCCLtimeout as timeout_exception:
                loginfo(f'Detected failure in the ranks {timeout_exception.timeouted_comms}')
                alive_comp_units = list_diff(alive_comp_units, timeout_exception.timeouted_comms)
                world_size = len(alive_comp_units)
                rank = alive_comp_units.index(comp_unit_id)
                # ...
                # do some checkups here
                # ...
                if world_size < 2:
                    loginfo(f"The new world is too small. Cancelling procedure")
                    break
                else:
                    reinit_fn(alive_comp_units)
        loginfo(f"Communication is over.")

def list_diff(a, b):
    return sorted(list(set(a).difference(set(b))))

def get_program_name():
    return __file__.split('/')[-1].split('.')[0]

if __name__ == "__main__":
    logfile_path = 'tmp/' + get_program_name()
    if not os.path.exists(logfile_path):
        os.mkdir(logfile_path)
    logfile_name = logfile_path + f'/_{datetime.datetime.utcnow()}.log'
    logging.basicConfig(
        filename=logfile_name,
        level=logging.INFO)
    loginfo("---------- program code: ----------")
    loginfo(open(__file__).read())
    loginfo("---------- log: ----------")

    num_executors = 2
    master_addr = "127.0.0.1"
    master_port = "12344"

    os.environ["WORLD_SIZE"] = str(num_executors)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["LOGFILE_NAME"] = logfile_name

    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

    processes = []
    ctx = mp.get_context('spawn')
    for slurm_process_id in range(num_executors):
        process = ctx.Process(target=slurm_process_program, args=(slurm_process_id, slurm_process_id%2)) 
        process.start()
        processes.append(process)
    
    time.sleep(30)
    loginfo("External environment is killing the rank 0..")
    import psutil

    _0_proc = processes.pop(0)
    parent = psutil.Process(_0_proc.pid)
    for child in parent.children(recursive=True):  # or parent.children() for recursive=False
        child.terminate()
    _0_proc.terminate()
    loginfo("External environment killed the rank 0")

    join_timeout = 120
    start = time.time()
    while time.time() - start <= join_timeout:
        if not any(p.is_alive() for p in processes):
            # All the processes are done, break now.
            break

        time.sleep(.5)  # Just to avoid hogging the CPU
    else:
    # We only enter this if we didn't 'break' above.
        print("timed out, killing all processes")
        for p in processes:
            p.terminate()
            p.join()