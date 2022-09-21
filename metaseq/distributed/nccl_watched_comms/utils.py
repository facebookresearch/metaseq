import datetime
import sys
import logging
import torch.distributed as dist

ALL_TO_ALL = 'all-to-all'
ALL_TO_ONE = 'all-to-one'

RECEIVER = "recv"
SENDER = "send"

def loginfo(msg):
    logging.info(msg)
    sys.stdout.flush()


def init_comm(rank, world_size, backend="nccl", timeout=datetime.timedelta(seconds=40)):

    dist.init_process_group(backend=backend,
                                    world_size=world_size, rank=rank,
                                    timeout=timeout)
    dist.barrier() # Initialize communication! (otherwise the first async op will be blocking)


class NCCLtimeout(Exception):
    def __init__(self, timeouted_comms, *args: object) -> None:
        super().__init__(*args)
        self.timeouted_comms = timeouted_comms


class NotGPUfailRelatedTimeout(Exception):
    pass

