import socket
import os
from collections import deque
from metaseq.distributed.nccl_watched_comms.utils import ALL_TO_ALL, ALL_TO_ONE

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

def check_required_ports(world_size, n_attempts=3, mode=ALL_TO_ALL):
    main_master_port = int(os.environ["MASTER_PORT"])
    master_addr = os.environ["MASTER_ADDR"] 
    if mode == ALL_TO_ALL:
        diapasone = range(main_master_port+1, main_master_port+world_size*(world_size+1))
    elif mode == ALL_TO_ONE:
        diapasone = range(main_master_port+1, main_master_port+2*world_size)

    required_ports = deque(diapasone)
    for _ in range(n_attempts):
        l = len(required_ports)
        for _ in range(l):
            port = required_ports.popleft()
            if check_a_port(master_addr, port) == 0:
                required_ports.append(port)
    if len(required_ports) > 0:
        raise PortBusyException(required_ports)