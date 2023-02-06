# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import logging
from typing import Dict
from urllib.parse import urlparse

from torch.distributed.constants import default_pg_timeout
from torch.distributed import register_rendezvous_handler, Store, TCPStore, rendezvous


RETRIES = 5
COOLDOWN = 0.25

logger = logging.getLogger(__name__)


def _create_c10d_store(hostname, port, rank, world_size, timeout) -> Store:
    # check if port is uint16_t
    if not 0 <= port < 2**16:
        raise ValueError(f"port must have value from 0 to 65535 but was {port}.")

    start_daemon = rank == 0
    return TCPStore(
        hostname, port, world_size, start_daemon, timeout, multi_tenant=True
    )


def _query_to_dict(query: str) -> Dict[str, str]:
    return dict(
        (pair[0], pair[1])
        for pair in (pair.split("=") for pair in filter(None, query.split("&")))
    )


def _tcp_retry_rendezvous_handler(url: str, timeout=default_pg_timeout, **kwargs):
    def _error(msg):
        return ValueError("tcpr:// rendezvous: " + msg)

    result = urlparse(url)
    if not result.port:
        raise _error("port number missing")
    query_dict = _query_to_dict(result.query)
    if "rank" not in query_dict:
        raise _error("rank parameter missing")
    if "world_size" not in query_dict:
        raise _error("world size parameter missing")

    rank = int(query_dict["rank"])
    world_size = int(query_dict["world_size"])
    assert result.hostname is not None

    for tries in range(1, RETRIES + 1):
        try:
            store = _create_c10d_store(
                result.hostname, result.port, rank, world_size, timeout
            )
            logger.warning(
                f"Successfully connected to primary node on attempt {tries}/{RETRIES}"
            )
            yield (store, rank, world_size)
            break
        except RuntimeError as re:
            logger.error(
                f"Failed to connect to primary node on attempt {tries}/{RETRIES}: {re}"
            )
            time.sleep(COOLDOWN * (2**tries))

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform re-rendezvous using tcpr:// method")


STORE_BASED_BARRIER_MARKER = "store_based_barrier_key:"


class ComplicitStore(Store):
    def __init__(self, store: Store, world_size: int):
        super().__init__()
        self.store = store
        self.world_size = world_size

    def set(self, *args, **kwargs):
        return self.store.set(*args, **kwargs)

    def get(self, *args, **kwargs):
        return self.store.get(*args, **kwargs)

    def add(self, key: str, *args, **kwargs):
        # The marker isn't always a prefix: it's sometimes prefixed with default_pg/.
        if STORE_BASED_BARRIER_MARKER in key:
            return self.world_size
        return self.store.add(key, *args, **kwargs)

    def compare_set(self, *args, **kwargs):
        return self.store.compare_set(*args, **kwargs)

    def delete_key(self, *args, **kwargs):
        return self.store.delete_key(*args, **kwargs)

    def num_keys(self, *args, **kwargs):
        return self.store.num_keys(*args, **kwargs)

    def set_timeout(self, *args, **kwargs):
        return self.store.set_timeout(*args, **kwargs)

    def wait(self, *args, **kwargs):
        return self.store.wait(*args, **kwargs)


STORE = None


def _tcp_retry_barrierless_rendezvous_handler(
    url: str, timeout=default_pg_timeout, **kwargs
):
    assert url.startswith("barrierlesstcpr://")
    url = url.replace("barrierlesstcpr://", "tcpr://")
    rendezvous_iterator = rendezvous(url, timeout=timeout)
    store, rank, world_size = next(rendezvous_iterator)
    global STORE
    STORE = ComplicitStore(store, world_size)
    return iter([(STORE, rank, world_size)])


register_rendezvous_handler("tcpr", _tcp_retry_rendezvous_handler)
register_rendezvous_handler(
    "barrierlesstcpr", _tcp_retry_barrierless_rendezvous_handler
)
