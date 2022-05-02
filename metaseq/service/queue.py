# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
from queue import PriorityQueue
import random


class KeyedPriorityQueueCollection:
    """
    Create a collection of priority queues that are ordered by
    a key.  Used for grouping specific types of workers
    """

    def __init__(self):
        self.queues = {}

    def put(self, key, item):
        """
        :param key: key of the queue to put the item into
        :param item: item to add to the queue
        """
        if key not in self.queues:
            self.queues[key] = PriorityQueue()
        self.queues[key].put(item)

    # TODO: this can be a max heap to avoid linear lookup
    def get_largest_queue_key(self):
        """
        ### Returns the key of the queue with the most jobs
        """
        if len(self.queues):
            return max(self.queues, key=lambda key: self.queues[key].qsize())
        else:
            return None

    def get_largest_queue(self):
        """
        ### Returns the queue with the most jobs
        """
        key = self.get_largest_queue_key()
        if key:
            return self.queues[key]
        else:
            return None


class PriorityQueueRingShardKeyable:
    """
    Interface for ensuring that the put method
    has a method to invoke for getting a queue_key
    """

    def queue_key(
        self,
    ) -> str:
        pass


class PriorityQueueRingShard:
    """
    Creates a hashed queue shard, with an
    added deskewing factor for avoiding hot keys (i.e.
    default settings on generation).  The hashing algorithm
    uses either a consistent modulo for bucketing, or in the
    case of deskewing, there is the introduction of a deskew
    factor which is inconsistent but ensures even distribution.
    """

    @staticmethod
    def key_from_dictionary(key_dict):
        """
        :param key_dict: dictionary of keys and values to build shard key from
        """
        return ":".join([f"{k}:{key_dict[k]}" for k in sorted(key_dict.keys())])

    def __init__(self, num_shards=1, deskew_factor=1):
        """
        :param num_shards: total number of shards to hash (i.e. number of workers)
        :param deskew_factor: number of virtual keys per shard.  Reduces key skewing.
        """
        self.num_shards = num_shards
        self.deskew_factor = deskew_factor
        self.deskewing = deskew_factor > 1
        self.queue_shards = [
            KeyedPriorityQueueCollection() for i in range(self.num_shards)
        ]

    def put(self, item: PriorityQueueRingShardKeyable):
        """
        :param key: key of the queue to put the item into
        :param item: item to add to the queue
        """
        key = item.queue_key()
        shard_index = self.get_shard_index_for_key(key)
        self.queue_shards[shard_index].put(key, item)

    def get_shard_index_for_key(self, key):
        """
        ### hashing is deterministic except when deskewing is enabled.
        :param key: the key to be sharded.
        """
        if self.deskewing:
            deskew_offset = random.randint(0, self.deskew_factor * self.num_shards)
            key = f"{deskew_offset}:{key}"
        return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % self.num_shards
