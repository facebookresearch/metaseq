# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from metaseq.service.queue import PriorityQueueRingShard
from dataclasses import dataclass
from typing import Any
import queue
import math


@dataclass
class WorkItem:
    """
    Sortable entry for the batching PriorityQueue.
    """

    cost: int  # lower is serviced first
    uid: int  # unique id to map back to multi-input requests
    return_queue: queue.Queue
    data: Any

    # for sorting / priority queue
    def __lt__(self, other: "WorkItem"):
        return (self.cost, self.uid) < (other.cost, other.uid)

    # for sorting / priority queue
    def __eq__(self, other: "WorkItem"):
        return (self.cost, self.uid) == (other.cost, other.uid)

    def queue_key(self):
        return PriorityQueueRingShard.key_from_dictionary(
            {
                "temperature": self.data["temperature"],
                "top_p": self.data["top_p"],
                "n": self.data["n"],
            }
        )

    @staticmethod
    def generate_worker(encoded_prompt, batch_queue, **generation_args):
        request_object = {"input": encoded_prompt, **generation_args}
        ret_queue = queue.Queue()
        enc_len = len(encoded_prompt)
        cost = enc_len + int(
            math.ceil((enc_len / 10) ** 2)
        )  # account for the cost of both linear and attention layers
        batch_queue.put(WorkItem(cost, 0, ret_queue, request_object))
        _, result = ret_queue.get()
        return result
