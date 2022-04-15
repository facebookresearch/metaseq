# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


class PartitionedStreamingDataset(torch.utils.data.IterableDataset):
    """Partition an IterableDataset and iterate over a single shard.

    If **drop_last** is ``False``, then the iterator will yield ``None`` for
    shards that don't have data.

    Args:
        dataset (~torch.utils.data.IterableDataset): dataset to partition
        num_shards (int): number of ways to partition the dataset
        shard_id (int): shard index to iterate over
        drop_last (bool, optional): drop the last item (default: False)
    """

    def __init__(
        self,
        dataset: torch.utils.data.IterableDataset,
        num_shards: int,
        shard_id: int,
        drop_last: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.drop_last = drop_last

        assert isinstance(dataset, torch.utils.data.IterableDataset)
        assert num_shards > 0
        assert shard_id >= 0 and shard_id < num_shards

    def set_epoch(self, epoch):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __iter__(self):
        chunk = []
        for item in self.dataset:
            chunk.append(item)
            if len(chunk) == self.num_shards:
                yield chunk[self.shard_id]
                chunk = []
        if len(chunk) > 0 and not self.drop_last:
            if self.shard_id < len(chunk):
                yield chunk[self.shard_id]
            else:
                yield None
