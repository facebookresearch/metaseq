#!/usr/bin/env python
from metaseq.distributed.stitch_fsdp_ckpt import consolidate_fsdp_shards
import fire


if __name__ == "__main__":
    # This is expected to be used before evaluation, not during training.
    fire.Fire(consolidate_fsdp_shards)
