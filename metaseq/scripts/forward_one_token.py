#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Host the demo.

Launch with `python -m metaseq_cli.interactive_hosted` to run locally.

See docs/api.md for more information.
"""

import os
import random

import torch

from metaseq import options
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as dist_utils
from metaseq.hub_utils import GeneratorInterface
from metaseq.service.queue import PriorityQueueRingShard
from metaseq.service.constants import (
    MAX_SEQ_LEN,
    MAX_BATCH_TOKENS,
    DEFAULT_PORT,
    TOTAL_WORLD_SIZE,
    CHECKPOINT_LOCAL,
    CHECKPOINT_FOLDER,
    LAUNCH_ARGS,
)
from metaseq.service.utils import build_logger
from metaseq.service.utils import encode_fn
import torch.profiler as profiler

port = DEFAULT_PORT
BATCH_QUEUE = PriorityQueueRingShard()

logger = build_logger()

@torch.no_grad()
def worker_main(cfg: MetaseqConfig, namespace_args=None):
    # disable multithreading in tokenizers and torch, as different Flask threads
    # may then fight for resources.
    # make sure generations are stochastic since we have many workers
    torch.manual_seed(random.randint(1, 20000))
    torch.cuda.manual_seed(random.randint(1, 20000))
    generator = GeneratorInterface(cfg)
    models = generator.load_model()  # noqa: F841
    logger.info("loaded model")

    static_input_tokens = torch.randint(4, 50000, (1, 1), device='cuda')
    # static_input_tokens = torch.tensor(encode_fn(generator, 'let us test this out'), dtype=torch.int, device='cuda').unsqueeze(0)
    logger.info("forwarded token")
    model = models[0].decoder
    model.eval()

    incremental_states = []
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(s):
        for i in range(10):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            # with profiler.profile() as prof:
            output = model(static_input_tokens)

            end.record()
            torch.cuda.synchronize()
            time = start.elapsed_time(end)
            # Params have been updated. static_y_pred, static_loss, and .grad
            # attributes hold values from computing on this iteration's data.
            logger.info(time)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    # Sets grads to None before capture, so backward() will create
    # .grad attributes with allocations from the graph's private pool

    with torch.cuda.graph(g):
        model(static_input_tokens)
    output = model(static_input_tokens)

    real_inputs = [torch.randint_like(static_input_tokens, 4, 50000) for _ in range(10)]

    for data in real_inputs:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # Fills the graph's input memory with new data to compute on
        static_input_tokens.copy_(data)
        # replay() includes forward, backward, and step.
        # You don't even need to call optimizer.zero_grad() between iterations
        # because the captured backward refills static .grad tensors in place.
        # with profiler.profile() as prof:
        g.replay()

        end.record()
        torch.cuda.synchronize()
        # prof.export_chrome_trace(
        #     os.path.join(cfg.checkpoint.save_dir, "profiler_trace.json")
        # )

        time = start.elapsed_time(end)
        # Params have been updated. static_y_pred, static_loss, and .grad
        # attributes hold values from computing on this iteration's data.
        logger.info("replay done")
        logger.info(time)


def _forward_custom_steps(model, static_input_tokens, start_step, end_step, incremental_state):

    return model(
        static_input_tokens[:, start_step: end_step],
        incremental_state=incremental_state
    )


def cli_main():
    """
    Hosted version of the web UI for generation.
    """
    # _copy_checkpoint_cache()

    global port, MODE, cfg
    parser = options.get_generation_parser()

    # dumb defaults overriding
    parser.set_defaults(lr_scheduler=None, criterion=None)
    flat_launch_args = []
    for s in LAUNCH_ARGS:
        flat_launch_args += s.split()
    args = options.parse_args_and_arch(parser, input_args=flat_launch_args)
    args.data = os.path.dirname(args.path)  # hardcode the data arg
    port = DEFAULT_PORT
    cfg = convert_namespace_to_omegaconf(args)
    cfg.distributed_training.distributed_world_size = TOTAL_WORLD_SIZE
    dist_utils.call_main(cfg, worker_main, namespace_args=args)


if __name__ == "__main__":
    cli_main()
