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
import sys
import logging

import torch

from metaseq import options
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as distributed_utils
from metaseq.hub_utils import GeneratorInterface
from metaseq.service.constants import (
    TOTAL_WORLD_SIZE,
    LAUNCH_ARGS,
)
from metaseq.service.utils import encode_fn, build_logger

logger = build_logger()


def input_loop():
    inp = []
    while True:
        try:
            # green display, bold user prompt
            display = (
                "\033[32mPrompt (ctrl-D to end input, ctrl-C to quit):\n\033[0;1m"
                if not inp
                else ""
            )
            data = input(display)
            inp.append(data)
        except KeyboardInterrupt:
            # reset the formatting
            sys.stdout.write("\033[0m")
            raise
        except EOFError:
            break
        # reset the formatting
        sys.stdout.write("\033[0m")
    logger.debug(f"Input: {inp}")
    return "\n".join(inp)


def worker_main(cfg: MetaseqConfig, namespace_args=None):
    global generator
    # make sure generations are stochastic since we have many workers
    torch.manual_seed(random.randint(1, 20000))
    torch.cuda.manual_seed(random.randint(1, 20000))

    generator = GeneratorInterface(cfg)
    models = generator.load_model()  # noqa: F841

    # quiet some of the stuff for visual aspects
    logging.getLogger("metaseq.hub_utils").setLevel(logging.WARNING)

    logger.info(f"loaded model {cfg.distributed_training.distributed_rank}")
    request_object = distributed_utils.broadcast_object(
        None, src_rank=0, group=distributed_utils.get_global_group()
    )
    if torch.distributed.get_rank() == 0:
        while True:
            prompt = input_loop()
            tokens = encode_fn(generator, prompt)
            request_object = {
                "inputs": [tokens],
                "max_tokens": [128],
            }
            distributed_utils.broadcast_object(
                request_object, src_rank=0, group=distributed_utils.get_global_group()
            )
            generations = generator.generate(**request_object)
            print(generations[0][0]["text"])
    else:
        # useful in FSDP setting
        while True:
            request_object = distributed_utils.broadcast_object(
                None, src_rank=0, group=distributed_utils.get_global_group()
            )
            _ = generator.generate(**request_object)


def cli_main():
    """
    Command line interactive.
    """
    parser = options.get_generation_parser()
    # dumb defaults overriding
    parser.set_defaults(lr_scheduler=None, criterion=None)
    flat_launch_args = []
    for s in LAUNCH_ARGS:
        flat_launch_args += s.split()
    args = options.parse_args_and_arch(parser, input_args=flat_launch_args)
    args.data = os.path.dirname(args.path)  # hardcode the data arg
    cfg = convert_namespace_to_omegaconf(args)
    cfg.distributed_training.distributed_world_size = TOTAL_WORLD_SIZE
    distributed_utils.call_main(cfg, worker_main, namespace_args=args)


if __name__ == "__main__":
    cli_main()
