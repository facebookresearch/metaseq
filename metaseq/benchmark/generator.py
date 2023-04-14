# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
import socket
import sys
from typing import Tuple

import numpy as np
import torch
from torch.profiler.profiler import (
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
)
from tqdm import tqdm

from metaseq import options
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as distributed_utils
from metaseq.hub_utils import GeneratorInterface

logging.basicConfig(format="%(asctime)s | %(name)s | %(message)s", level=logging.INFO)
logger: logging.Logger = logging.getLogger("metaseq.benchmark.generator")

WARMUP_STEPS = 5
NUM_REPEATS = 5
generator = None


def main(cfg: MetaseqConfig) -> None:
    global generator
    torch.manual_seed(random.randint(1, 20000))

    logger.info("Instantiating a generator interface and loading a model")
    generator = GeneratorInterface(cfg)
    generator.load_model()

    batch_size = cfg.dataset.batch_size
    input_length = cfg.task.max_source_positions
    output_length = cfg.task.max_target_positions

    logger.info("Running warm-up steps before benchmarking")
    for _ in range(WARMUP_STEPS):
        inputs = np.random.randint(100, 50000, size=(batch_size, input_length))
        generator.generate(inputs, max_tokens=[output_length] * batch_size)

    logger.info(
        f"Benchmarking with batch size {batch_size}, input length "
        f"{input_length}, and output length {output_length}"
    )
    time_elapsed, peak_memory = benchmark(batch_size, input_length, output_length)
    if torch.distributed.get_rank() == 0:
        print(f"Latency: {time_elapsed.mean():.4f} +/- {time_elapsed.std():.4f} ms")
        print(f"Peak memory usage: {peak_memory / 1024 ** 3:.4f} GB")


def benchmark(
    batch_size: int, input_length: int, output_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    time_elapsed = []
    for _ in range(NUM_REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        generator.generate(
            inputs=np.random.randint(100, 50000, size=(batch_size, input_length)),
            max_tokens=[output_length] * batch_size,
            temperature=cfg.generation.temperature,
            n=cfg.generation.beam,
            top_p=0.9,
        )
        end.record()
        torch.cuda.synchronize()
        time_elapsed.append(start.elapsed_time(end))

    peak_memory = torch.tensor(
        torch.cuda.memory_stats()["allocated_bytes.all.peak"],
        device=torch.cuda.current_device(),
    )
    torch.distributed.all_reduce(peak_memory)
    return np.array(time_elapsed), peak_memory.cpu().numpy()


def profile(batch_size: int, input_length: int, output_length: int) -> None:
    tracing_schedule = schedule(skip_first=5, wait=5, warmup=5, active=2, repeat=1)
    worker = socket.gethostname() + "_" + str(torch.distributed.get_rank())
    trace_handler = tensorboard_trace_handler(
        dir_name="traces", worker_name=worker, use_gzip=True
    )

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
        schedule=tracing_schedule,
        on_trace_ready=trace_handler,
        with_stack=True,
    ) as prof:
        for _ in tqdm(range(17)):
            generator.generate(
                inputs=np.random.randint(100, 50000, size=(batch_size, input_length)),
                max_tokens=[output_length] * batch_size,
                temperature=cfg.generation.temperature,
                n=cfg.generation.beam,
                top_p=0.9,
            )
            prof.step()


if __name__ == "__main__":
    """
    Example usage:
        python metaseq/benchmark/generator.py \
        --merges-filename /data/checkpoints/gpt2-merges.txt \
        --vocab-filename /data/checkpoints/gpt2-vocab.json \
        --path /data/checkpoints/opt-125m/reshard-no-os/reshard.pt \
        --model-parallel-size 2 --distributed-world-size 2 \
        --beam 1 --batch-size 4 --max-source-positions 4 --max-target-positions 16
    """

    parser = options.get_generation_parser()
    parser.set_defaults(lr_scheduler=None, criterion=None)
    LAUNCH_ARGS = ["--task language_modeling", "--bpe hf_byte_bpe", "/tmp"]
    launch_args = sys.argv[1:] + [item for arg in LAUNCH_ARGS for item in arg.split()]
    args = options.parse_args_and_arch(parser, input_args=launch_args)

    args.bpe_merges = args.merges_filename
    args.bpe_vocab = args.vocab_filename
    cfg = convert_namespace_to_omegaconf(args)
    distributed_utils.call_main(cfg, main)
