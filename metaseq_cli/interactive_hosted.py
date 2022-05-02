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
import queue
import pkg_resources
import random
import shutil
import threading

import torch
from flask import Flask, request

from metaseq import options
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as dist_utils
from metaseq.hub_utils import GeneratorInterface
from metaseq.service.queue import PriorityQueueRingShard
from metaseq.service.workers import WorkItem
from metaseq.service.constants import (
    MAX_SEQ_LEN,
    MAX_BATCH_TOKENS,
    DEFAULT_PORT,
    TOTAL_WORLD_SIZE,
    CHECKPOINT_LOCAL,
    CHECKPOINT_FOLDER,
    LAUNCH_ARGS,
)
from metaseq.service.utils import get_my_ip, encode_fn, build_logger
from metaseq.service.responses import OAIResponse


app = Flask(__name__)

# global state (mutable!)
cfg = None
port = DEFAULT_PORT
BATCH_QUEUE = PriorityQueueRingShard()

logger = build_logger()


def batching_loop(timeout=100, max_tokens=MAX_BATCH_TOKENS):
    """
    batching_loop is an infinite loop responsible for executing generations.

    GPUs benefit from batching requests, but we expect workloads to come
    in non-uniformly. This loop groups requests together (via BATCH_QUEUE)
    and executes them in one batch. In order to keep latency low, unfilled
    batches are executed within a window of :timeout: milliseconds.

    batching_loop also performs dynamic batching, in order to minimize the
    amount of padding by grouping like-sized workloads together. As a result
    batching loop will provide preferential treatment to smaller workloads.  At
    the current moment, there is no TTL logic to ensure a maximum wait time.

    For a rough overview of dynamic batching, see
    https://parl.ai/docs/tutorial_worlds.html#dynamic-batching.

    :param timeout: The max queue time before a non-full batch is launched.
    :param max_tokens: the maximum number of tokens that can be processed
        concurrently. model specific and empirical.
    """
    # TODO(roller):
    # - group by generation type, topp etc, as we cannot share these
    # - modify timeout logic to be cumulative
    global BATCH_QUEUE

    batch = []
    while True:
        try:
            # for now, we only have 1 worker, so can always index to shard 0
            target_queue = BATCH_QUEUE.queue_shards[0].get_largest_queue()
            if not target_queue:
                continue
            # dynamic batching: group like-sized items to reduce the cost
            # of padding. See PR#20 for additional context.
            item = target_queue.get(timeout=timeout / 1000)
            # accumulate the batch until it gets too big
            longest = max([item] + batch).cost
            batch_cost = longest * (len(batch) + 1)
            if batch and batch_cost > max_tokens:
                # we're over budget, put it back in the queue
                target_queue.put(item)
                raise queue.Empty
            else:
                # batch is empty or under budget
                batch.append(item)
        except queue.Empty:
            if batch:
                request_object = {
                    "inputs": [],
                    "min_tokens": [],
                    "max_tokens": [],
                }
                for work_item in batch:
                    ro = work_item.data
                    request_object["inputs"].append(ro["input"])
                    request_object["min_tokens"].append(ro.get("min_tokens", 0))
                    request_object["max_tokens"].append(
                        ro.get("max_tokens", MAX_SEQ_LEN)
                    )
                    # assumption: everyone has the same remaining args
                    for key in [
                        "temperature",
                        "top_p",
                        "n",
                        "best_of",
                        "echo",
                        "logprobs",
                        "stop",
                    ]:
                        if key in ro:
                            request_object[key] = ro[key]
                # do the actual generations
                request_object["seed"] = random.randint(1, 20000)
                dist_utils.broadcast_object(
                    request_object, src_rank=0, group=dist_utils.get_global_group()
                )
                generations = generator.generate(**request_object)
                # broadcast them back
                for work_item, gen in zip(batch, generations):
                    work_item.return_queue.put((work_item.uid, gen))

                batch.clear()
            else:
                # back to the loop
                continue


def worker_main(cfg1: MetaseqConfig, namespace_args=None):
    # disable multithreading in tokenizers and torch, as different Flask threads
    # may then fight for resources.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    global generator
    global MODE
    # make sure generations are stochastic since we have many workers
    torch.manual_seed(random.randint(1, 20000))
    torch.cuda.manual_seed(random.randint(1, 20000))
    MODE = "worker"
    cfg = cfg1

    generator = GeneratorInterface(cfg)
    models = generator.load_model()  # noqa: F841

    logger.info(f"loaded model {cfg.distributed_training.distributed_rank}")
    request_object = dist_utils.broadcast_object(
        None, src_rank=0, group=dist_utils.get_global_group()
    )
    if torch.distributed.get_rank() == 0:
        logger.info(f"Worker engaged! {get_my_ip()}:{port}")
        thread = threading.Thread(target=batching_loop, daemon=True)
        thread.start()
        app.run(host="0.0.0.0", port=port, threaded=True)
    else:
        # useful in FSDP setting
        logger.info(f"Looping engaged! {get_my_ip()}:{port}")
        while True:
            request_object = dist_utils.broadcast_object(
                None, src_rank=0, group=dist_utils.get_global_group()
            )
            _ = generator.generate(**request_object)


@app.route("/completions", methods=["POST"])
@app.route("/v1/engines/<engine>/completions", methods=["POST"])
@app.route("/v2/engines/<engine>/completions", methods=["POST"])
@app.route("/engines/<engine>/completions", methods=["POST"])
def completions(engine=None):
    # prompt can be 4 types:
    # - str. Basic case. Return one generation.
    # - list of ints. Pretokenized. Return one generation
    # - list of str. Multiple generations, one per prompt
    # - list of list of ints. Pretokenized multiple generations.

    # our approach is to turn everything into the last case

    prompts = request.json["prompt"]
    del request.json["prompt"]
    generation_args = request.json

    if isinstance(prompts, str):
        # single string. tokenize and turn it to the single pre-tokenized case
        prompts = [encode_fn(generator, prompts)]
    assert isinstance(prompts, list)
    assert len(prompts) > 0
    if isinstance(prompts[0], str):
        # multi string
        prompts = [encode_fn(generator, p) for p in prompts]
    elif isinstance(prompts[0], int):
        # single pre-tokenized
        prompts = [prompts]
    assert isinstance(prompts[0], list)
    # final case: multi pre-tokenized
    assert len(prompts[0]) > 0

    if "min_tokens" in generation_args:
        generation_args["min_tokens"] = int(generation_args["min_tokens"])
    if "max_tokens" in generation_args:
        generation_args["max_tokens"] = int(generation_args["max_tokens"])
    if "stop" in generation_args:
        stop = generation_args["stop"]
        if stop is None:
            pass
        elif isinstance(stop, str):
            stop = [encode_fn(generator, stop)[0]]
        else:
            stop = [encode_fn(generator, s)[0] for s in stop]
        generation_args["stop"] = stop
    if "temperature" in generation_args:
        generation_args["temperature"] = round(float(generation_args["temperature"]), 1)
    else:
        generation_args["temperature"] = 1.0
    if "top_p" in generation_args:
        generation_args["top_p"] = round(float(generation_args["top_p"]), 1)
    else:
        generation_args["top_p"] = 1.0
    # beam search top n
    if "n" in generation_args:
        generation_args["n"] = int(generation_args["n"])
    else:
        generation_args["n"] = 1

    ret_queue = queue.Queue()
    for i, prompt in enumerate(prompts):
        request_object = {"input": prompt, **generation_args}
        max_len = generation_args.get("max_tokens", 0)
        BATCH_QUEUE.put(WorkItem(len(prompt) + max_len, i, ret_queue, request_object))
    unordered_results = []
    for _ in prompts:
        unordered_results.append(ret_queue.get())
    # resort results by the original ordering
    # weirdly, openai returns to you a flat list if you gave multiple prompts
    reordered = sorted(unordered_results, key=lambda x: x[0])
    results = []
    for prompt, (_, generations) in zip(prompts, reordered):
        results += generations
    # transform the result into the openai format
    return OAIResponse(results).__dict__()


@app.route("/")
def index():
    # TODO(roller): decouple demopage.html
    fn = pkg_resources.resource_filename("metaseq", "service/index.html")
    with open(fn) as f:
        return f.read()


def _copy_checkpoint_cache():
    if CHECKPOINT_LOCAL == CHECKPOINT_FOLDER:
        # user didn't have a local SSD
        return
    if os.path.exists(os.path.dirname(CHECKPOINT_LOCAL)):
        logger.info("Local checkpoint copy already exists, skipping copy")
    else:
        logger.info(
            f"Making a local copy of the checkpoint. {CHECKPOINT_FOLDER} -> {CHECKPOINT_LOCAL}"
        )
        shutil.copytree(CHECKPOINT_FOLDER, os.path.dirname(CHECKPOINT_LOCAL))

    dict_path = os.path.join(os.path.dirname(CHECKPOINT_LOCAL), "dict.txt")
    if not os.path.exists(dict_path):
        with open(dict_path, "w+") as f:
            f.write("\n".join([f"{i} 1" for i in range(4, 50271 + 1)]))
        logger.info("Hackishly generated a dict.txt for use with 175B model")


def cli_main():
    """
    Hosted version of the web UI for generation.
    """
    _copy_checkpoint_cache()

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
