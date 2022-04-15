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

import logging
import os
import queue
import random
import shutil
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import torch
from flask import Flask, request

from metaseq import options
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as dist_utils
from metaseq.eval.hub_utils import GeneratorInterface

app = Flask(__name__)


# global constants
MAX_SEQ_LEN = 2048
BATCH_SIZE = 2048  # silly high bc we dynamically batch by MAX_BATCH_TOKENS
MAX_BATCH_TOKENS = 3072
DEFAULT_PORT = 6010
MODEL_PARALLEL = 8
TOTAL_WORLD_SIZE = 8
BPE_MERGES = "/data/opt/tokenizers/gpt2-merges.txt"
BPE_VOCAB = "/data/opt/tokenizers/gpt2-vocab.json"

# where to find the raw files on nfs
CHECKPOINT_FOLDER = "/data/users/roller/reshard_mp8_175b"
# where to store them on SSD for faster loading
CHECKPOINT_LOCAL = "/mnt/scratch/roller/reshard_mp8_175b/reshard.pt"
# args we want to pass to metaseq's argparser
LAUNCH_ARGS = [
    f"--model-parallel-size {MODEL_PARALLEL}",
    f"--distributed-world-size {TOTAL_WORLD_SIZE}",
    "--task language_modeling",
    f"--bpe-merges {BPE_MERGES}",
    f"--bpe-vocab {BPE_VOCAB}",
    "--bpe hf_byte_bpe",
    f"--path {CHECKPOINT_LOCAL}",
    "--beam 1 --nbest 1",
    "--distributed-port 13000",
    "--checkpoint-shard-count 1",
    "--use-sharded-state",
    f"--batch-size {BATCH_SIZE}",
    f"--buffer-size {BATCH_SIZE * MAX_SEQ_LEN}",
    f"--max-tokens {BATCH_SIZE * MAX_SEQ_LEN}",
    "/tmp",  # required "data" argument.
]


# global state (mutable!)
cfg = None
port = DEFAULT_PORT
BATCH_QUEUE = queue.PriorityQueue()


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("metaseq_cli.interactive")


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


def _normalize_newlines(s):
    # note that web browsers send \r\n but our training data uses \n.
    return s.replace("\r\n", "\n").replace("\r", "\n")


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
            # dynamic batching: group like-sized items to reduce the cost
            # of padding. See PR#20 for additional context.
            item = BATCH_QUEUE.get(timeout=timeout / 1000)
            # accumulate the batch until it gets too big
            longest = max([item] + batch).cost
            batch_cost = longest * (len(batch) + 1)
            if batch and batch_cost > max_tokens:
                # we're over budget, put it back in the queue
                BATCH_QUEUE.put(item)
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


def _generate_worker(encoded_prompt, **generation_args):
    request_object = {"input": encoded_prompt, **generation_args}
    ret_queue = queue.Queue()
    BATCH_QUEUE.put(WorkItem(len(encoded_prompt), 0, ret_queue, request_object))
    _, result = ret_queue.get()
    return result


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
    models = generator.load_model()

    logger.info(f"loaded model {cfg.distributed_training.distributed_rank}")
    request_object = dist_utils.broadcast_object(
        None, src_rank=0, group=dist_utils.get_global_group()
    )
    if torch.distributed.get_rank() == 0:
        logger.info(f"Worker engaged! {_get_my_ip()}:{port}")
        thread = threading.Thread(target=batching_loop, daemon=True)
        thread.start()
        app.run(host="0.0.0.0", port=port, threaded=True)
    else:
        # useful in FSDP setting
        logger.info(f"Looping engaged! {_get_my_ip()}:{port}")
        while True:
            request_object = dist_utils.broadcast_object(
                None, src_rank=0, group=dist_utils.get_global_group()
            )
            _ = generator.generate(**request_object)


# ----- COMMON LOGIC ----
def _get_my_ip():
    return socket.gethostbyname(socket.gethostname())


def _encode_fn(x):
    assert generator.bpe is not None
    return generator.bpe.bpe.encode(x).ids


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
        prompts = [_encode_fn(prompts)]
    assert isinstance(prompts, list)
    assert len(prompts) > 0
    if isinstance(prompts[0], str):
        # multi string
        prompts = [_encode_fn(p) for p in prompts]
    elif isinstance(prompts[0], int):
        # single pre-tokenized
        prompts = [prompts]
    assert isinstance(prompts[0], list)
    # final case: multi pre-tokenized
    assert len(prompts[0]) > 0

    if "stop" in generation_args:
        stop = generation_args["stop"]
        if stop is None:
            pass
        elif isinstance(stop, str):
            stop = [_encode_fn(stop)[0]]
        else:
            stop = [_encode_fn(s)[0] for s in stop]
        generation_args["stop"] = stop

    ret_queue = queue.Queue()
    for i, prompt in enumerate(prompts):
        request_object = {"input": prompt, **generation_args}
        max_len = generation_args.get("max_tokens", 0)
        BATCH_QUEUE.put(WorkItem(len(prompt) + max_len, i, ret_queue, request_object))
    unordered_results = []
    for prompt in prompts:
        unordered_results.append(ret_queue.get())
    # resort results by the original ordering
    # weirdly, openai returns to you a flat list if you gave multiple prompts
    results = []
    for uid, generations in sorted(unordered_results, key=lambda x: x[0]):
        results += generations
    # transform the result into the openai format
    return {
        "id": str(uuid.uuid4()),
        "object": "text_completion",
        "created": int(time.time()),
        "model": CHECKPOINT_LOCAL,
        "choices": [
            {
                "text": result["text"],
                "logprobs": {
                    "tokens": result["tokens"],
                    "token_logprobs": result["token_scores"],
                    "text_offset": result["text_offset"],
                    "top_logprobs": result["top_logprobs"],
                    "finish_reason": "length",  # TODO: implement this
                },
            }
            for result in results
        ],
    }


@app.route("/generate", methods=["POST"])
def generate_view():
    prefix = _normalize_newlines(request.form["prefix"])
    prompt = _encode_fn(prefix)
    length_limit = int(request.form["length"])
    try:
        generation = _generate_worker(
            prompt,
            min_tokens=length_limit,
            max_tokens=length_limit,
            temperature=0.7,
            top_p=0.9,
        )[0]
        generation["prompt"] = prefix  # we don't want tokens lol
        generation["result"] = "success"
        return generation
    except Exception as e:
        raise
        return {
            "result": "error",
            "prompt": prefix,
            "text": f"There was an error on worker: {str(e)}. Tell Stephen.",
        }


@app.route("/")
def index():
    # TODO(roller): decouple demopage.html
    with open("demopage.html") as f:
        return f.read()


def _copy_checkpoint_cache():
    if os.path.exists(os.path.dirname(CHECKPOINT_LOCAL)):
        logger.info("Local checkpoint copy already exists, skipping copy.")
        return
    logger.info("Making a local copy of the checkpoint.")
    shutil.copytree(CHECKPOINT_FOLDER, os.path.dirname(CHECKPOINT_LOCAL))


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
