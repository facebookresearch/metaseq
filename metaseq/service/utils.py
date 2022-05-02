# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import socket
import logging
import sys
import os


def normalize_newlines(s: str):
    """
    normalizes new lines, i.e. '\r\n' to '\n'
    """
    # note that web browsers send \r\n but our training data uses \n.
    return s.replace("\r\n", "\n").replace("\r", "\n")


def get_my_ip():
    """
    returns ip / hostname of current host
    """
    return socket.gethostbyname(socket.gethostname())


def encode_fn(generator, x):
    """
    encode a given value to list of bpe tokens
    """
    assert generator.bpe is not None
    return generator.bpe.bpe.encode(normalize_newlines(x)).ids


def build_logger():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    logger = logging.getLogger("metaseq_cli.interactive")
    return logger
