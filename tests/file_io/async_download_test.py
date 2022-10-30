#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility for test cached file locking during concurrent downloads.
"""

import array
import logging
import multiprocessing
import os
import random
import re
from tempfile import TemporaryDirectory
import uuid

from metaseq.file_io.azure_blob import AzureBlobPathHandler

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO
)


ENV_SAS_TOKEN = "AZURE_STORAGE_SAS_TOKEN"
ENV_BASE_PATH = "AZURE_TEST_BASE_PATH"


def get_base_path():
    base_path = os.environ.get(ENV_BASE_PATH, None)
    assert base_path is not None, f"Required env var not specified: {ENV_BASE_PATH}"
    # Set a unique base path for each test run
    return os.path.join(base_path, str(uuid.uuid4()))


def create_remote_test_file(handler, path, length=4096):
    value = random.randint(0, 16)
    with handler._open(path, "wb") as f:
        array.array("b", [value] * length).tofile(f)
    return path


def download_files(handler, remote_path, i):
    logging.info(f"Downloading {remote_path} from worker {i}...")
    return handler._get_local_path(remote_path, force=True)


class TestDriver:
    def test(self):
        num_files = 4
        max_workers = 8
        file_size_bytes = 2 * 1024 * 1024
        remote_path_base = get_base_path()

        with TemporaryDirectory() as cache_dir:
            handler = AzureBlobPathHandler(cache_dir=cache_dir)
            remote_paths = []
            try:
                # Create a remote directory with N files
                for i in range(num_files):
                    remote_path = os.path.join(
                        remote_path_base, "testdir", f"file{i}.bin"
                    )
                    create_remote_test_file(
                        handler, remote_path, length=file_size_bytes
                    )
                    remote_paths.append(remote_path)

                # Concurrently download all of them to the same cache_dir
                processes = []
                ctx = multiprocessing.get_context("spawn")
                for i in range(max_workers):
                    process = ctx.Process(
                        target=download_files,
                        args=(handler, os.path.dirname(remote_paths[0]), i),
                    )
                    process.start()
                    processes.append(process)

                # Join
                for process in processes:
                    process.join()

                # Find the expected local path
                m = re.match("(az|blob)://[^/]+/[^/]+/(.+)", remote_paths[0])
                assert m is not None, f"Invalid remote path: {remote_paths[0]}"
                cache_path = os.path.dirname(
                    os.path.join(cache_dir, "blob_cache", m.groups(1)[1])
                )

                # Make sure all files were successfully downloaded
                actual = {f for f in os.listdir(cache_path) if not f.endswith(".lock")}
                expected = {f"file{i}.bin" for i in range(num_files)}
                assert actual == set(expected), f"{expected}\n{actual}"
            except FileNotFoundError:
                from pdb import set_trace

                set_trace()
            finally:
                if remote_paths:
                    logging.info(f"Cleaning up {len(remote_paths)} files...")
                    for remote_path in remote_paths:
                        handler._rm(remote_path)


if __name__ == "__main__":
    assert (
        ENV_SAS_TOKEN in os.environ and ENV_BASE_PATH in os.environ
    ), "Missing required env vars"
    TestDriver().test()
