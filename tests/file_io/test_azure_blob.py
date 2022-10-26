# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import array
import hashlib
import logging
import os
import random
import re
import unittest
import uuid
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import NamedTuple


try:
    import azure.storage.blob as azure_blob
    from metaseq.file_io.azure_blob import AzureBlobPathHandler
except ImportError:
    azure_blob = None
    AzureBlobPathHandler = None


logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO
)

ENV_SAS_TOKEN = "AZURE_STORAGE_SAS_TOKEN"
ENV_BASE_PATH = "AZURE_TEST_BASE_PATH"


if azure_blob:

    class TestContext(NamedTuple):
        handler: AzureBlobPathHandler
        container_path: str
        container_name: str
        base_path: str
        cache_dir: str

        def get_local_test_file(self, name, length=4096):
            path = os.path.join(self.cache_dir, name)
            value = random.randint(0, 16)
            with open(path, "wb") as f:
                array.array("b", [value] * length).tofile(f)
            return path

        def cleanup_remote(self):
            for relpath in self.handler._ls(self.base_path):
                file = os.path.join(self.container_path, relpath)
                self.handler._rm(file)

else:
    EnvironmentTokenProvider = None
    TestContext = None


@unittest.skipIf(not azure_blob, "Requires azure-blob-storage install")
@unittest.skipIf(
    not os.environ.get(ENV_SAS_TOKEN, None),
    f"Required env var not specified: {ENV_SAS_TOKEN}",
)
@unittest.skipIf(
    not os.environ.get(ENV_BASE_PATH, None),
    f"Required env var not specified: {ENV_BASE_PATH}",
)
class TestAzureBlob(unittest.TestCase):
    """
    Tests rely on access to Blob Storage as specified in 2 environment variables:
        - AZURE_STORAGE_SAS_TOKEN: SAS token providing read/write authorization
        - AZURE_TEST_BASE_PATH: base path for all read/write operations. Should be in the form:
            az://<account_name>/<container_name>/<relative_path>/

    Any artifacts written under AZURE_TEST_BASE_PATH will be cleaned up on a best effort basis,
    to the extent AzureBlobPathHandler._rm() is in working condition.
    """

    def get_base_path(self):
        base_path = os.environ.get(ENV_BASE_PATH, None)
        assert base_path is not None, f"Required env var not specified: {ENV_BASE_PATH}"
        # Set a unique base path for each test run
        return os.path.join(base_path, str(uuid.uuid4()))

    def get_container_path(self, base_path):
        m = re.match("((az|blob)://[^/]+/[^/]+).+", base_path)
        assert m is not None, (
            f"Invalid base path: '{base_path}'."
            + " Expected 'az://<account_name>/<container_name>/<relative_path>/'"
        )
        return m.groups(1)[0] + "/"

    def get_md5(self, local_path):
        with open(local_path, "rb") as f:
            return self.get_md5_from_file(f)

    def get_md5_from_file(self, f):
        return hashlib.md5(f.read()).hexdigest()

    @contextmanager
    def context(self) -> TestContext:
        """
        Initializes a test context:
        1. Create a cache_dir on local disk that will be cleaned up on exit
        2. Initialize the AzureBlobPathHandler with a SAS token from env variables
        3. Tell the handler to close BlobServiceClient on exit so we don't leak sockets
        """
        with TemporaryDirectory() as cache_dir:
            try:
                handler = AzureBlobPathHandler(cache_dir=cache_dir)
                base_path = self.get_base_path()
                container_path = self.get_container_path(base_path)
                container_name = container_path.rstrip("/").split("/")[-1]
                ctx = TestContext(
                    handler=handler,
                    base_path=base_path,
                    container_path=container_path,
                    container_name=container_name,
                    cache_dir=cache_dir,
                )
                yield ctx
            finally:
                if ctx is not None:
                    ctx.cleanup_remote()
                if handler is not None:
                    handler._close()

    def test_parse_uri_when_valid(self):
        valid_uris = {
            "az://account/container/path": ("account", "container", "path"),
            "az://account/container/dir/path": ("account", "container", "dir/path"),
            "az://account/container/dir/path/": ("account", "container", "dir/path/"),
            "blob://account/container/path": ("account", "container", "path"),
            "blob://account/container/dir/path": ("account", "container", "dir/path"),
            "blob://account/container/dir/path/": ("account", "container", "dir/path/"),
        }

        with self.context() as ctx:
            for uri, expected in valid_uris.items():
                actual = ctx.handler._parse_uri(uri)
                self.assertEqual(actual, expected)

    def test_parse_uri_invalid_protocol(self):
        with self.context() as ctx:
            try:
                uri = "s3://bucket/obj"
                ctx.handler._parse_uri(uri)
                self.fail("Invalid protocol was accepted: " + uri)
            except Exception as e:
                if not isinstance(e, ValueError):
                    raise e

    def test_status_operations(self):
        with self.context() as ctx:
            handler = ctx.handler

            # Set up test files
            filenames = ["file1.bin", "file2.bin"]
            remote_dir = os.path.join(ctx.base_path, "testdir")
            remote_paths = [os.path.join(remote_dir, f) for f in filenames]
            for i, filename in enumerate(filenames):
                local_path = ctx.get_local_test_file(filename)
                handler._copy_from_local(local_path, remote_paths[i])

            # Verify _exists for directory
            path = os.path.join(ctx.base_path, "testdir/")
            self.assertTrue(handler._exists(path))
            path = os.path.join(ctx.base_path, "testdir")
            self.assertTrue(handler._exists(path))
            path = os.path.join(ctx.base_path, "dne/")
            self.assertFalse(handler._exists(path))

            # Verify _exists for file
            self.assertTrue(handler._exists(remote_paths[0]))
            self.assertFalse(handler._exists(os.path.join(ctx.base_path, "dne")))

            # Verify _ls
            actual = set(handler._ls(os.path.join(ctx.base_path, "testdir/")))
            expected = {p.replace(ctx.container_path, "") for p in remote_paths}
            self.assertEqual(actual, expected, f"Failed to _ls() at {path}")

    def test_get_local_path(self):
        with self.context() as ctx:
            handler = ctx.handler

            # Set up test files
            filenames = ["file1.bin", "file2.bin"]
            remote_dir = os.path.join(ctx.base_path, "testdir")
            local_paths = [ctx.get_local_test_file(f) for f in filenames]
            remote_paths = [os.path.join(remote_dir, f) for f in filenames]
            for i in range(len(filenames)):
                handler._copy_from_local(local_paths[i], remote_paths[i])

            # Verify _get_local_path for directory
            download_dir = handler._get_local_path(remote_dir)
            for filename in filenames:
                expected_file = os.path.join(download_dir, filename)
                self.assertTrue(
                    os.path.exists(expected_file),
                    f"Expected file to exist: {expected_file}",
                )

            # Verify _get_local_path for file
            download_path = handler._get_local_path(remote_paths[0])
            self.assertEqual(
                self.get_md5(local_paths[0]),
                self.get_md5(download_path),
                "Cached file MD5 did not match",
            )

    def test_copy(self):
        with self.context() as ctx:
            handler = ctx.handler

            # Set up test files
            filenames = ["file1.bin"]
            remote_dir = os.path.join(ctx.base_path, "testdir")
            local_paths = [ctx.get_local_test_file(f) for f in filenames]
            remote_paths = [os.path.join(remote_dir, f) for f in filenames]
            for i in range(len(filenames)):
                handler._copy_from_local(local_paths[i], remote_paths[i])

            # Verify _copy
            src_path = remote_paths[0]
            copy_path = src_path + ".bak"
            success = handler._copy(src_path, copy_path)
            self.assertTrue(
                success, f"Failed to _copy() from {src_path} to {copy_path}"
            )
            self.assertTrue(
                handler._exists(copy_path),
                f"Failed to _copy() from {src_path} to {copy_path}",
            )

    def test_open(self):
        with self.context() as ctx:
            handler = ctx.handler

            # Set up test files
            filenames = ["file1.bin"]
            remote_dir = os.path.join(ctx.base_path, "testdir")
            local_paths = [ctx.get_local_test_file(f) for f in filenames]
            remote_paths = [os.path.join(remote_dir, f) for f in filenames]
            for i in range(len(filenames)):
                handler._copy_from_local(local_paths[i], remote_paths[i])

            # Verify _open(mode="rb")
            expected = self.get_md5(local_paths[0])
            with handler._open(remote_paths[0], "rb") as f:
                actual = self.get_md5_from_file(f)
            self.assertEqual(expected, actual, "Streamed file MD5 did not match")

            # Verify _open(mode="wb")
            chunk_size = 2 * 1024 * 1024
            new_remote_path = remote_paths[0] + ".streamed"
            with open(local_paths[0], "rb") as f_local, handler._open(
                new_remote_path, "wb"
            ) as f:
                data = f_local.read(chunk_size)
                while data:
                    f.write(data)
                    data = f_local.read(chunk_size)


if __name__ == "__main__":
    unittest.main()
