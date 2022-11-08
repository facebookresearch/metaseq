# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from datetime import datetime
import base64
import io
import logging
import os
import shutil
import time
from typing import IO, Any, Dict, Generator, List, Optional, Tuple, Union

from metaseq.file_io.common import file_lock, get_cache_dir, PathHandler


try:
    import azure.core.exceptions as azure_exceptions
    import azure.storage.blob as azure_blob

    # Reduce noise by suppresssing HTTP logging in the client by default
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
        logging.WARNING
    )
except ImportError:
    azure_exceptions = None
    azure_blob = None


logger = logging.getLogger(__name__)


DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024


class AzureBlobTokenProvider:
    @abstractmethod
    def get_sas_token(self, account: str) -> str:
        """
        Returns a SAS token for the specified storage account
        """


class AzureBlobReader(io.RawIOBase):
    """
    Implements a readonly file-like interface around the Azure BlobClient.
    BlobClient provides an iterator (StorageStreamDownloader.chunks())
    that lazily downloads parts of the blob into memory one fixed size
    chunk at a time. This class handles reads by slicing parts of
    the current chunk until it's fully consumed, and advancing
    the iterator as needed until EOF.
    """

    def __init__(
        self,
        client,  # azure_blob.BlobClient
        chunk_size: int,
    ):
        stream = client.download_blob()
        self._client = client
        self._chunk_iter = stream.chunks()
        self._chunk_size = chunk_size
        self._chunk = None
        self._chunk_pos = 0

    def _next_chunk(self):
        blob_name = self._client.blob_name
        self._chunk = next(self._chunk_iter)
        logger.info(
            "Read next chunk: blob_name=%s, length=%d", blob_name, len(self._chunk)
        )
        self._chunk_pos = 0

    def _get_chunk_data(self, size: int) -> bytes:
        """
        Return up to `size` bytes from the currently loaded chunk. If there
        isn't enough data remaining, just return what's available.
        If this is the first read, or the current chunk has been read entirely,
        make a single network request to download the next `chunk_size` bytes
        and return data as described above.
        """
        if size == 0:
            return bytes()

        try:
            if self._chunk is None or self._chunk_pos >= len(self._chunk):
                self._next_chunk()

            data = self._chunk[self._chunk_pos : self._chunk_pos + size]
            self._chunk_pos += len(data)
            return data
        except StopIteration:
            return bytes()

    def seekable(self):
        return False

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        raise io.UnsupportedOperation()

    def tell(self):
        raise io.UnsupportedOperation()

    def readable(self):
        return True

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            return self.readall()
        return self._get_chunk_data(size)

    def readall(self) -> bytes:
        stream = io.BytesIO()
        self.readinto(stream)
        return stream.getvalue()

    def readinto(self, stream):
        size = self._chunk_size
        data = self._get_chunk_data(size)
        # Getting zero bytes back == EOF
        while len(data) > 0:
            stream.write(data)
            data = self._get_chunk_data(size)

    def writeable(self):
        return False

    def write(self, b: bytes):
        raise io.UnsupportedOperation()

    def truncate(self, size):
        raise io.UnsupportedOperation()

    def close(self):
        pass


class AzureBlobWriter(io.RawIOBase):
    """
    Provides a write-only file-like interface around the Azure BlobClient.
    This class keeps a fixed size in-memory write buffer, uploading it
    to Azure when it fills up or when flush() is called. Each flush()
    sends a PutBlock request to Blob Storage including the buffered data
    and a unique block id. The blob is finalized on close() with
    a PutBlockList request specifying the block ids to commit.
    """

    def __init__(
        self,
        client,  # azure_blob.BlobClient,
        chunk_size: int,
    ):
        self._client = client
        self._chunk_size = chunk_size
        self._chunk_idx = -1
        self._chunk = None
        self._blocks = []

    def _new_block_id(self):
        self._chunk_idx += 1
        # BlockBlobs and AppendBlobs have a maximum of 50_000 blocks
        # This uses block indexes (00000 ... 49999) as block ids
        # From the docs: block ids must be base64 strings, < 64 bytes in size
        # and the same length for all blocks in the same blob
        block_id = "{0:05}".format(self._chunk_idx).encode("utf-8")
        block_id = base64.urlsafe_b64encode(block_id)
        return block_id.decode("utf-8")

    def _next_chunk(self):
        self.flush()
        self._chunk = io.BytesIO()

    def _append_to_chunk(self, b: bytes):
        if self._chunk is None or self._chunk.tell() >= self._chunk_size:
            self._next_chunk()
        return self._chunk.write(b)

    def seekable(self):
        return False

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        raise io.UnsupportedOperation()

    def tell(self):
        raise io.UnsupportedOperation()

    def readable(self):
        return False

    def read(self, size: int = -1) -> bytes:
        raise io.UnsupportedOperation()

    def readall(self) -> bytes:
        raise io.UnsupportedOperation()

    def readinto(self, stream):
        raise io.UnsupportedOperation()

    def writeable(self):
        return True

    def write(self, b: bytes):
        return self._append_to_chunk(b)

    def truncate(self, size):
        raise io.UnsupportedOperation()

    def flush(self):
        if self._chunk is None or self._chunk.tell() == 0:
            return

        block_id = self._new_block_id()
        block_length = self._chunk.tell()
        self._chunk.seek(0)

        logger.info(
            "Uploading a new block: blob_name=%s, block_id=%s, idx=%d, length=%d",
            self._client.blob_name,
            block_id,
            self._chunk_idx,
            block_length,
        )
        self._client.stage_block(block_id, self._chunk, block_length)
        self._blocks.append(azure_blob.BlobBlock(block_id=block_id))

    def close(self):
        self.flush()
        if self._blocks:
            logger.info(
                "Committing blocks: blob_name=%s, count=%d",
                self._client.blob_name,
                len(self._blocks),
            )
            self._client.commit_block_list(self._blocks)

    def __exit__(self, _, __, ___):
        # Make sure close() is called to commit everything
        self.close()


ENV_SAS_TOKEN = "AZURE_STORAGE_SAS_TOKEN"


class EnvironmentTokenProvider(AzureBlobTokenProvider):
    def get_sas_token(self, _: str) -> str:
        assert (
            ENV_SAS_TOKEN in os.environ
        ), f"Missing required env variable: {ENV_SAS_TOKEN}"
        return os.environ[ENV_SAS_TOKEN]


class AzureBlobPathHandler(PathHandler):
    """
    Support for Microsoft Azure Blob Storage
    """

    SUPPORTED_PREFIXES = ["az://", "blob://"]
    CACHE_SUBDIR_NAME = "blob_cache"

    def __init__(
        self,
        token_provider: Optional[AzureBlobTokenProvider] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            token_provider (AzureBlobTokenProvider): provider used to generate SAS tokens
                for authenticating with Blob Storage.
            cache_dir (str): Local filesystem directory to use for caching. If None,
                uses default from `file_io.get_cache_dir()`.
        """
        self.token_provider = token_provider or EnvironmentTokenProvider()
        self.cache_dir = cache_dir

    def _get_supported_prefixes(self) -> List[str]:
        return self.SUPPORTED_PREFIXES

    def _parse_uri(self, uri: str) -> Tuple[str, str]:
        """
        Parses a "blob://<account>/<container>/<path>" URI into components
            (`account`, `container`, `path`)
        Args:
            uri (str): A blob:// URI.
        Returns:
            account (str): the storage account.
            container (str): the storage container.
            path (str): the blob path.
        """
        for prefix in self.SUPPORTED_PREFIXES:
            if not uri.startswith(prefix):
                continue
            splits = uri.replace(prefix, "").split("/")
            account = splits[0]
            container = splits[1]
            path = "/".join(splits[2:])
            return account, container, path

        raise ValueError(f"Unsupported URI: {uri}")

    def _get_service_uri(
        self, account: str, container: str, blob_path: str, include_auth: bool = False
    ) -> str:
        account_uri = f"https://{account}.blob.core.windows.net"
        uri = os.path.join(account_uri, container, blob_path)

        if include_auth:
            sas_token = self.token_provider.get_sas_token(account)
            uri += "?" + sas_token

        return uri

    def _get_client(self, account: str):
        if not hasattr(self, "client"):
            account_uri = f"https://{account}.blob.core.windows.net"
            sas_token = self.token_provider.get_sas_token(account)
            client = azure_blob.BlobServiceClient(account_uri, credential=sas_token)
            self.client = client

        return self.client

    def _get_blob_properties(self, path: str) -> Optional[Dict]:
        account, container, blob = self._parse_uri(path)
        client = self._get_client(account)
        props = client.get_blob_client(
            container=container, blob=blob
        ).get_blob_properties()

        return {k: v for k, v in props.items()}

    def _enumerate_blobs(self, path: str) -> Generator[str, None, None]:
        account, container, path_prefix = self._parse_uri(path)
        client = self._get_client(account)
        return client.get_container_client(container).list_blobs(
            name_starts_with=path_prefix
        )

    def _exists(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if there is a resource at the given URI.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            bool: true if the path exists
        """
        self._check_kwargs(kwargs)
        return self._isfile(path) or self._isdir(path)

    def _isfile(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if the resource at the given URI is a file.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            bool: true if the path is a file
        """
        self._check_kwargs(kwargs)

        try:
            props = self._get_blob_properties(path)
            return props is not None
        except azure_exceptions.ResourceNotFoundError:
            return False
        except azure_exceptions.AzureError as e:
            logger.exception(e)
            return False

    def _isdir(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if the resource at the given URI is a directory.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            bool: true if the path is a directory
        """
        self._check_kwargs(kwargs)

        _, _, dirpath = self._parse_uri(path)

        try:
            # Enumeration should find at least 1 longer child path
            blob = next(self._enumerate_blobs(path))
            return len(blob.name) > len(dirpath)
        except StopIteration:
            return False
        except azure_exceptions.AzureError as e:
            logger.exception(e)
            return False

    def _ls(self, path: str, **kwargs: Any) -> List[str]:
        """
        List the contents of the directory at the provided URI.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            List[str]: list of contents in given path
        """
        self._check_kwargs(kwargs)

        return [blob.name for blob in self._enumerate_blobs(path)]

    def _local_cache_path(self, path: str):
        """
        Helper that returns a local cache path for a given uri.
        Args:
            path (str): A URI supported by this PathHandler.
        Returns:
            local_cache_path (str): a file path which exists on the local file system,
            in a cache directory.
        """
        _, _, blob = self._parse_uri(path)
        return os.path.join(get_cache_dir(self.cache_dir), self.CACHE_SUBDIR_NAME, blob)

    def _get_local_path(self, path: str, force: bool = False, **kwargs: Any) -> str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.
        If URI points to a remote resource, this function may download and cache
        the resource to local disk. In this case, the cache stays on filesystem
        (under `file_io.get_cache_dir()`) and will be used by a different run.
        Therefore this function is meant to be used with read-only resources.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            local_path (str): a file path which exists on the local file system
        """
        self._check_kwargs(kwargs)

        assert self._exists(path), f"File not found: {path}"
        account, container, _ = self._parse_uri(path)
        result_path = self._local_cache_path(path)

        # TODO: this can be parallelized for directories
        for blob in self._enumerate_blobs(path):
            blob_path = os.path.join("az://", account, container, blob.name)
            self._get_local_path_single(blob_path, force=force)

        return result_path

    def _get_local_path_single(self, path: str, force: bool = False) -> str:
        local_path = self._local_cache_path(path)
        with file_lock(local_path):
            if os.path.exists(local_path):
                # Redownload if remote is newer.
                props = self._get_blob_properties(path)
                remote_modified = props["last_modified"]
                local_modified = datetime.fromtimestamp(
                    os.path.getmtime(local_path)
                ).astimezone()

                if remote_modified <= local_modified and not force:
                    logger.info(
                        "URL {} was already cached in {}".format(path, local_path)
                    )
                    return local_path

            logger.info("Caching {} ...".format(path))
            tmp = local_path + ".tmp"
            # clean-up tmp if found, because if tmp exists, it must be a dirty
            # result of a previously process that didn't cleanup itself.
            if os.path.isfile(tmp):
                os.unlink(tmp)

            account, container, blob_path = self._parse_uri(path)
            client = self._get_client(account)
            try:
                blob_stream = client.get_blob_client(
                    container=container, blob=blob_path
                ).download_blob()

                with open(tmp, "wb") as f_tmp:
                    blob_stream.readinto(f_tmp)

                shutil.move(tmp, local_path)
            finally:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass

            logger.info("URL {} cached in {}".format(path, local_path))
            return local_path

    def _open(
        self,
        path: str,
        mode: str = "rb",
        buffering: int = -1,
        **kwargs: Any,
    ) -> Union[IO[str], IO[bytes]]:
        """
        Open a stream to a URI, similar to the built-in `open`.
        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): An optional integer used to set the buffering policy.
                Pass 0 to switch buffering off and an integer >= 1 to indicate the
                size in bytes of a fixed-size chunk buffer. When no buffering
                argument is given, the default buffering policy depends on the
                underlying I/O implementation.
        Returns:
            file: a file-like object.
        """
        self._check_kwargs(kwargs)
        assert mode in ("rb", "wb"), "Supported modes: rb, wb"

        chunk_size = buffering if buffering > 0 else DEFAULT_CHUNK_SIZE

        account, container, blob_path = self._parse_uri(path)
        blob_client = self._get_client(account).get_blob_client(
            container=container,
            blob=blob_path,
        )

        logger.info("Opening blob: path=%s, mode=%s", path, mode)

        if "r" in mode:
            return AzureBlobReader(blob_client, chunk_size)
        if "w" in mode:
            return AzureBlobWriter(blob_client, chunk_size)

        raise ValueError("Invalid mode: " + mode)

    def _mkdirs(self, path: str, **kwargs: Any) -> None:
        """
        No-op since blob storage has a flat structure and no explicit notion of directories.
        """
        pass

    def _copy_from_local(
        self, local_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> bool:
        """
        Copies a local file to the specified URI.
        If the URI is another local path, this should be functionally identical
        to copy.
        Args:
            local_path (str): a file path which exists on the local file system
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing URI
        Returns:
            status (bool): True on success
        """
        self._check_kwargs(kwargs)

        # Just checking this to avoid expensive API calls in self._isdir().
        if local_path.endswith("/") or dst_path.endswith("/"):
            raise NotImplementedError(
                "AzureBlobPathHandler does not currently support uploading directories"
            )

        account, container, blob_path = self._parse_uri(dst_path)
        blob_client = self._get_client(account).get_blob_client(
            container=container, blob=blob_path
        )

        with open(local_path, "rb") as src_file:
            src_length = os.fstat(src_file.fileno()).st_size
            try:
                blob_client.upload_blob(src_file, length=src_length)
                return True
            except azure_exceptions.AzureError as e:
                logger.error(f"Error in file upload - {str(e)}")
                return False

    def _copy(
        self, src_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> bool:
        """
        Copies a source path to a destination path.
        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file
        Returns:
            status (bool): True on success
        """
        self._check_kwargs(kwargs)

        src_account, src_container, src_blob_path = self._parse_uri(src_path)
        dst_account, dst_container, dst_blob_path = self._parse_uri(dst_path)
        src_uri = self._get_service_uri(
            src_account,
            src_container,
            src_blob_path,
            include_auth=True,
        )
        dst_blob = self._get_client(dst_account).get_blob_client(
            container=dst_container, blob=dst_blob_path
        )

        try:
            _ = dst_blob.start_copy_from_url(src_uri)
            return self._wait_for_copy(dst_blob)
        except azure_exceptions.AzureError as e:
            logger.exception(e)
            return False

    def _wait_for_copy(
        self,
        blob,  # azure_blob.BlobClient
        timeout_secs=1800,
        polling_secs=30,
    ) -> bool:
        props = blob.get_blob_properties()
        deadline = int(datetime.utcnow().timestamp()) + timeout_secs

        while (
            props.copy.status == "pending"
            and int(datetime.utcnow().timestamp()) < deadline
        ):
            time.sleep(polling_secs)
            props = blob.get_blob_properties()

        return props.copy.status != "pending"

    def _rm(self, path: str, **kwargs: Any) -> None:
        """
        Remove the file (not directory) at the provided URI.
        Args:
            path (str): A URI supported by this PathHandler
        """
        self._check_kwargs(kwargs)

        account, container, blob_path = self._parse_uri(path)
        blob_client = self._get_client(account).get_blob_client(
            container=container, blob=blob_path
        )

        try:
            blob_client.delete_blob()
        except azure_exceptions.AzureError as e:
            raise OSError(
                f"Error in rm path {path} - " f"{type(e).__name__}: {e}"
            ) from e

    def _close(self):
        """
        Closes any sockets the Azure client may have opened.
        """
        if hasattr(self, "client"):
            self.client.close()
