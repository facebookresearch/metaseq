#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import shutil
from typing import List, Optional

import torch
from omegaconf.dictconfig import DictConfig

logger = logging.getLogger(__file__)

from .s3_utils import S3PathHandler  # noqa: E402

try:
    from iopath.common.file_io import PathManager

    IOPathPathManager = PathManager()
except ImportError:
    IOPathPathManager = None

try:
    IOPathPathManager.register_handler(S3PathHandler())
except KeyError:
    pass
except Exception:
    logging.exception("Failed to register S3 Path Handler. Try pip install boto3")

from torch.distributed._shard.checkpoint import (
    save_state_dict,
    load_state_dict,
    FileSystemReader,
    FileSystemWriter,
)


class PathManager:
    """
    Wrapper for insulating OSS I/O (using Python builtin operations) from
    fvcore's PathManager abstraction (for transparently handling various
    internal backends).
    """

    @staticmethod
    def open(
        path: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ):
        if IOPathPathManager:
            return IOPathPathManager.open(
                path=path,
                mode=mode,
                buffering=buffering,
                encoding=encoding,
                errors=errors,
                newline=newline,
            )
        return open(
            path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    @staticmethod
    def copy(src_path: str, dst_path: str, overwrite: bool = False) -> bool:
        if IOPathPathManager:
            return IOPathPathManager.copy(
                src_path=src_path, dst_path=dst_path, overwrite=overwrite
            )
        return shutil.copyfile(src_path, dst_path)

    @staticmethod
    def get_local_path(path: str, **kwargs) -> str:
        if IOPathPathManager:
            return IOPathPathManager.get_local_path(path, **kwargs)
        return path

    @staticmethod
    def exists(path: str) -> bool:
        if IOPathPathManager:
            return IOPathPathManager.exists(path)
        return os.path.exists(path)

    @staticmethod
    def isfile(path: str) -> bool:
        if IOPathPathManager:
            return IOPathPathManager.isfile(path)
        return os.path.isfile(path)

    @staticmethod
    def islink(path: str) -> Optional[bool]:
        if not PathManager.path_requires_pathmanager(path):
            return os.path.islink(path)
        return None

    @staticmethod
    def ls(path: str) -> List[str]:
        if IOPathPathManager:
            return IOPathPathManager.ls(path)
        return os.listdir(path)

    @staticmethod
    def mkdirs(path: str) -> None:
        if IOPathPathManager:
            return IOPathPathManager.mkdirs(path)
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def rm(path: str) -> None:
        if IOPathPathManager:
            return IOPathPathManager.rm(path)
        os.remove(path)
        assert not os.path.exists(path)

    @staticmethod
    def chmod(path: str, mode: int) -> None:
        if not PathManager.path_requires_pathmanager(path):
            os.chmod(path, mode)

    @staticmethod
    def register_handler(handler) -> None:
        if IOPathPathManager:
            return IOPathPathManager.register_handler(handler=handler)

    @staticmethod
    def copy_from_local(
        local_path: str, dst_path: str, overwrite: bool = False, **kwargs
    ) -> None:
        if IOPathPathManager:
            return IOPathPathManager.copy_from_local(
                local_path=local_path, dst_path=dst_path, overwrite=overwrite, **kwargs
            )
        return shutil.copyfile(local_path, dst_path)

    @staticmethod
    def path_requires_pathmanager(path: str) -> bool:
        """Do we require PathManager to access given path?"""
        if IOPathPathManager:
            for p in IOPathPathManager._path_handlers.keys():
                if path.startswith(p):
                    return True
        return False

    @staticmethod
    def supports_rename(path: str) -> bool:
        # PathManager doesn't yet support renames
        return not PathManager.path_requires_pathmanager(path)

    @staticmethod
    def rename(src: str, dst: str):
        os.rename(src, dst)

    """
    ioPath async PathManager methods:
    """

    @staticmethod
    def opena(
        path: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        callback_after_file_close=None,
    ):
        """
        Return file descriptor with asynchronous write operations.
        """
        global IOPathPathManager
        return IOPathPathManager.opena(
            path=path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            callback_after_file_close=callback_after_file_close,
        )

    @staticmethod
    def async_close() -> bool:
        """
        Wait for files to be written and clean up asynchronous PathManager.
        NOTE: `PathManager.async_close()` must be called at the end of any
        script that uses `PathManager.opena(...)`.
        """
        global IOPathPathManager
        if IOPathPathManager:
            return IOPathPathManager.async_close()
        return False


def recursively_cast_dictconfigs(cfg):
    if isinstance(cfg, DictConfig):
        cfg = eval(str(cfg))
    assert not isinstance(cfg, DictConfig)
    if isinstance(cfg, dict):
        return {k2: recursively_cast_dictconfigs(v2) for k2, v2 in cfg.items()}
    else:
        # Easy to support List, Tuple if needed
        return cfg


def torch_load_cpu(path):
    state = torch.load(path, map_location=torch.device("cpu"))
    # TODO (mingzhe): change to new API once ready.
    #state = {}
    #load_state_dict(
    #    state_dict=state,
    #    storage_reader=FileSystemReader(path)
    #)
    # If model was trained with fp16, model from loaded state_dict can be moved to fp16
    if not isinstance(state, dict):
        return state
    if "cfg" in state:
        state["cfg"] = recursively_cast_dictconfigs(state["cfg"])
        if (
            state["cfg"]["common"]["fp16"]
            or state["cfg"]["common"]["memory_efficient_fp16"]
        ):
            if os.environ.get("USE_PTD_FSDP", "False") == "True":
                tmp = torch.randn(1, dtype=torch.float16)
                state["model"] = {k: v.type_as(tmp) for k, v in state["model"].items()}
            else:
                state["model"] = {k: v.half() for k, v in state["model"].items()}

    return state


def save_json(content, path, indent=4):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent)


def load_json(p):
    return json.load(open(p))


def load_jsonl(path):
    with open(path).read() as jsonl_content:
        result = [json.loads(jline) for jline in jsonl_content.splitlines()]
    return result


def load_and_pop_last_optimizer_state(pth):
    st = torch_load_cpu(pth)
    st.pop("last_optimizer_state", None)
    return st
