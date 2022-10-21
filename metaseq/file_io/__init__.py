# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace  # noqa: F401
import logging

import torch

from omegaconf.dictconfig import DictConfig
from metaseq.file_io.common import g_pathmgr as PathManager


__all__ = [
    "PathManager",
]


logger = logging.getLogger(__file__)


try:
    from .s3 import S3PathHandler  # noqa: E402

    PathManager.register_handler(S3PathHandler())
except KeyError:
    pass
except Exception:
    logger.exception("Failed to register S3PathHandler. Try pip install boto3")


try:
    from .azure_blob import AzureBlobPathHandler  # noqa: E402

    PathManager.register_handler(AzureBlobPathHandler())
except ImportError:
    logger.exception(
        "Failed to register AzureBlobPathHandler. Try pip install azure-storage-blob"
    )
except Exception as e:
    logger.exception(e)


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
    # If model was trained with fp16, model from loaded state_dict can be moved to fp16
    if not isinstance(state, dict):
        return state
    if "cfg" in state:
        state["cfg"] = recursively_cast_dictconfigs(state["cfg"])

    return state


def load_and_pop_last_optimizer_state(pth):
    st = torch_load_cpu(pth)
    st.pop("last_optimizer_state", None)
    return st
