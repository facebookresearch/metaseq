# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace  # noqa: F401
import logging
import collections.abc

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


torch.ops.load_library("dietgpu")

# finds compressible tensors
def recursively_dietgpu_find(obj, toplevel=True, tensors=None, locations=None, path=[]):
    if toplevel:
        # These are lists of tensors and their paths in the datastructure, keyed by dtype
        tensors = defaultdict(lambda: [])
        locations = defaultdict(lambda: [])
    if isinstance(obj, collections.abc.Mapping):
        for k,v in obj.iteritems():
            recursively_dietgpu_find(v, toplevel=False, tensors=tensors, locations = locations, path = path + [k])
    if isinstance(obj, list):
        for i,x in enumerate(obj):
            recursively_dietgpu_find(x, toplevel=False, tensors=tensors, locations = locations, path = path + [i])
    if isinstance(obj, torch.Tensor) and obj.dtype in [torch.float16, torch.bfloat16, torch.float32]:
        tensors[obj.dtype].append(obj)
        locations[obj.dtype].append(path)
    if toplevel:
        return tensors, locations


# First, uses recursively_dietgpu_find to find eligible tensors and their "paths" in the overall dict
# After compressing, uses the paths to mutate the given object to use the compressed tensors
# Includes the compressed paths to guide decompression
# TODO this doesn't compress int8 tensors, which would use False for the dietgpu float mode fwiw
def dietgpu_compress(obj):
    assert isinstance(obj, collections.abc.Mapping)
    tensors, locations = recursively_dietgpu_find(obj)
    smaller_tensors = {d: torch.ops.dietgpu.compress_data_simple(True, tensors) for d,tensors in tensors.iteritems()}
    # TODO this should probably be immutably zipping
    for dtype in tensors:
        for loc, tensor in zip(locations[dtype], tensors[dtype]):
            # let's find the parent container for the compressed tensor
            tensor_container = obj
            i = 0
            while i != len(loc) - 1:
                tensor_container = tensor_container[loc[i]]
                i += 1
            tensor_container[loc[-1]] = tensor
    obj["dietgpu_locs"] = locations


# If this contains dietgpu tensors, iterates through and grabs them all
# Then decompresses
# Finally, mutates the given object to have the decompressed tensors in place of the compressed ones
def dietgpu_decompress(obj):
    if "dietgpu_locs" not in obj:
        return obj
    locations = obj["dietgpu_locs"]
    for dtype in locations:
        compressed_tensors = []
        for loc in locations[dtype]:
            tensor_container = obj
            i = 0
            while i != len(loc) - 1:
                tensor_container = tensor_container[loc[i]]
                i += 1
            compressed_tensors.append(tensor_container[loc[-1]])
        tensors = torch.ops.dietgpu.decompress_data_simple(True, compressed_tensors)
        for loc, tensor in zip(locations[dtype], tensors):
            # let's find the parent container for the compressed tensor
            tensor_container = obj
            i = 0
            while i != len(loc) - 1:
                tensor_container = tensor_container[loc[i]]
                i += 1
            tensor_container[loc[-1]] = tensor
    del obj["dietgpu_locs"]


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
    dietgpu_decompress(state)
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
