import ast
import collections
import logging
import os
import re
import socket
from typing import Any, Dict, List, Optional, Tuple
import math
import torch
from omegaconf import OmegaConf

from metaseq.dataclass.configs import CheckpointConfig
from metaseq.dataclass.utils import overwrite_args_by_name, overwrite_keys_not_present
from metaseq.distributed import utils as distributed_utils
from metaseq.file_io import PathManager, torch_load_cpu
from metaseq.launcher.opt_job_constants import ComputeEnvs
import metaseq.checkpoint_utils as checkpoint_utils



logger = logging.getLogger(__name__)


OPT_KEY = "last_optimizer_state"

def load_pretrained(
        model,
        filename,
        reset_optimizer=True,
        reset_lr_scheduler=True,
        optimizer_overrides=None,
        reset_meters=True,
    ):
        """
        Load all training state from a checkpoint file.
        rank = 0 will load the checkpoint, and then broadcast it to all
        other ranks.
        """
        extra_state, _optim_history, last_optim_state = None, [], None

        is_distributed = True
        bexists = False

        logger.info(f"attempting to load checkpoint from: {filename}")

        if PathManager.isfile(filename):
            bexists = True
        else:
            # this is a big hacky as when we increase the world size, then filename doesn't really point
            # to a real file, we convert it to multiple files to be loaded later.
            # so here we just check if there are some files existing in the dir.
            files_in_local_dir = os.listdir(os.path.dirname(filename))
            filename_prefix = os.path.splitext(os.path.basename(filename))[0].replace(
                "model-part-0", ""
            )
            matched_files = [
                f for f in files_in_local_dir if f.startswith(filename_prefix)
            ]
            bexists = len(matched_files) > 0

        if bexists:
            logger.info(f"Preparing to load checkpoint {filename}")
            # FSDP requires loading checkpoint shards on all ranks
            load_on_all_ranks = True

            if load_on_all_ranks:
                state = load_checkpoint_to_cpu(
                    filename,
                )
                last_optim_state = state.get("last_optimizer_state", None)
                if last_optim_state == -1:
                    master_path = re.sub("shard[0-9]+", "shard0", filename)
                    last_optim_state = torch.load(master_path, map_location="cpu")[
                        "last_optimizer_state"
                    ]

                logger.info(f"Loaded state for {filename}")

            else:
                last_optim_state = None
                state = None

            
            # load model parameters
            try:
                model.load_state_dict(state["model"], strict=True)
                # save memory for later steps
                del state["model"]
               
            except Exception:
                raise Exception(
                    "Cannot load model parameters from checkpoint {}; "
                    "please ensure that the architectures match.".format(filename)
                )
            extra_state = state["extra_state"]
            
        
            logger.info(
                f"Loaded checkpoint {filename}"
            )
        else:
            logger.info("No existing checkpoint found {}".format(filename))

        return model
