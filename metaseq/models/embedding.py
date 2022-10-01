from typing import Optional

import torch
from torch import nn as nn


def Embedding(
    num_embeddings,
    embedding_dim,
    padding_idx,
    initialize_params_on_gpu=False,
    dtype: Optional[torch.dtype] = None,
):
    # Passing weights initialized on GPU.
    device = torch.cuda.current_device() if initialize_params_on_gpu else None
    if dtype is None:
        dtype = torch.float
    weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
    nn.init.normal_(weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(weight[padding_idx], 0)
    m = nn.Embedding(
        num_embeddings, embedding_dim, padding_idx=padding_idx, _weight=weight
    )
    return m