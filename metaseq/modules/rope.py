# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.from typing import Optional, Tuple

from torch import nn
import torch
from typing import Optional, Tuple


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        """
        The rotary embedding operation. Takes two tensors with shapes like
            (batch_size, n_heads, sequence_length, dim)
        and returns two tensors of the same shape.

        Args:
            dim: double the number of frequencies
            theta: base frequency
        """
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        self.register_buffer("freqs", freqs)
        self.freqs_cis: Optional[torch.Tensor] = None

    def _precompute_until(self, end: int):
        """
        Precomputation for fast mode.
        """
        t = torch.arange(end, device=self.freqs.device)  # type: ignore
        freqs = torch.outer(t, self.freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def get_freqs_cis(self, start: int, end: int) -> torch.Tensor:
        if self.freqs_cis is None or end > self.freqs_cis.shape[0]:
            self.freqs_cis = self._precompute_until(2 * end)
        return self.freqs_cis[start:end]  # type: ignore

    def forward(self, xq, xk, seq_dim: int, start: int):
        """
        Args:
            xq: first input tensor to apply the operation to
            xk: second input tensor to apply the operation to
            start: start index of the sequence. Making this larger than 0 lets you omit
                parts of the start of the data.
            seq_dim: must be -2
        Returns:
            Tuple of two tensors: result of the operation applied to the input tensors
        """
        seq_len = xq.shape[seq_dim]
        freqs_cis = self.get_freqs_cis(start, start + seq_len)
        return apply_rotary_emb(xq, xk, seq_dim, freqs_cis=freqs_cis, backward=False)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int):
    ndim = x.ndim
    assert 0 <= seq_dim < ndim
    assert freqs_cis.shape == (x.shape[seq_dim], x.shape[-1])
    shape = [d if i == seq_dim or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_dim: int,
    freqs_cis: torch.Tensor,
    backward: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    If backward=False:
        - inputs: (xq, xk)
        - outputs: (xq_out, xk_out)
    If backward=True:
        - inputs: (grad_xq_out, grad_xk_out)
        - outputs: (grad_xq, grad_xk)
    """
    if backward:
        freqs_cis = freqs_cis.conj()
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_, seq_dim)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)