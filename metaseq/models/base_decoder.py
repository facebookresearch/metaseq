# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch.nn as nn
from torch import Tensor

from metaseq import utils
from metaseq.incremental_decoding_utils import with_incremental_state


@with_incremental_state
class BaseDecoder(nn.Module):
    """Base class for incremental decoders.

    Incremental decoding is a mode at inference time where the Model
    only receives a single timestep of input corresponding to the previous
    output token (for teacher forcing) and must produce the next output
    *incrementally*. Thus the model must cache any long-term state that is
    needed about the sequence, e.g., hidden states, convolutional states, etc.

    This interface also defines the :func:`reorder_incremental_state` method,
    which is used during beam search to select and reorder the incremental state
    based on the selection of beams.

    To learn more about how incremental decoding works, refer to `this blog
    <http://www.telesens.co/2019/04/21/understanding-incremental-decoding-in-fairseq/>`_.

    Note that incremental_state will take different values depending on the
    situation. At train and validation time, incremental_state will be None,
    indicating that no incremental state is available and does not need to be
    computed.

    During generation, incremental_state will begin as an empty
    dictionary, indicating no incremental_state is available, but SHOULD be
    computed. This class modifies this dictionary inline via
    reorder_incremental_state. After that first initial step, incremental_state
    will be full of model-specific state.
    """

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary

    def forward(self, prev_output_tokens, incremental_state=None, **kwargs):
        """
        Args:
            prev_output_tokens (LongTensor): shifted output tokens of shape
                `(batch, tgt_len)`, for teacher forcing
            incremental_state (dict, optional): dictionary used for storing
                state during :ref:`Incremental decoding`. Note that this
                dictionary is modified inline iff incremental_state is not None.

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def extract_features(self, prev_output_tokens, incremental_state=None, **kwargs):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def output_layer(self, features, **kwargs):
        """
        Project features to the default output size, e.g., vocabulary size.

        Args:
            features (Tensor): features returned by *extract_features*.
        """
        raise NotImplementedError

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder incremental state.

        This will be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        pass

    def get_normalized_probs(self, logits: Tensor, log_probs: bool):
        """Get normalized probabilities (or log probs) from a net's output."""
        if log_probs:
            return utils.log_softmax(logits, dim=-1)
        else:
            return utils.softmax(logits, dim=-1)

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        return 1e6  # an arbitrary large number
