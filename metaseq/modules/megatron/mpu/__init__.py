# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .cross_entropy import vocab_parallel_cross_entropy

from .initialize import destroy_model_parallel
from .initialize import get_data_parallel_group
from .initialize import get_data_parallel_rank
from .initialize import get_tensor_model_parallel_group
from .initialize import get_tensor_model_parallel_rank
from .initialize import get_tensor_model_parallel_world_size
from .initialize import initialize_model_parallel

from .layers import LinearWithGradAccumulationAndAsyncCommunication
from .layers import ColumnParallelLinear
from .layers import RowParallelLinear
from .layers import VocabParallelEmbedding

from .mappings import copy_to_tensor_model_parallel_region
from .mappings import reduce_from_tensor_model_parallel_region
from .mappings import scatter_to_tensor_model_parallel_region
from .mappings import gather_from_tensor_model_parallel_region
from .mappings import scatter_to_sequence_parallel_region
from .mappings import reduce_scatter_to_sequence_parallel_region
from .mappings import _reduce_scatter_along_first_dim
from .mappings import _gather_along_first_dim

from .random import get_cuda_rng_tracker
from .random import model_parallel_cuda_manual_seed
from .random import gather_split_1d_tensor
from .random import split_tensor_into_1d_equal_chunks

from .utils import divide
from .utils import split_tensor_along_last_dim
from .utils import ensure_divisibility
from .utils import VocabUtility
