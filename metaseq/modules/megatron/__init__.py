# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .mpu.cross_entropy import vocab_parallel_cross_entropy

from .mpu.initialize import get_tensor_model_parallel_world_size
from .mpu.initialize import initialize_model_parallel
from .mpu.initialize import get_tensor_model_parallel_group
from .mpu.initialize import get_data_parallel_group
from .mpu.initialize import destroy_model_parallel

from .mpu.layers import LinearWithGradAccumulationAndAsyncCommunication
from .mpu.layers import ColumnParallelLinear
from .mpu.layers import RowParallelLinear
from .mpu.layers import VocabParallelEmbedding

from .mpu.mappings import gather_from_tensor_model_parallel_region
from .mpu.mappings import _reduce_scatter_along_first_dim
from .mpu.mappings import _gather_along_first_dim

from .mpu.random import get_cuda_rng_tracker
from .mpu.random import model_parallel_cuda_manual_seed

from .mpu.utils import split_tensor_along_last_dim

from .model.fused_softmax import ScaledUpperTriangMaskedSoftmax
from .model.fused_softmax import ScaledMaskedSoftmax

from .fused_kernels import scaled_upper_triang_masked_softmax_cuda
