# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .global_vars import get_args
from .global_vars import get_global_memory_buffer

from .model.fused_softmax import ScaledUpperTriangMaskedSoftmax
from .model.fused_softmax import ScaledMaskedSoftmax

from .fused_kernels import scaled_upper_triang_masked_softmax_cuda
