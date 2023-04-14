/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

// Reference:
// https://github.com/NVIDIA/apex/blob/89cc215a49b0e99263a8184f17f17275879015aa/apex/contrib/csrc/optimizers/fused_adam_cuda.cpp

#include <torch/extension.h>

// CUDA forward declaration
void fused_strided_check_finite(at::Tensor & overflow_flag, at::Tensor & p_copy, int stride, int clear_overflow_first);

void fused_adam_cuda(at::Tensor & p, at::Tensor & p_copy, at::Tensor & m, at::Tensor & v, at::Tensor & g, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay);
void fused_reversible_adam_cuda(at::Tensor & p, at::Tensor & p_copy, at::Tensor & m, at::Tensor & v, at::Tensor & g, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay);
void fused_maybe_adam_undo_cuda(at::Tensor & overflow_flag, at::Tensor & p, at::Tensor & m, at::Tensor & v, at::Tensor & g, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay);

void fused_adam_cuda_mt(int chunk_size, at::Tensor overflow_flag, std::vector<std::vector<at::Tensor>> tensor_lists, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay);

void maybe_cast_cuda(at::Tensor & overflow_flag, at::Tensor & p_in, at::Tensor & p_out);
void maybe_cast_cuda_mt(int chunk_size, at::Tensor overflow_flag, std::vector<std::vector<at::Tensor>> tensor_lists);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// C++ interface
void strided_check_finite(
		at::Tensor& overflow_flag,
		at::Tensor& p_copy,
		int stride,
		int clear_overflow_first
	 ) {
	CHECK_INPUT(p_copy);
	fused_strided_check_finite(overflow_flag, p_copy, stride, clear_overflow_first);
}
void adam(at::Tensor & p, at::Tensor & p_copy, at::Tensor & m, at::Tensor & v, at::Tensor & g, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay) {
        CHECK_INPUT(p);
        if (p_copy.numel() > 0) CHECK_INPUT(p_copy);
        CHECK_INPUT(m);
        CHECK_INPUT(v);
        CHECK_INPUT(g);
        int64_t num_elem = p.numel();
        TORCH_CHECK(m.numel() == num_elem, "number of elements in m and p tensors should be equal");
        TORCH_CHECK(v.numel() == num_elem, "number of elements in v and p tensors should be equal");
        TORCH_CHECK(g.numel() == num_elem, "number of elements in g and p tensors should be equal");
        TORCH_CHECK(p_copy.numel() == num_elem || p_copy.numel() == 0, "number of elements in p_copy and p tensors should be equal, or p_copy should be empty");

        fused_adam_cuda(p, p_copy, m, v, g, lr, beta1, beta2, eps, grad_scale, step, mode, bias_correction, decay);
}
void reversible_adam(at::Tensor & p, at::Tensor & p_copy, at::Tensor & m, at::Tensor & v, at::Tensor & g, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay) {
        CHECK_INPUT(p);
        if (p_copy.numel() > 0) CHECK_INPUT(p_copy);
        CHECK_INPUT(m);
        CHECK_INPUT(v);
        CHECK_INPUT(g);
        int64_t num_elem = p.numel();
        TORCH_CHECK(m.numel() == num_elem, "number of elements in m and p tensors should be equal");
        TORCH_CHECK(v.numel() == num_elem, "number of elements in v and p tensors should be equal");
        TORCH_CHECK(g.numel() == num_elem, "number of elements in g and p tensors should be equal");
        TORCH_CHECK(p_copy.numel() == num_elem || p_copy.numel() == 0, "number of elements in p_copy and p tensors should be equal, or p_copy should be empty");

        fused_reversible_adam_cuda(p, p_copy, m, v, g, lr, beta1, beta2, eps, grad_scale, step, mode, bias_correction, decay);
}
void maybe_adam_undo(at::Tensor & overflow_flag, at::Tensor & p, at::Tensor & m, at::Tensor & v, at::Tensor & g, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay) {
        CHECK_INPUT(p);
        CHECK_INPUT(m);
        CHECK_INPUT(v);
        CHECK_INPUT(g);
        int64_t num_elem = p.numel();
        TORCH_CHECK(m.numel() == num_elem, "number of elements in m and p tensors should be equal");
        TORCH_CHECK(v.numel() == num_elem, "number of elements in v and p tensors should be equal");
        TORCH_CHECK(g.numel() == num_elem, "number of elements in g and p tensors should be equal");

        fused_maybe_adam_undo_cuda(overflow_flag, p, m, v, g, lr, beta1, beta2, eps, grad_scale, step, mode, bias_correction, decay);
}
void maybe_cast(at::Tensor & overflow_flag, at::Tensor & p_in, at::Tensor & p_out) {
	CHECK_INPUT(p_in);
	CHECK_INPUT(p_out);
	int64_t num_elem = p_in.numel();
	TORCH_CHECK(p_out.numel() == num_elem, "number of elements in p_in and p_out should be equal");

	maybe_cast_cuda(overflow_flag, p_in, p_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("strided_check_finite", &strided_check_finite, "Strided finite check.");
        m.def("adam", &adam, "Adam optimized CUDA implementation.");
        m.def("reversible_adam", &reversible_adam, "Reversible Adam optimized CUDA implementation.");
        m.def("adam_mt", &fused_adam_cuda_mt, "Multi tensor Adam optimized CUDA implementation.");
        m.def("maybe_adam_undo", &maybe_adam_undo, "Undo function for Adam optimized CUDA implementation.");
        m.def("maybe_cast", &maybe_cast, "Unpack byte tensor containing e5m2 floats.");
        m.def("maybe_cast_mt", &maybe_cast_cuda_mt, "Unpack byte tensor containing e5m2 floats.");
}