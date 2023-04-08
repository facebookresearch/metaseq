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
// https://github.com/NVIDIA/apex/blob/89cc215a49b0e99263a8184f17f17275879015aa/csrc/layer_norm_cuda.cpp

#include <torch/extension.h>
#include <vector>
#include <cassert>
#include "compat.h"

namespace {
void compute_n1_n2(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    int& n1,
    int& n2)
{
    int idiff = input.ndimension() - normalized_shape.size();
    n2 = 1;
    for (int i = 0;  i < (int)normalized_shape.size();  ++i) {
	    assert( input.sizes()[i+idiff] == normalized_shape[i] );
	    n2 *= normalized_shape[i];
    }
    n1 = 1;
    for (int i = 0;  i < idiff;  ++i) {
	    n1 *= input.sizes()[i];
    }
}

void check_args(
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    at::Tensor beta
    )
{
    TORCH_CHECK(!gamma.defined() || gamma.sizes().equals(normalized_shape));
    TORCH_CHECK(!beta.defined() || beta.sizes().equals(normalized_shape));
}

void check_args(
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma
    )
{
    TORCH_CHECK(!gamma.defined() || gamma.sizes().equals(normalized_shape));
}


void check_args(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    int& n1,
    int& n2
    )
{
    int64_t normalized_ndim = normalized_shape.size();

    if (normalized_ndim < 1) {
      std::stringstream ss;
      ss << "Expected normalized_shape to be at least 1-dimensional, i.e., "
         << "containing at least one element, but got normalized_shape="
         << normalized_shape;
      throw std::runtime_error(ss.str());
    }

    auto input_shape = input.sizes();
    auto input_ndim = input.dim();

    if (input_ndim < normalized_ndim ||
        !input_shape.slice(input_ndim - normalized_ndim).equals(normalized_shape)) {
      std::stringstream ss;
      ss << "Given normalized_shape=" << normalized_shape
         << ", expected input with shape [*";
      for (auto size : normalized_shape) {
        ss << ", " << size;
      }
      ss << "], but got input of size" << input_shape;
      throw std::runtime_error(ss.str());
    }

    compute_n1_n2(input,normalized_shape,n1,n2);
}

void check_args(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    at::Tensor beta,
    int& n1,
    int& n2
    )
{
    check_args(input,normalized_shape,n1,n2);
    check_args(normalized_shape,gamma,beta);
}

void check_args(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    int& n1,
    int& n2
    )
{
    check_args(input,normalized_shape,n1,n2);
    check_args(normalized_shape,gamma);
}
}

void cuda_layer_norm(
    at::Tensor* output,
    at::Tensor* mean,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor* gamma,
    at::Tensor* beta,
    double epsilon);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> layer_norm(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    double epsilon) {
  CHECK_INPUT(input);
  int n1,n2;
  check_args(input,normalized_shape,n1,n2);
  at::Tensor output = at::empty_like(input);
  at::Tensor mean = at::empty({n1}, input.options().dtype(input.scalar_type()==at::ScalarType::Half || input.scalar_type()==at::ScalarType::BFloat16 ? at::ScalarType::Float : input.scalar_type()));
  at::Tensor invvar = at::empty_like(mean);
  cuda_layer_norm(&output,&mean,&invvar,&input,n1,n2,
      normalized_shape,NULL,NULL,epsilon);
  return {output, mean, invvar};
}

std::vector<at::Tensor> layer_norm_affine(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    at::Tensor beta,
    double epsilon) {
  CHECK_INPUT(input);
  CHECK_INPUT(gamma);
  CHECK_INPUT(beta);
  int n1,n2;
  check_args(input,normalized_shape,gamma,beta,n1,n2);
  at::Tensor output = at::empty_like(input);
  const auto stats_dtype = (input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16) ? at::ScalarType::Float : input.scalar_type();
  at::Tensor mean = at::empty({n1}, input.options().dtype(stats_dtype));
  at::Tensor invvar = at::empty_like(mean);
  cuda_layer_norm(&output,&mean,&invvar,&input,n1,n2,
      normalized_shape,&gamma,&beta,epsilon);
  return {output, mean, invvar};
}

std::vector<at::Tensor> layer_norm_affine_mixed_dtypes(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    at::Tensor beta,
    double epsilon) {
  CHECK_INPUT(input);
  int n1, n2;
  check_args(input, normalized_shape, n1, n2);
  at::Tensor output = at::empty_like(input, gamma.options().dtype(gamma.scalar_type()));
  at::Tensor mean = at::empty({n1}, input.options().dtype(input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16 ? at::ScalarType::Float : input.scalar_type()));
  at::Tensor invvar = at::empty_like(mean);
   cuda_layer_norm(&output, &mean, &invvar, &input, n1, n2,
      normalized_shape, &gamma, &beta, epsilon);
  return {output, mean, invvar};
}

void cuda_layer_norm_gradient(
    at::Tensor* dout,
    at::Tensor* mean,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor* gamma,
    at::Tensor* beta,
    double epsilon,
    at::Tensor* grad_input,
    at::Tensor* grad_gamma,
    at::Tensor* grad_beta
    );

at::Tensor layer_norm_gradient(
    at::Tensor dout,
    at::Tensor mean,
    at::Tensor invvar,
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    double epsilon) {
  CHECK_INPUT(dout);
  CHECK_INPUT(mean);
  CHECK_INPUT(invvar);
  CHECK_INPUT(input);
  int n1,n2;
  check_args(input,normalized_shape,n1,n2);
  at::Tensor grad_input = at::empty_like(input);
  cuda_layer_norm_gradient(&dout,&mean,&invvar,&input,n1,n2,
      normalized_shape,NULL,NULL,epsilon,
      &grad_input,NULL,NULL);
  return grad_input;
}

std::vector<at::Tensor> layer_norm_gradient_affine(
    at::Tensor dout,
    at::Tensor mean,
    at::Tensor invvar,
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    at::Tensor beta,
    double epsilon) {
  CHECK_INPUT(dout);
  CHECK_INPUT(mean);
  CHECK_INPUT(invvar);
  CHECK_INPUT(input);
  CHECK_INPUT(gamma);
  CHECK_INPUT(beta);
  int n1,n2;
  check_args(input,normalized_shape,gamma,beta,n1,n2);
  at::Tensor grad_input = at::empty_like(input);
  at::Tensor grad_gamma = at::empty_like(gamma);
  at::Tensor grad_beta = at::empty_like(beta);
  cuda_layer_norm_gradient(&dout,&mean,&invvar,&input,n1,n2,
      normalized_shape,&gamma,&beta,epsilon,
      &grad_input,&grad_gamma,&grad_beta);
  return {grad_input, grad_gamma, grad_beta};
}

void cuda_rms_norm(
    at::Tensor* output,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor* gamma,
    double epsilon);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> rms_norm(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    double epsilon) {
  CHECK_INPUT(input);
  int n1,n2;
  check_args(input,normalized_shape,n1,n2);
  at::Tensor output = at::empty_like(input);
  at::Tensor invvar = at::empty({n1}, input.options().dtype(input.scalar_type()==at::ScalarType::Half || input.scalar_type()==at::ScalarType::BFloat16 ? at::ScalarType::Float : input.scalar_type()));
  cuda_rms_norm(&output,&invvar,&input,n1,n2,
      normalized_shape,NULL,epsilon);
  return {output, invvar};
}

std::vector<at::Tensor> rms_norm_affine(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    double epsilon) {
  CHECK_INPUT(input);
  CHECK_INPUT(gamma);
  int n1,n2;
  check_args(input,normalized_shape,gamma,n1,n2);
  at::Tensor output = at::empty_like(input);
  const auto stats_dtype = (input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16) ? at::ScalarType::Float : input.scalar_type();
  at::Tensor invvar = at::empty({n1}, input.options().dtype(stats_dtype));
  cuda_rms_norm(&output,&invvar,&input,n1,n2,
      normalized_shape,&gamma,epsilon);
  return {output, invvar};
}

std::vector<at::Tensor> rms_norm_affine_mixed_dtypes(
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    double epsilon) {
  CHECK_INPUT(input);
  int n1, n2;
  check_args(input, normalized_shape, n1, n2);
  at::Tensor output = at::empty_like(input, gamma.options().dtype(gamma.scalar_type()));
  at::Tensor invvar = at::empty({n1}, input.options().dtype(input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16 ? at::ScalarType::Float : input.scalar_type()));

   cuda_rms_norm(&output,&invvar, &input, n1, n2,
      normalized_shape, &gamma,epsilon);
  return {output,invvar};
}

void cuda_rms_norm_gradient(
    at::Tensor* dout,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor* gamma,
    double epsilon,
    at::Tensor* grad_input,
    at::Tensor* grad_gamma);

at::Tensor rms_norm_gradient(
    at::Tensor dout,
    at::Tensor invvar,
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    double epsilon) {
  CHECK_INPUT(dout);
  CHECK_INPUT(invvar);
  CHECK_INPUT(input);
  int n1,n2;
  check_args(input,normalized_shape,n1,n2);
  at::Tensor grad_input = at::empty_like(input);
  cuda_rms_norm_gradient(&dout,&invvar,&input,n1,n2,
      normalized_shape,NULL,epsilon,
      &grad_input,NULL);
  return grad_input;
}

std::vector<at::Tensor> rms_norm_gradient_affine(
    at::Tensor dout,
    at::Tensor invvar,
    at::Tensor input,
    #ifdef VERSION_GE_1_1
    at::IntArrayRef normalized_shape,
    #else
    at::IntList normalized_shape,
    #endif
    at::Tensor gamma,
    double epsilon) {
  CHECK_INPUT(dout);
  CHECK_INPUT(invvar);
  CHECK_INPUT(input);
  CHECK_INPUT(gamma);
  int n1,n2;
  check_args(input,normalized_shape,gamma,n1,n2);
  at::Tensor grad_input = at::empty_like(input);
  at::Tensor grad_gamma = at::empty_like(gamma);
  cuda_rms_norm_gradient(&dout,&invvar,&input,n1,n2,
      normalized_shape,&gamma,epsilon,
      &grad_input,&grad_gamma);
  return {grad_input, grad_gamma};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_affine", &layer_norm_affine, "LayerNorm forward (CUDA)");
  m.def("forward", &layer_norm, "LayerNorm forward (CUDA)");
  m.def("backward_affine", &layer_norm_gradient_affine, "LayerNorm backward (CUDA)");
  m.def("backward", &layer_norm_gradient, "LayerNorm backward (CUDA)");

  m.def("forward_affine_mixed_dtypes", &layer_norm_affine_mixed_dtypes, "LayerNorm forward with mixed dtypes (CUDA) compatible with Megatron's implementation");

  m.def("rms_forward_affine", &rms_norm_affine, "RMSNorm forward (CUDA)");
  m.def("rms_forward", &rms_norm, "RMSNorm forward (CUDA)");
  m.def("rms_backward_affine", &rms_norm_gradient_affine, "RMSNorm backward (CUDA)");
  m.def("rms_backward", &rms_norm_gradient, "RMSNorm backward (CUDA)");

  m.def("rms_forward_affine_mixed_dtypes", &rms_norm_affine_mixed_dtypes, "RMSNorm forward with mixed dtypes (CUDA) compatible with Megatron's implementation");
}