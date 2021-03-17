#pragma once
#include <torch/types.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

namespace cvpods {

at::Tensor border_align_cuda_forward(
    const at::Tensor& feature,
    const at::Tensor& boxes,
    const at::Tensor& wh,
    const int pool_size);


at::Tensor border_align_cuda_backward(
    const at::Tensor& gradOutput,
    const at::Tensor& feature,
    const at::Tensor& boxes,
    const at::Tensor& wh,
    const int pool_size);


at::Tensor BorderAlign_Forward(
    const at::Tensor& feature,
    const at::Tensor& boxes,
    const at::Tensor& wh,
    const int pool_size) {
    return border_align_cuda_forward(feature, boxes, wh, pool_size);
}


at::Tensor BorderAlign_Backward(
    const at::Tensor& gradOutput,
    const at::Tensor& feature,
    const at::Tensor& boxes,
    const at::Tensor& wh,
    const int pool_size) {
    return border_align_cuda_backward(gradOutput, feature, boxes, wh, pool_size);
}

} // namespace cvpods