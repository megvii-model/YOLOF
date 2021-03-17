#include <torch/types.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <THC/THC.h>
#include <math.h>
#include <stdio.h>

namespace cvpods {
at::Tensor psroi_pooling_forward_cuda(
    at::Tensor& features,
    at::Tensor& rois,
    at::Tensor& mappingchannel,
    const int pooled_height,
    const int pooled_width,
    const float spatial_scale,
    const int group_size,
    const int output_dim);

at::Tensor psroi_pooling_backward_cuda(
    at::Tensor& top_grad,
    at::Tensor& rois,
    at::Tensor& mappingchannel,
    const int batch_size,
    const int bottom_dim,
    const int bottom_height,
    const int bottom_width,
    const float spatial_scale);
}
