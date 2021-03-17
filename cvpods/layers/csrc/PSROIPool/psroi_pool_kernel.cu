#include <torch/types.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <stdio.h>
#include <math.h>
#include <float.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__global__ void PSROIPoolForward(
    const T* bottom_data,
    const T spatial_scale,
    const int num_rois,
    const int height,
    const int width,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    const T* bottom_rois,
    const int group_size,
    const int output_dim,
    T* top_data,
    int* mapping_channel,
    cudaStream_t stream)
{
    const long output_size = output_dim * pooled_height * pooled_width * num_rois;
    const long nthreads = output_size;

    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int ctop = (index / pooled_width / pooled_height) % output_dim;
        int n = index / pooled_width / pooled_height / output_dim;

        bottom_rois += n * 5;
        int roi_batch_ind = bottom_rois[0];
        T roi_start_w = bottom_rois[1] * spatial_scale;
        T roi_start_h = bottom_rois[2] * spatial_scale;
        T roi_end_w = (bottom_rois[3] + 1) * spatial_scale;
        T roi_end_h = (bottom_rois[4] + 1) * spatial_scale;
        
        T roi_width = roi_end_w - roi_start_w;
        T roi_height = roi_end_h - roi_start_h;

        // skip invalid rois
        if(roi_width <= 0 || roi_height <= 0)
        {
            continue;
        }

        T bin_size_h = roi_height / static_cast<T>(pooled_height);
        T bin_size_w = roi_width / static_cast<T>(pooled_width);

        int hstart = floor(static_cast<T>(ph) * bin_size_h + roi_start_h);
        int wstart = floor(static_cast<T>(pw) * bin_size_w + roi_start_w);
        int hend = ceil(static_cast<T>(ph + 1) * bin_size_h + roi_start_h);
        int wend = ceil(static_cast<T>(pw + 1) * bin_size_w + roi_start_w);

        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart, 0), height);
        hend = min(max(hend, 0), height);
        wstart = min(max(wstart, 0), width);
        wend = min(max(wend, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        int gw = pw;
        int gh = ph;
        int c = (ctop * group_size + gh) * group_size + gw;

        bottom_data += (roi_batch_ind * channels + c) * height * width;
        T out_sum = 0;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            int bottom_index = h * width + w;
            out_sum += bottom_data[bottom_index];
          }
        }
        float bin_area = (hend - hstart) * (wend - wstart);
        //top_data[index] = nthreads;
        top_data[index] = is_empty ? 0. : out_sum / bin_area;
        mapping_channel[index] = c;
    }
}

template <typename T>
__global__ void PSROIPoolBackward(const T* top_diff,
    const int* mapping_channel,
    const int batch_size,
    const int num_rois,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_width,
    const int pooled_height,
    const int output_dim,
    T* bottom_diff,
    const T* bottom_rois,
    cudaStream_t stream)
{
    const long output_size = output_dim * pooled_height * pooled_width * num_rois;
    const long nthreads = output_size;

    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int n = index / pooled_width / pooled_height / output_dim;

        // [start, end) interval for spatial sampling
        bottom_rois += n * 5;
        int roi_batch_ind = bottom_rois[0];
        T roi_start_w = bottom_rois[1] * spatial_scale;
        T roi_start_h = bottom_rois[2] * spatial_scale;
        T roi_end_w = (bottom_rois[3] + 1) * spatial_scale;
        T roi_end_h = (bottom_rois[4] + 1) * spatial_scale;
        
        T roi_width = roi_end_w - roi_start_w;
        T roi_height = roi_end_h - roi_start_h;

        // skip invalid rois
        if(roi_width <= 0 || roi_height <= 0)
        {
            continue;
        }

        // Compute w and h at bottom
        T bin_size_h = roi_height / static_cast<T>(pooled_height);
        T bin_size_w = roi_width / static_cast<T>(pooled_width);

        int hstart = floor(static_cast<T>(ph) * bin_size_h + roi_start_h);
        int wstart = floor(static_cast<T>(pw) * bin_size_w + roi_start_w);
        int hend = ceil(static_cast<T>(ph + 1) * bin_size_h + roi_start_h);
        int wend = ceil(static_cast<T>(pw + 1) * bin_size_w + roi_start_w);
        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart, 0), height);
        hend = min(max(hend, 0), height);
        wstart = min(max(wstart, 0), width);
        wend = min(max(wend, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Compute c at bottom
        int c = mapping_channel[index];
        T* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
        float bin_area = (hend - hstart) * (wend - wstart);
        T diff_val = is_empty ? 0. : top_diff[index] / bin_area;
        for (int h = hstart; h < hend; ++h)
        {
            for (int w = wstart; w < wend; ++w)
            {
                int bottom_index = h * width + w;
                //caffe_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
                atomicAdd(offset_bottom_diff + bottom_index, diff_val);
            }
        }
    }
}

namespace cvpods{

at::Tensor psroi_pooling_forward_cuda(
    at::Tensor& features,
    at::Tensor& rois,
    at::Tensor& mapping_channel,
    const int pooled_height,
    const int pooled_width,
    const float spatial_scale,
    const int group_size,
    const int output_dim)
{
    int* mapping_channel_out = mapping_channel.contiguous().data_ptr<int>();
    //Get # of Rois
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    AT_ASSERTM(size_rois == 5, "rois channels must be 5");

    at::Tensor output = at::zeros({num_rois, output_dim, pooled_height, pooled_width}, features.options());

    int data_height = features.size(2);
    int data_width = features.size(3);
    int num_channels = features.size(1);

    const int kThreadsPerBlock = 1024;
    const long output_size = (long)num_rois * pooled_height * pooled_width * num_channels;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(features.scalar_type(), "PSROIPoolForward", [&] {
        scalar_t* data_in = features.contiguous().data_ptr<scalar_t>();
        scalar_t* rois_in = rois.contiguous().data_ptr<scalar_t>();
        scalar_t* output_out = output.contiguous().data_ptr<scalar_t>();

        // call the gpu kernel for psroi_pooling
        PSROIPoolForward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
            data_in, spatial_scale, num_rois,
            data_height, data_width, num_channels,
            pooled_height, pooled_width, rois_in,
            group_size, output_dim,
            output_out, mapping_channel_out, stream
        );
    });

    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        printf("error in psroi_pooling_forward_cuda: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    return output;
}

at::Tensor psroi_pooling_backward_cuda(
    at::Tensor& top_grad,
    at::Tensor& rois,
    at::Tensor& mapping_channel,
    const int batch_size,
    const int bottom_dim,
    const int bottom_height,
    const int bottom_width,
    const float spatial_scale)
{
    int output_dim = top_grad.size(1);
    int pooled_height = top_grad.size(2);
    int pooled_width = top_grad.size(3);
    at::Tensor bottom_grad = at::zeros({batch_size, bottom_dim, bottom_height, bottom_width}, top_grad.options());

    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    AT_ASSERTM(size_rois == 5, "rois channels must be 5");

    int* mapping_channel_flat = mapping_channel.contiguous().data_ptr<int>();

    const int kThreadsPerBlock = 1024;
    const long output_size = (long)output_dim * pooled_height * pooled_width * num_rois;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(top_grad.scalar_type(), "PSROIPoolBackward", [&] {
        scalar_t* top_grad_flat = top_grad.contiguous().data_ptr<scalar_t>();
        scalar_t* rois_flat = rois.contiguous().data_ptr<scalar_t>();
        scalar_t* bottom_grad_flat = bottom_grad.contiguous().data_ptr<scalar_t>();

        // call the gpu kernel for psroi_pooling
        PSROIPoolBackward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
            top_grad_flat, mapping_channel_flat,
            batch_size, num_rois, spatial_scale, bottom_dim,
            bottom_height, bottom_width, pooled_width,
            pooled_height, output_dim,
            bottom_grad_flat, rois_flat, stream);
    });

    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        printf("error in psroi_pooling_backward_cuda: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    return bottom_grad;
}

}
