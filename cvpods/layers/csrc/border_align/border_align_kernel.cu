#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>


#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__device__ T bilinear_interpolate(
    const T* bottom_data,
    const int height,
    const int width,
    T y,
    T x) {

    int y_low = (int) y;
    int x_low = (int) x;
    int y_high;
    int x_high;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (T) y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (T) x_low;            
    } else {
        x_high = x_low + 1;
    }

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly, hx = 1. - lx;
    // do bilinear interpolation
    T v1 = bottom_data[y_low * width + x_low];
    T v2 = bottom_data[y_low * width + x_high];
    T v3 = bottom_data[y_high * width + x_low];
    T v4 = bottom_data[y_high * width + x_high];
    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    return val;
}

template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height,
    const int width,
    T y, T x,
    T & w1, T & w2, T & w3, T & w4,
    int & x_low, int & x_high, int & y_low, int & y_high) {

    y_low = (int) y;
    x_low = (int) x;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (T) y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (T) x_low;
    } else {
        x_high = x_low + 1;
    }

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly, hx = 1. - lx;

    w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    return;
}


namespace cvpods {

template <typename T>
__global__ void BorderAlign_Forward(
    const int nthreads,
    T* feature,
    T* boxes,
    T* wh,
    const int channel,
    const int box_size,
    const int height,
    const int width,
    const int pool_size,
    T *output)
{
    T *feature_iter, *output_iter, *boxes_iter, *wh_iter;
    int batch_idx, box_idx, extreme_idx, fm_channel_idx;
    T stride, x_stride, y_stride;
    T x, y;
    T max, val;

    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
    extreme_idx = threadIdx.y;
    batch_idx = index / channel / box_size;
    box_idx = index % box_size + batch_idx * box_size;
    fm_channel_idx = (int)(index / box_size) % channel;

    boxes_iter = boxes + box_idx * 4 + extreme_idx / 2 * 2;
    wh_iter = wh + box_idx * 2;
    output_iter = output + index * 4 + extreme_idx;
    feature_iter = feature + (batch_idx * channel * 4 + extreme_idx * channel + fm_channel_idx) * height * width;

    x = *boxes_iter;
    y = *(boxes_iter + 1);

    switch(extreme_idx){
        case 0: stride=*wh_iter / pool_size;       x_stride=stride; y_stride=0;       break;
        case 1: stride=*(wh_iter + 1) / pool_size; x_stride=0;      y_stride=stride;  break;
        case 2: stride=*wh_iter / pool_size;       x_stride=-stride;y_stride=0;       break;
        case 3: stride=*(wh_iter + 1) / pool_size; x_stride=0;      y_stride=-stride; break;
    }

    max = bilinear_interpolate(feature_iter, height, width, y, x);
    for(int i = 1; i <= pool_size; i++) {
        x += x_stride;
        y += y_stride;
        val = bilinear_interpolate(feature_iter, height, width, y, x);
        if (val > max) {
            max = val;
        }
    }
    // Update output
    *output_iter = max;
    }
}


at::Tensor border_align_cuda_forward(
    const at::Tensor& feature,
    const at::Tensor& boxes,
    const at::Tensor& wh,
    const int pool_size)
{
    at::TensorArg feature_arg{feature, "feature", 1};
    at::TensorArg boxes_arg{boxes, "boxes", 2};
    at::TensorArg wh_arg{wh, "wh", 3};

    at::checkAllSameGPU("border_align_cuda_forward", {feature_arg, boxes_arg, wh_arg});

    AT_ASSERTM(feature.ndimension() == 4,
        "non-empty 4D(batch mode) tensor expected for feature");

    AT_ASSERTM(boxes.ndimension() == 3,
        "boxes must be 3D tensor with size of [B, H*W, 8]");

    int batch = feature.size(0);
    int fm_channel = feature.size(1);
    int out_channel = fm_channel / 4;
    int height = feature.size(2);
    int width = feature.size(3);
    int box_size = boxes.size(1);
    int output_size = batch * out_channel * box_size;

    at::Tensor pool_output = at::zeros({batch, out_channel, box_size, 4}, feature.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(std::min(THCCeilDiv((long)output_size, 64L), 4096L));
    dim3 block(128, 4);

    AT_DISPATCH_FLOATING_TYPES(feature.scalar_type(), "BorderAlign_Forward", [&] {

        scalar_t *feature_data = feature.contiguous().data_ptr<scalar_t>();
        scalar_t *boxes_data = boxes.contiguous().data_ptr<scalar_t>();
        scalar_t *wh_data = wh.contiguous().data_ptr<scalar_t>();
        scalar_t *pool_data = pool_output.contiguous().data_ptr<scalar_t>();

        BorderAlign_Forward<scalar_t><<<grid, block, 0, stream>>>(
            output_size,
            feature_data,
            boxes_data,
            wh_data,
            out_channel,
            box_size,
            height,
            width,
            pool_size,
            pool_data);
        }
    );
    THCudaCheck(cudaGetLastError());
    return pool_output;
}




template <typename T>
__global__ void BorderAlign_Backward(
    const int nthreads,
    T *gradInput,
    T *gradOutput,
    T *feature,
    T *boxes,
    T *wh,
    const int channel,
    const int box_size,
    const int height,
    const int width,
    const int pool_size)
{
    T *gradinput_iter, *gradoutput_iter, *feature_iter, *boxes_iter, *wh_iter;
    int batch_idx, box_idx, extreme_idx, fm_channel_idx;
    T stride, x_stride, y_stride;
    T x, y;
    T max, val;
    int argmax;
    T w1, w2, w3, w4;
    int x_low, x_high, y_low, y_high;

    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
    extreme_idx = threadIdx.y;
    batch_idx = index / channel / box_size;
    box_idx = index % box_size + batch_idx * box_size;
    fm_channel_idx = (int)(index / box_size) % channel;

    boxes_iter = boxes + box_idx * 4 + extreme_idx / 2 * 2;
    wh_iter = wh + box_idx * 2;
    feature_iter = feature + (batch_idx * channel * 4 + extreme_idx * channel + fm_channel_idx) * height * width;
    gradinput_iter = gradInput + (batch_idx * channel * 4 + extreme_idx * channel + fm_channel_idx) * height * width;
    gradoutput_iter = gradOutput + index * 4 + extreme_idx;

    x = *boxes_iter;
    y = *(boxes_iter + 1);

    switch(extreme_idx){
        case 0: stride=*wh_iter / pool_size;       x_stride=stride; y_stride=0;       break;
        case 1: stride=*(wh_iter + 1) / pool_size; x_stride=0;      y_stride=stride;  break;
        case 2: stride=*wh_iter / pool_size;       x_stride=-stride;y_stride=0;       break;
        case 3: stride=*(wh_iter + 1) / pool_size; x_stride=0;      y_stride=-stride; break;
    }

    max = bilinear_interpolate(feature_iter, height, width, y, x);
    argmax = 0;
    for(int i = 1; i <= pool_size; i++) {
        x += x_stride;
        y += y_stride;
        val = bilinear_interpolate(feature_iter, height, width, y, x);
        if (val > max) {
            max = val;
            argmax = i;
        }
    }
    x -= x_stride * (T)(pool_size - argmax);
    y -= y_stride * (T)(pool_size - argmax);
    bilinear_interpolate_gradient(
        height, width, y, x, w1, w2, w3, w4, x_low, x_high, y_low, y_high);
    // Update gradOutput
    atomicAdd(gradinput_iter + y_low * width + x_low, *gradoutput_iter * w1);
    atomicAdd(gradinput_iter + y_low * width + x_high, *gradoutput_iter * w2);
    atomicAdd(gradinput_iter + y_high * width + x_low, *gradoutput_iter * w3);
    atomicAdd(gradinput_iter + y_high * width + x_high, *gradoutput_iter * w4);
    }
}


at::Tensor border_align_cuda_backward(
    const at::Tensor& gradOutput,
    const at::Tensor& feature,
    const at::Tensor& boxes,
    const at::Tensor& wh,
    const int pool_size)
{
    int batch = feature.size(0);
    int fm_channel = feature.size(1);
    int out_channel = fm_channel / 4;
    int height = feature.size(2);
    int width = feature.size(3);
    int box_size = boxes.size(1);
    int output_size = batch * out_channel * box_size;

    auto gradInput = at::zeros_like(feature);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(std::min(THCCeilDiv((long)output_size, 64L), 4096L));
    dim3 block(128, 4);

    AT_DISPATCH_FLOATING_TYPES(feature.scalar_type(), "BorderAlign_Backward", [&] {

        scalar_t *gradOutput_data = gradOutput.contiguous().data_ptr<scalar_t>();
        scalar_t *gradInput_data = gradInput.contiguous().data_ptr<scalar_t>();
        scalar_t *feature_data = feature.contiguous().data_ptr<scalar_t>();
        scalar_t *boxes_data = boxes.contiguous().data_ptr<scalar_t>();
        scalar_t *wh_data = wh.contiguous().data_ptr<scalar_t>();
        
        BorderAlign_Backward<scalar_t><<<grid, block, 0, stream>>>(
            output_size,
            gradInput_data,
            gradOutput_data,
            feature_data,
            boxes_data,
            wh_data,
            out_channel,
            box_size,
            height,
            width,
            pool_size);
        }
    );
    THCudaCheck(cudaGetLastError());
    return gradInput;
}
}
