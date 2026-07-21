#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {
constexpr int THREADS = 256;

__global__ void momentum_kernel(const float* current, const float* previous, float alpha, float floor, float* output, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float value = current[index] + alpha * (current[index] - previous[index]);
    output[index] = value < floor ? floor : value;
}

__global__ void ratio_kernel(float* work, const float* observed, int64_t wz, int64_t wy, int64_t wx, int64_t z, int64_t y, int64_t x, float floor, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int64_t ix = index % wx;
    const int64_t iy = (index / wx) % wy;
    const int64_t iz = index / (wx * wy);
    if (iz < z && iy < y && ix < x) {
        const int64_t observed_index = (iz * y + iy) * x + ix;
        const float denominator = work[index] < floor ? floor : work[index];
        work[index] = observed[observed_index] / denominator;
    } else {
        work[index] = 0.0f;
    }
}

__global__ void update_kernel(const float* momentum, const float* correction, const float* weights, float floor, float* output, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float value = momentum[index] * correction[index] * weights[index];
    output[index] = value < floor ? floor : value;
}

__global__ void momentum_pack_kernel(const float* current, float* previous, float alpha, float floor, float* storage, int64_t y, int64_t x, int64_t pitch, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float value = fmaxf(current[index] + alpha * (current[index] - previous[index]), floor);
    previous[index] = value;
    const int64_t ix = index % x;
    const int64_t iyz = index / x;
    storage[iyz * pitch + ix] = value;
}

__global__ void ratio_pitched_kernel(float* storage, const float* observed, int64_t wz, int64_t wy, int64_t wx, int64_t pitch, int64_t z, int64_t y, int64_t x, float floor, float normalization, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int64_t ix = index % wx;
    const int64_t iy = (index / wx) % wy;
    const int64_t iz = index / (wx * wy);
    const int64_t physical = (iz * wy + iy) * pitch + ix;
    if (iz < z && iy < y && ix < x) {
        const int64_t observed_index = (iz * y + iy) * x + ix;
        storage[physical] = observed[observed_index] * normalization / fmaxf(storage[physical], floor * normalization);
    } else {
        storage[physical] = 0.0f;
    }
}

__global__ void multiply_otf_kernel(float2* frequency, const float2* otf, bool conjugate, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float2 a = frequency[index];
    float2 b = otf[index];
    if (conjugate) b.y = -b.y;
    frequency[index] = make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__global__ void multiply_otf_half_kernel(float2* frequency, const __half* otf, bool conjugate, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float2 a = frequency[index];
    const float br = __half2float(otf[2 * index]);
    float bi = __half2float(otf[2 * index + 1]);
    if (conjugate) bi = -bi;
    frequency[index] = make_float2(a.x * br - a.y * bi, a.x * bi + a.y * br);
}

__device__ __forceinline__ float weight_to_float(float value) { return value; }
__device__ __forceinline__ float weight_to_float(__half value) { return __half2float(value); }

template <typename Weight>
__global__ void update_pitched_kernel(float* momentum, const float* storage, const Weight* weights, float floor, float inv_normalization, int64_t y, int64_t x, int64_t pitch, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int64_t ix = index % x;
    const int64_t iyz = index / x;
    const float weight = weight_to_float(weights[index]);
    momentum[index] = fmaxf(momentum[index] * storage[iyz * pitch + ix] * inv_normalization * weight, floor);
}

__device__ __forceinline__ float tv_normed_component(
    const float* image, int64_t iz, int64_t iy, int64_t ix,
    int64_t z, int64_t y, int64_t x, int component,
    float scale_z, float scale_y, float scale_x) {
    const int64_t index = (iz * y + iy) * x + ix;
    float gz = 0.0f, gy = 0.0f, gx = 0.0f;
    if (iz > 0) gz = (image[index] - image[index - y * x]) * scale_z;
    if (iy > 0) gy = (image[index] - image[index - x]) * scale_y;
    if (ix > 0) gx = (image[index] - image[index - 1]) * scale_x;
    const float inv_mag = rsqrtf(gz * gz + gy * gy + gx * gx + 1.0e-8f);
    return (component == 0 ? gz : (component == 1 ? gy : gx)) * inv_mag;
}

__global__ void tv_factor_kernel(
    const float* image, float* scratch, int64_t z, int64_t y, int64_t x, int64_t pitch,
    float tv_lambda, float scale_z, float scale_y, float scale_x, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int64_t ix = index % x;
    const int64_t iy = (index / x) % y;
    const int64_t iz = index / (x * y);
    float div = 0.0f;
    if (iz + 1 < z) div += (tv_normed_component(image, iz, iy, ix, z, y, x, 0, scale_z, scale_y, scale_x) - tv_normed_component(image, iz + 1, iy, ix, z, y, x, 0, scale_z, scale_y, scale_x)) * scale_z;
    if (iy + 1 < y) div += (tv_normed_component(image, iz, iy, ix, z, y, x, 1, scale_z, scale_y, scale_x) - tv_normed_component(image, iz, iy + 1, ix, z, y, x, 1, scale_z, scale_y, scale_x)) * scale_y;
    if (ix + 1 < x) div += (tv_normed_component(image, iz, iy, ix, z, y, x, 2, scale_z, scale_y, scale_x) - tv_normed_component(image, iz, iy, ix + 1, z, y, x, 2, scale_z, scale_y, scale_x)) * scale_x;
    float factor = 1.0f / (1.0f - tv_lambda * div);
    factor = fminf(fmaxf(factor, 0.1f), 10.0f);
    scratch[(iz * y + iy) * pitch + ix] = factor;
}

__global__ void tv_apply_kernel(float* image, const float* scratch, int64_t y, int64_t x, int64_t pitch, float floor, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int64_t ix = index % x;
    const int64_t iyz = index / x;
    image[index] = fmaxf(image[index] * scratch[iyz * pitch + ix], floor);
}

__device__ __forceinline__ float sparse_term_contribution(
    const float* image, int64_t qz, int64_t qy, int64_t qx,
    int64_t pz, int64_t py, int64_t px,
    int64_t support_z, int64_t support_y, int64_t support_x, int64_t work_y, int64_t work_x,
    float weighting, float z_scale) {
    if (qz <= 0 || qz >= support_z - 1 || qy <= 0 || qy >= support_y - 1 || qx <= 0 || qx >= support_x - 1) return 0.0f;
    const int64_t sy = work_x;
    const int64_t sz = work_y * work_x;
    const int64_t q = (qz * work_y + qy) * work_x + qx;
    const float core = image[q];
    const float dxx = -image[q + 1] + 2.0f * core - image[q - 1];
    const float dyy = -image[q + sy] + 2.0f * core - image[q - sy];
    const float dzz = z_scale * z_scale * (-image[q + sz] + 2.0f * core - image[q - sz]);
    const float dxy = image[q + sy + 1] - image[q + 1] - image[q + sy] + core;
    const float dxz = z_scale * (image[q + sz + 1] - image[q + 1] - image[q + sz] + core);
    const float dyz = z_scale * (image[q + sz + sy] - image[q + sy] - image[q + sz] + core);
    const float w2 = weighting * weighting;
    const float s2 = (1.0f - weighting) * (1.0f - weighting);
    const float root = sqrtf(w2 * (dxx*dxx + dyy*dyy + dzz*dzz + 2.0f*dxy*dxy + 2.0f*dxz*dxz + 2.0f*dyz*dyz) + s2 * core * core + 1.0e-8f);
    const int dz = static_cast<int>(pz - qz), dy = static_cast<int>(py - qy), dx = static_cast<int>(px - qx);
    float numerator = 0.0f;
    if (dz == 0 && dy == 0 && dx >= -1 && dx <= 1) numerator += w2 * dxx * (dx == 0 ? 2.0f : -1.0f);
    if (dz == 0 && dx == 0 && dy >= -1 && dy <= 1) numerator += w2 * dyy * (dy == 0 ? 2.0f : -1.0f);
    if (dy == 0 && dx == 0 && dz >= -1 && dz <= 1) numerator += w2 * dzz * z_scale * z_scale * (dz == 0 ? 2.0f : -1.0f);
    if (dz == 0 && dy >= 0 && dy <= 1 && dx >= 0 && dx <= 1) numerator += 2.0f * w2 * dxy * (dy == dx ? 1.0f : -1.0f);
    if (dy == 0 && dz >= 0 && dz <= 1 && dx >= 0 && dx <= 1) numerator += 2.0f * w2 * dxz * z_scale * (dz == dx ? 1.0f : -1.0f);
    if (dx == 0 && dz >= 0 && dz <= 1 && dy >= 0 && dy <= 1) numerator += 2.0f * w2 * dyz * z_scale * (dz == dy ? 1.0f : -1.0f);
    if (dz == 0 && dy == 0 && dx == 0) numerator += s2 * core;
    return numerator / root;
}

__device__ __forceinline__ float sparse_2d_term_contribution(
    const float* image, int64_t qy, int64_t qx, int64_t py, int64_t px,
    int64_t support_y, int64_t support_x, int64_t work_x, float weighting) {
    if (qy <= 0 || qy >= support_y - 1 || qx <= 0 || qx >= support_x - 1) return 0.0f;
    const int64_t q = qy * work_x + qx;
    const float core = image[q];
    const float dxx = -image[q + work_x] + 2.0f * core - image[q - work_x];
    const float dyy = -image[q + 1] + 2.0f * core - image[q - 1];
    const float dxy = image[q + work_x + 1] - image[q + work_x] - image[q + 1] + core;
    const float w2 = weighting * weighting;
    const float s2 = (1.0f - weighting) * (1.0f - weighting);
    const float root = sqrtf(w2 * (dxx*dxx + dyy*dyy + 2.0f*dxy*dxy) + s2 * core * core + 1.0e-8f);
    const int dy = static_cast<int>(py - qy), dx = static_cast<int>(px - qx);
    float numerator = 0.0f;
    if (dx == 0 && dy >= -1 && dy <= 1) numerator += w2 * dxx * (dy == 0 ? 2.0f : -1.0f);
    if (dy == 0 && dx >= -1 && dx <= 1) numerator += w2 * dyy * (dx == 0 ? 2.0f : -1.0f);
    if (dy >= 0 && dy <= 1 && dx >= 0 && dx <= 1) numerator += 2.0f * w2 * dxy * (dy == dx ? 1.0f : -1.0f);
    if (dy == 0 && dx == 0) numerator += s2 * core;
    return numerator / root;
}

__global__ void sparse_hessian_gradient_kernel(
    const float* image, float* gradient, int64_t z, int64_t y, int64_t x,
    int64_t work_y, int64_t work_x, float weighting, float z_scale, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int64_t px = index % x;
    const int64_t py = (index / x) % y;
    const int64_t pz = index / (x * y);
    if (z == 1) {
        const int offsets_2d[6][2] = {{0,0}, {0,-1}, {0,1}, {-1,0}, {1,0}, {-1,-1}};
        float value_2d = 0.0f;
        for (int i = 0; i < 6; ++i) {
            value_2d += sparse_2d_term_contribution(image, py + offsets_2d[i][0], px + offsets_2d[i][1], py, px, y, x, work_x, weighting);
        }
        gradient[index] = value_2d;
        return;
    }
    // Unique penalty centres whose 3D Hessian stencil can contain this voxel.
    const int offsets[10][3] = {
        {0,0,0}, {0,0,-1}, {0,0,1}, {0,-1,0}, {0,1,0}, {-1,0,0}, {1,0,0},
        {0,-1,-1}, {-1,0,-1}, {-1,-1,0}
    };
    float value = 0.0f;
    for (int i = 0; i < 10; ++i) {
        value += sparse_term_contribution(image, pz + offsets[i][0], py + offsets[i][1], px + offsets[i][2], pz, py, px, z, y, x, work_y, work_x, weighting, z_scale);
    }
    gradient[index] = value;
}

__global__ void sparse_hessian_update_kernel(float* image, const float* gradient, int64_t y, int64_t x, int64_t work_y, int64_t work_x, float step_over_scale, float floor, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int64_t ix = index % x;
    const int64_t iy = (index / x) % y;
    const int64_t iz = index / (x * y);
    const int64_t work_index = (iz * work_y + iy) * work_x + ix;
    image[work_index] = fmaxf(image[work_index] - step_over_scale * gradient[index], floor);
}
}

void momentum_cuda(torch::Tensor current, torch::Tensor previous, double alpha, double floor, torch::Tensor output) {
    const auto count = current.numel();
    const auto stream = at::cuda::getCurrentCUDAStream();
    momentum_kernel<<<(count + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        current.data_ptr<float>(), previous.data_ptr<float>(), static_cast<float>(alpha), static_cast<float>(floor), output.data_ptr<float>(), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ratio_cuda(torch::Tensor work, torch::Tensor observed, int64_t z, int64_t y, int64_t x, double floor) {
    const auto count = work.numel();
    TORCH_CHECK(work.dim() == 3, "work tensor must be 3D");
    const auto stream = at::cuda::getCurrentCUDAStream();
    ratio_kernel<<<(count + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        work.data_ptr<float>(), observed.data_ptr<float>(), work.size(0), work.size(1), work.size(2), z, y, x,
        static_cast<float>(floor), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void update_cuda(torch::Tensor momentum, torch::Tensor correction, torch::Tensor weights, double floor, torch::Tensor output) {
    const auto count = momentum.numel();
    const auto stream = at::cuda::getCurrentCUDAStream();
    update_kernel<<<(count + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        momentum.data_ptr<float>(), correction.data_ptr<float>(), weights.data_ptr<float>(), static_cast<float>(floor), output.data_ptr<float>(), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void momentum_pack_cuda(torch::Tensor current, torch::Tensor previous, double alpha, double floor, torch::Tensor storage, int64_t z, int64_t y, int64_t x, int64_t pitch) {
    const int64_t count = z * y * x;
    const auto stream = at::cuda::getCurrentCUDAStream();
    momentum_pack_kernel<<<(count + THREADS - 1) / THREADS, THREADS, 0, stream>>>(current.data_ptr<float>(), previous.data_ptr<float>(), static_cast<float>(alpha), static_cast<float>(floor), storage.data_ptr<float>(), y, x, pitch, count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ratio_pitched_cuda(torch::Tensor storage, torch::Tensor observed, int64_t wz, int64_t wy, int64_t wx, int64_t pitch, double floor, double normalization) {
    const int64_t count = wz * wy * wx;
    const auto stream = at::cuda::getCurrentCUDAStream();
    ratio_pitched_kernel<<<(count + THREADS - 1) / THREADS, THREADS, 0, stream>>>(storage.data_ptr<float>(), observed.data_ptr<float>(), wz, wy, wx, pitch, observed.size(0), observed.size(1), observed.size(2), static_cast<float>(floor), static_cast<float>(normalization), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void multiply_otf_cuda(torch::Tensor frequency, torch::Tensor otf, bool conjugate) {
    const int64_t count = frequency.numel();
    const auto stream = at::cuda::getCurrentCUDAStream();
    multiply_otf_kernel<<<(count + THREADS - 1) / THREADS, THREADS, 0, stream>>>(reinterpret_cast<float2*>(frequency.data_ptr<c10::complex<float>>()), reinterpret_cast<const float2*>(otf.data_ptr<c10::complex<float>>()), conjugate, count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void multiply_otf_half_cuda(torch::Tensor frequency, torch::Tensor otf_parts, bool conjugate) {
    const int64_t count = frequency.numel();
    const auto stream = at::cuda::getCurrentCUDAStream();
    multiply_otf_half_kernel<<<(count + THREADS - 1) / THREADS, THREADS, 0, stream>>>(reinterpret_cast<float2*>(frequency.data_ptr<c10::complex<float>>()), reinterpret_cast<const __half*>(otf_parts.data_ptr<at::Half>()), conjugate, count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void update_pitched_cuda(torch::Tensor momentum, torch::Tensor storage, torch::Tensor weights, double floor, double normalization, int64_t z, int64_t y, int64_t x, int64_t pitch) {
    const int64_t count = z * y * x;
    const auto stream = at::cuda::getCurrentCUDAStream();
    update_pitched_kernel<float><<<(count + THREADS - 1) / THREADS, THREADS, 0, stream>>>(momentum.data_ptr<float>(), storage.data_ptr<float>(), weights.data_ptr<float>(), static_cast<float>(floor), static_cast<float>(1.0 / normalization), y, x, pitch, count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void update_pitched_half_cuda(torch::Tensor momentum, torch::Tensor storage, torch::Tensor weights, double floor, double normalization, int64_t z, int64_t y, int64_t x, int64_t pitch) {
    const int64_t count = z * y * x;
    const auto stream = at::cuda::getCurrentCUDAStream();
    update_pitched_kernel<__half><<<(count + THREADS - 1) / THREADS, THREADS, 0, stream>>>(momentum.data_ptr<float>(), storage.data_ptr<float>(), reinterpret_cast<const __half*>(weights.data_ptr<at::Half>()), static_cast<float>(floor), static_cast<float>(1.0 / normalization), y, x, pitch, count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tv_update_cuda(torch::Tensor image, torch::Tensor scratch, double tv_lambda, double scale_z, double scale_y, double scale_x, double floor) {
    const int64_t count = image.numel();
    const auto stream = at::cuda::getCurrentCUDAStream();
    tv_factor_kernel<<<(count + THREADS - 1) / THREADS, THREADS, 0, stream>>>(image.data_ptr<float>(), scratch.data_ptr<float>(), image.size(0), image.size(1), image.size(2), scratch.size(2), static_cast<float>(tv_lambda), static_cast<float>(scale_z), static_cast<float>(scale_y), static_cast<float>(scale_x), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    tv_apply_kernel<<<(count + THREADS - 1) / THREADS, THREADS, 0, stream>>>(image.data_ptr<float>(), scratch.data_ptr<float>(), image.size(1), image.size(2), scratch.size(2), static_cast<float>(floor), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_hessian_gradient_cuda(torch::Tensor image, torch::Tensor gradient, int64_t z, int64_t y, int64_t x, double weighting, double z_scale) {
    const int64_t count = z * y * x;
    const auto stream = at::cuda::getCurrentCUDAStream();
    sparse_hessian_gradient_kernel<<<(count + THREADS - 1) / THREADS, THREADS, 0, stream>>>(image.data_ptr<float>(), gradient.data_ptr<float>(), z, y, x, image.size(1), image.size(2), static_cast<float>(weighting), static_cast<float>(z_scale), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_hessian_update_cuda(torch::Tensor image, torch::Tensor gradient, int64_t z, int64_t y, int64_t x, double step_over_scale, double floor) {
    const int64_t count = z * y * x;
    const auto stream = at::cuda::getCurrentCUDAStream();
    sparse_hessian_update_kernel<<<(count + THREADS - 1) / THREADS, THREADS, 0, stream>>>(image.data_ptr<float>(), gradient.data_ptr<float>(), y, x, image.size(1), image.size(2), static_cast<float>(step_over_scale), static_cast<float>(floor), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
