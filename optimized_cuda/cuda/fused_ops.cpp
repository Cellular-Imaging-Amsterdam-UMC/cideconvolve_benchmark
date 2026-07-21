#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void momentum_cuda(torch::Tensor current, torch::Tensor previous, double alpha, double floor, torch::Tensor output);
void ratio_cuda(torch::Tensor work, torch::Tensor observed, int64_t z, int64_t y, int64_t x, double floor);
void update_cuda(torch::Tensor momentum, torch::Tensor correction, torch::Tensor weights, double floor, torch::Tensor output);
void momentum_pack_cuda(torch::Tensor current, torch::Tensor previous, double alpha, double floor, torch::Tensor storage, int64_t z, int64_t y, int64_t x, int64_t pitch);
void ratio_pitched_cuda(torch::Tensor storage, torch::Tensor observed, int64_t wz, int64_t wy, int64_t wx, int64_t pitch, double floor, double normalization);
void multiply_otf_cuda(torch::Tensor frequency, torch::Tensor otf, bool conjugate);
void multiply_otf_half_cuda(torch::Tensor frequency, torch::Tensor otf_parts, bool conjugate);
void update_pitched_cuda(torch::Tensor momentum, torch::Tensor storage, torch::Tensor weights, double floor, double normalization, int64_t z, int64_t y, int64_t x, int64_t pitch);
void update_pitched_half_cuda(torch::Tensor momentum, torch::Tensor storage, torch::Tensor weights, double floor, double normalization, int64_t z, int64_t y, int64_t x, int64_t pitch);
void tv_update_cuda(torch::Tensor image, torch::Tensor scratch, double tv_lambda, double scale_z, double scale_y, double scale_x, double floor);
void sparse_hessian_gradient_cuda(torch::Tensor image, torch::Tensor gradient, int64_t z, int64_t y, int64_t x, double weighting, double z_scale);
void sparse_hessian_update_cuda(torch::Tensor image, torch::Tensor gradient, int64_t z, int64_t y, int64_t x, double step_over_scale, double floor);
void bind_direct_fft(pybind11::module_ &module);

void check(torch::Tensor tensor) {
    TORCH_CHECK(tensor.is_cuda(), "tensor must be CUDA");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "tensor must be float32");
    TORCH_CHECK(tensor.is_contiguous(), "tensor must be contiguous");
}

void momentum(torch::Tensor current, torch::Tensor previous, double alpha, double floor, torch::Tensor output) {
    c10::cuda::CUDAGuard guard(current.device());
    check(current); check(previous); check(output);
    TORCH_CHECK(current.sizes() == previous.sizes() && current.sizes() == output.sizes(), "shape mismatch");
    momentum_cuda(current, previous, alpha, floor, output);
}

void ratio(torch::Tensor work, torch::Tensor observed, int64_t z, int64_t y, int64_t x, double floor) {
    c10::cuda::CUDAGuard guard(work.device());
    check(work); check(observed);
    ratio_cuda(work, observed, z, y, x, floor);
}

void update(torch::Tensor momentum_value, torch::Tensor correction, torch::Tensor weights, double floor, torch::Tensor output) {
    c10::cuda::CUDAGuard guard(momentum_value.device());
    check(momentum_value); check(correction); check(weights); check(output);
    TORCH_CHECK(momentum_value.sizes() == correction.sizes() && momentum_value.sizes() == weights.sizes() && momentum_value.sizes() == output.sizes(), "shape mismatch");
    update_cuda(momentum_value, correction, weights, floor, output);
}

void momentum_pack(torch::Tensor current, torch::Tensor previous, double alpha, double floor, torch::Tensor storage, int64_t z, int64_t y, int64_t x, int64_t pitch) {
    c10::cuda::CUDAGuard guard(current.device());
    check(current); check(previous); check(storage);
    momentum_pack_cuda(current, previous, alpha, floor, storage, z, y, x, pitch);
}

void ratio_pitched(torch::Tensor storage, torch::Tensor observed, int64_t wz, int64_t wy, int64_t wx, int64_t pitch, double floor, double normalization) {
    c10::cuda::CUDAGuard guard(storage.device());
    check(storage); check(observed);
    ratio_pitched_cuda(storage, observed, wz, wy, wx, pitch, floor, normalization);
}

void multiply_otf(torch::Tensor frequency, torch::Tensor otf, bool conjugate) {
    c10::cuda::CUDAGuard guard(frequency.device());
    TORCH_CHECK(frequency.is_cuda() && otf.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(frequency.scalar_type() == torch::kComplexFloat && otf.scalar_type() == torch::kComplexFloat, "frequency and OTF must be complex64");
    TORCH_CHECK(frequency.is_contiguous() && otf.is_contiguous() && frequency.numel() == otf.numel(), "frequency/OTF layout mismatch");
    multiply_otf_cuda(frequency, otf, conjugate);
}

void multiply_otf_half(torch::Tensor frequency, torch::Tensor otf_parts, bool conjugate) {
    c10::cuda::CUDAGuard guard(frequency.device());
    TORCH_CHECK(frequency.is_cuda() && otf_parts.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(frequency.scalar_type() == torch::kComplexFloat && otf_parts.scalar_type() == torch::kFloat16, "expected complex64 frequency and float16 OTF parts");
    TORCH_CHECK(frequency.is_contiguous() && otf_parts.is_contiguous() && otf_parts.numel() == 2 * frequency.numel(), "frequency/OTF layout mismatch");
    multiply_otf_half_cuda(frequency, otf_parts, conjugate);
}

void update_pitched(torch::Tensor momentum_value, torch::Tensor storage, torch::Tensor weights, double floor, double normalization, int64_t z, int64_t y, int64_t x, int64_t pitch) {
    c10::cuda::CUDAGuard guard(momentum_value.device());
    check(momentum_value); check(storage); check(weights);
    update_pitched_cuda(momentum_value, storage, weights, floor, normalization, z, y, x, pitch);
}

void update_pitched_half(torch::Tensor momentum_value, torch::Tensor storage, torch::Tensor weights, double floor, double normalization, int64_t z, int64_t y, int64_t x, int64_t pitch) {
    c10::cuda::CUDAGuard guard(momentum_value.device());
    check(momentum_value); check(storage);
    TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == torch::kFloat16 && weights.is_contiguous(), "weights must be contiguous CUDA float16");
    update_pitched_half_cuda(momentum_value, storage, weights, floor, normalization, z, y, x, pitch);
}

void tv_update(torch::Tensor image, torch::Tensor scratch, double tv_lambda, double scale_z, double scale_y, double scale_x, double floor) {
    c10::cuda::CUDAGuard guard(image.device());
    check(image); check(scratch);
    TORCH_CHECK(image.dim() == 3 && scratch.dim() == 3, "TV benchmark requires 3D tensors");
    TORCH_CHECK(scratch.size(0) == image.size(0) && scratch.size(1) == image.size(1) && scratch.size(2) >= image.size(2), "TV scratch is too small");
    tv_update_cuda(image, scratch, tv_lambda, scale_z, scale_y, scale_x, floor);
}

void sparse_hessian_gradient(torch::Tensor image, torch::Tensor gradient, int64_t z, int64_t y, int64_t x, double weighting, double z_scale) {
    c10::cuda::CUDAGuard guard(image.device());
    check(image); check(gradient);
    TORCH_CHECK(image.dim() == 3 && gradient.dim() == 3, "sparse Hessian benchmark requires 3D tensors");
    TORCH_CHECK(gradient.size(0) == z && gradient.size(1) == y && gradient.size(2) == x, "gradient/support shape mismatch");
    sparse_hessian_gradient_cuda(image, gradient, z, y, x, weighting, z_scale);
}

void sparse_hessian_update(torch::Tensor image, torch::Tensor gradient, int64_t z, int64_t y, int64_t x, double step_over_scale, double floor) {
    c10::cuda::CUDAGuard guard(image.device());
    check(image); check(gradient);
    sparse_hessian_update_cuda(image, gradient, z, y, x, step_over_scale, floor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("momentum", &momentum, "Fused SHB momentum and clamp");
    module.def("ratio", &ratio, "Fused support masking and ratio, in place");
    module.def("update", &update, "Fused RL update and clamp");
    module.def("momentum_pack", &momentum_pack, "Overwrite previous estimate and pack in-place FFT input");
    module.def("ratio_pitched", &ratio_pitched, "Build ratio in an in-place cuFFT buffer");
    module.def("multiply_otf", &multiply_otf, "Multiply complex spectrum by FP32 OTF");
    module.def("multiply_otf_half", &multiply_otf_half, "Multiply complex spectrum by packed FP16 OTF");
    module.def("update_pitched", &update_pitched, "In-place RL update from pitched cuFFT output");
    module.def("update_pitched_half", &update_pitched_half, "In-place RL update with FP16 boundary weights");
    module.def("tv_update", &tv_update, "Fused production-equivalent 3D TV update using caller scratch");
    module.def("sparse_hessian_gradient", &sparse_hessian_gradient, "Explicit sparse-Hessian gradient without autograd");
    module.def("sparse_hessian_update", &sparse_hessian_update, "Fused normalized sparse-Hessian update");
    bind_direct_fft(module);
}
