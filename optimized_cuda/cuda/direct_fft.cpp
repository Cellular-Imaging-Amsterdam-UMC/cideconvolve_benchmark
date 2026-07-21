#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cufft.h>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

static void cufft_check(cufftResult status, const char* operation) {
    if (status != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with cuFFT status " + std::to_string(static_cast<int>(status)));
    }
}

class DirectFFTPlan {
public:
    DirectFFTPlan(int64_t z, int64_t y, int64_t x)
        : z_(z), y_(y), x_(x), device_index_(at::cuda::current_device()) {
        c10::cuda::CUDAGuard guard(device_index_);
        TORCH_CHECK(z > 0 && y > 0 && x > 0, "FFT dimensions must be positive");
        cufft_check(cufftCreate(&forward_), "cufftCreate(R2C)");
        cufft_check(cufftCreate(&inverse_), "cufftCreate(C2R)");
        cufft_check(cufftSetAutoAllocation(forward_, 0), "cufftSetAutoAllocation(R2C)");
        cufft_check(cufftSetAutoAllocation(inverse_, 0), "cufftSetAutoAllocation(C2R)");
        size_t forward_work = 0, inverse_work = 0;
        cufft_check(cufftMakePlan3d(forward_, static_cast<int>(z), static_cast<int>(y), static_cast<int>(x), CUFFT_R2C, &forward_work), "cufftMakePlan3d(R2C)");
        cufft_check(cufftMakePlan3d(inverse_, static_cast<int>(z), static_cast<int>(y), static_cast<int>(x), CUFFT_C2R, &inverse_work), "cufftMakePlan3d(C2R)");
        workspace_bytes_ = std::max(forward_work, inverse_work);
    }

    ~DirectFFTPlan() {
        c10::cuda::CUDAGuard guard(device_index_);
        if (forward_) cufftDestroy(forward_);
        if (inverse_) cufftDestroy(inverse_);
    }

    size_t workspace_bytes() const { return workspace_bytes_; }
    int64_t physical_x() const { return 2 * (x_ / 2 + 1); }

    void forward(torch::Tensor storage, torch::Tensor workspace) {
        c10::cuda::CUDAGuard guard(device_index_);
        prepare(storage, workspace, forward_);
        cufft_check(cufftExecR2C(forward_, reinterpret_cast<cufftReal*>(storage.data_ptr<float>()), reinterpret_cast<cufftComplex*>(storage.data_ptr<float>())), "cufftExecR2C");
    }

    void inverse(torch::Tensor storage, torch::Tensor workspace) {
        c10::cuda::CUDAGuard guard(device_index_);
        prepare(storage, workspace, inverse_);
        cufft_check(cufftExecC2R(inverse_, reinterpret_cast<cufftComplex*>(storage.data_ptr<float>()), reinterpret_cast<cufftReal*>(storage.data_ptr<float>())), "cufftExecC2R");
    }

private:
    void prepare(torch::Tensor storage, torch::Tensor workspace, cufftHandle plan) {
        TORCH_CHECK(storage.is_cuda() && storage.scalar_type() == torch::kFloat32 && storage.is_contiguous(), "storage must be contiguous CUDA float32");
        TORCH_CHECK(storage.get_device() == device_index_, "storage is on the wrong CUDA device");
        TORCH_CHECK(storage.numel() >= z_ * y_ * physical_x(), "storage is too small");
        TORCH_CHECK(workspace.is_cuda() && workspace.scalar_type() == torch::kUInt8 && workspace.is_contiguous(), "workspace must be contiguous CUDA uint8");
        TORCH_CHECK(workspace.get_device() == device_index_, "workspace is on the wrong CUDA device");
        TORCH_CHECK(static_cast<size_t>(workspace.numel()) >= workspace_bytes_, "workspace is too small");
        const auto stream = at::cuda::getCurrentCUDAStream();
        cufft_check(cufftSetStream(plan, stream), "cufftSetStream");
        cufft_check(cufftSetWorkArea(plan, workspace.data_ptr()), "cufftSetWorkArea");
    }

    int64_t z_, y_, x_;
    c10::DeviceIndex device_index_;
    cufftHandle forward_ = 0, inverse_ = 0;
    size_t workspace_bytes_ = 0;
};

void bind_direct_fft(py::module_ &module) {
    py::class_<DirectFFTPlan>(module, "DirectFFTPlan")
        .def(py::init<int64_t, int64_t, int64_t>())
        .def_property_readonly("workspace_bytes", &DirectFFTPlan::workspace_bytes)
        .def_property_readonly("physical_x", &DirectFFTPlan::physical_x)
        .def("forward", &DirectFFTPlan::forward)
        .def("inverse", &DirectFFTPlan::inverse);
}
