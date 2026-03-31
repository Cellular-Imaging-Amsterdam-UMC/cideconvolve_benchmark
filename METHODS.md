# CIDeconvolve — Deconvolution Methods

CIDeconvolve bundles **11 deconvolution methods** from 6 independent
libraries.  Each method generates a theoretically correct PSF on-the-fly
using metadata extracted from the input OME-TIFF (NA, refractive indices,
wavelengths, voxel spacing, microscope type) and then applies the chosen
deconvolution algorithm.

> **PSF generation** uses a vendored copy of
> [PSF Generator](https://bigwww.epfl.ch/algorithms/psfgenerator/) (EPFL).
> It selects a *Vectorial* propagator for high-NA objectives (NA ≥ 0.9) or
> a *Scalar* propagator otherwise, and applies the
> [Gibson–Lanni](https://doi.org/10.1364/JOSAA.9.000154) aberration model
> for refractive-index mismatch between immersion and sample media.

---

## Method overview

| # | Method key | Algorithm | Library | Compute | 2-D | 3-D | Linux | Windows | WSL2 |
|---|-----------|-----------|---------|---------|-----|-----|-------|---------|------|
| 1 | `sdeconv_rl` | Richardson–Lucy | sdeconv (vendored) | CUDA / CPU | ✅ | ✅ | 🟢 GPU | 🟢 GPU | 🟢 GPU |
| 2 | `sdeconv_wiener` | Wiener filter | sdeconv (vendored) | CUDA / CPU | ✅ | ✅ | 🟢 GPU | 🟢 GPU | 🟢 GPU |
| 3 | `sdeconv_spitfire` | SPITFIRE (sparse + TV) | sdeconv (vendored) | CUDA / CPU | ✅ | ✅ | 🟢 GPU | 🟢 GPU | 🟢 GPU |
| 4 | `pycudadecon_rl_cuda` | Richardson–Lucy | pycudadecon | CUDA | ❌ | ✅ | 🟢 GPU | 🟢 GPU | 🟢 GPU |
| 5 | `deconwolf_rl` | Richardson–Lucy | deconwolf | OpenCL / CPU | ❌ | ✅ | 🟢 GPU | 🟢 GPU | 🟡 CPU |
| 6 | `deconwolf_shb` | SHB (RL variant) | deconwolf | OpenCL / CPU | ❌ | ✅ | 🟢 GPU | 🟢 GPU | 🟡 CPU |
| 7 | `deconvlab2_rl` | Richardson–Lucy | DeconvolutionLab2 | CPU (Java) | ✅ | ✅ | 🟡 CPU | 🟡 CPU | 🟡 CPU |
| 8 | `deconvlab2_rltv` | RL + Total Variation | DeconvolutionLab2 | CPU (Java) | ✅ | ✅ | 🟡 CPU | 🟡 CPU | 🟡 CPU |
| 9 | `redlionfish_rl` | Richardson–Lucy | RedLionfish | OpenCL / CPU | ❌ | ✅ | 🟢 GPU | 🟢 GPU | 🟡 CPU |
| 10 | `skimage_rl` | Richardson–Lucy | scikit-image | CPU | ✅ | ✅ | 🟡 CPU | 🟡 CPU | 🟡 CPU |
| 11 | `skimage_cucim_rl` | Richardson–Lucy | scikit-image + cuCIM | CUDA | ✅ | ✅ | 🟢 GPU | ❌ | 🟢 GPU |

> 🟢 **GPU** = runs with GPU acceleration.  🟡 **CPU** = runs but CPU-only.
> **❌** = not available on that platform.  
> **WSL2** = Windows Subsystem for Linux 2 (including Docker Desktop with WSL2 backend).  
> OpenCL methods (deconwolf, RedLionfish) cannot use the GPU under WSL2 because
> WSL2 does not expose an OpenCL GPU driver — they fall back to CPU.  CUDA methods
> work with GPU on all three platforms via the NVIDIA CUDA driver.

---

## Method details

### 1–3. sdeconv — `sdeconv_rl`, `sdeconv_wiener`, `sdeconv_spitfire`

A vendored pure-Python/PyTorch library implementing three classic
deconvolution algorithms:

- **Richardson–Lucy (RL)** — iterative maximum-likelihood restoration.
  The foundational algorithm described by
  [Richardson (1972)](https://doi.org/10.1364/JOSA.62.000055) and
  [Lucy (1974)](https://doi.org/10.1086/111605).
- **Wiener filter** — frequency-domain least-squares restoration with
  noise regularisation (single-pass, very fast).
- **SPITFIRE** — SParse fluoRescence Image resToration using a combined
  sparsity + total-variation (TV) prior.  Based on
  [Descloux et al. (2019)](https://doi.org/10.1038/s41592-019-0515-9).

All three methods use PyTorch tensors and can run on CUDA GPUs when
available (`device=cuda`), falling back to CPU otherwise.

- **Source:** vendored in `vendor/sdeconv/` (originally from
  [sdeconv on GitHub](https://github.com/sylvainprigent/sdeconv))
- **2-D support:** ✅ Yes
- **GPU:** CUDA via PyTorch

### 4. pycudadecon — `pycudadecon_rl_cuda`

CUDA-accelerated Richardson–Lucy deconvolution developed at the
Howard Hughes Medical Institute (HHMI) Janelia Research Campus as part of
the **cudaDecon** project.  Extremely fast for 3-D volumes but
requires an NVIDIA GPU.

- **Source:** [tlambert03/pycudadecon](https://github.com/tlambert03/pycudadecon)
- **Based on:** [scopetools/cudaDecon](https://github.com/scopetools/cudaDecon)
- **Publication:** Part of the lattice light-sheet microscopy pipeline
  described by [Chen et al. (2014)](https://doi.org/10.1126/science.1257998).
- **2-D support:** ❌ 3-D only (CUDA FFT requires Z > 1)
- **GPU:** CUDA (required)
- **Install (Docker):** `conda install -c conda-forge pycudadecon`

### 5–6. deconwolf — `deconwolf_rl`, `deconwolf_shb`

A high-performance C implementation that tightly integrates FFTW wisdom
planning and optional OpenCL GPU acceleration.  Provides two algorithms:

- **RL** — standard Richardson–Lucy.
- **SHB** — Schaefer–Haase–Bhargava RL variant with improved convergence
  for noisy data.

Deconwolf is compiled from source in the Docker image with OpenCL support
enabled.  On systems without an OpenCL device it falls back to
multi-threaded CPU execution.

- **Source:** [elgw/deconwolf](https://github.com/elgw/deconwolf)
- **Publication:** [Wernersson et al. (2024)](https://doi.org/10.1038/s41592-024-02294-7)
  *"Deconwolf enables high-performance deconvolution of widefield fluorescence
  microscopy images."* Nature Methods.
- **2-D support:** ❌ 3-D only
- **GPU:** OpenCL (with CPU fallback)
- **Platform note:** Runs on Linux, Windows (native binaries with GPU on the
  [releases page](https://github.com/elgw/deconwolf/releases)), and WSL2
  (CPU only — no OpenCL GPU passthrough).

### 7–8. DeconvolutionLab2 — `deconvlab2_rl`, `deconvlab2_rltv`

A widely-used Java/ImageJ-based deconvolution toolbox from EPFL.  Runs
headless via a command-line interface using the bundled ImageJ JAR.

- **RL** — standard Richardson–Lucy.
- **RLTV** — Richardson–Lucy with Total Variation regularisation.

CPU-only, but well-tested and supports both 2-D and 3-D images.

- **Source:** [Biomedical Imaging Group, EPFL](http://bigwww.epfl.ch/deconvolution/deconvolutionlab2/)
- **Publication:** [Sage et al. (2017)](https://doi.org/10.1016/j.ymeth.2016.12.015)
  *"DeconvolutionLab2: An open-source software for deconvolution microscopy."*
  Methods **115**, 28–41.
- **2-D support:** ✅ Yes
- **GPU:** None (CPU only, multi-threaded via Java)

### 9. RedLionfish — `redlionfish_rl`

OpenCL-accelerated Richardson–Lucy deconvolution developed at the
Rosalind Franklin Institute.

- **Source:** [rosalindfranklininstitute/RedLionfish](https://github.com/rosalindfranklininstitute/RedLionfish)
- **2-D support:** ❌ 3-D only
- **GPU:** OpenCL (with CPU fallback)

### 10. scikit-image — `skimage_rl`

The `restoration.richardson_lucy` implementation from scikit-image.  Pure
Python/NumPy/SciPy — no GPU acceleration but universally available and
easy to install.

- **Source:** [scikit-image.org](https://scikit-image.org/)
- **Publication:** [van der Walt et al. (2014)](https://doi.org/10.7717/peerj.453)
  *"scikit-image: image processing in Python."* PeerJ **2**, e453.
- **2-D support:** ✅ Yes
- **GPU:** None (CPU only)

### 11. scikit-image + cuCIM — `skimage_cucim_rl`

Uses NVIDIA's [cuCIM](https://github.com/rapidsai/cucim) library (part of
the RAPIDS ecosystem) as a drop-in GPU-accelerated backend for the
scikit-image Richardson–Lucy implementation.

- **Source:** [rapidsai/cucim](https://github.com/rapidsai/cucim)
- **2-D support:** ✅ Yes
- **GPU:** CUDA (required)
- **Platform note:** Linux only — cuCIM does not support Windows.

---

## Platform notes — Linux vs Windows vs WSL2

| Feature | Linux | Windows (native) | WSL2 / Docker Desktop |
|---------|-------|-------------------|------------------------|
| CUDA GPU | ✅ Full support | ✅ Full support | ✅ Full support via WSL2 CUDA driver |
| OpenCL GPU | ✅ Full support (pocl / NVIDIA ICD) | ✅ Full support (NVIDIA ICD) | ❌ **Not available** — WSL2 has no OpenCL GPU passthrough; falls back to CPU |
| deconwolf | ✅ OpenCL GPU | ✅ OpenCL GPU ([binaries](https://github.com/elgw/deconwolf/releases)) | ✅ CPU only (no OpenCL GPU in WSL2) |
| cuCIM (`skimage_cucim_rl`) | ✅ Full support | ❌ Not available | ✅ CUDA GPU via Linux container |
| pycudadecon | ✅ conda-forge | ✅ conda-forge | ✅ conda-forge (Linux env) |
| RedLionfish | ✅ OpenCL GPU | ✅ OpenCL GPU | ✅ CPU only (no OpenCL GPU in WSL2) |
| DeconvolutionLab2 | ✅ Java 17 | ✅ Java 17 | ✅ Java 17 |

**Recommendation:** For full method coverage **with GPU acceleration**,
run CIDeconvolve on **Linux** (native or Docker).  On Windows, native
installation gives OpenCL GPU support for deconwolf and RedLionfish;
WSL2 / Docker Desktop gives CUDA GPU support for all CUDA methods but
OpenCL methods lose GPU acceleration.  See the
[NVIDIA Container Toolkit docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
for Docker GPU passthrough setup.

---

## References

1. Richardson, W. H. (1972). "Bayesian-Based Iterative Method of Image Restoration." *JOSA* **62**(1), 55–59. [doi:10.1364/JOSA.62.000055](https://doi.org/10.1364/JOSA.62.000055)
2. Lucy, L. B. (1974). "An iterative technique for the rectification of observed distributions." *AJ* **79**(6), 745. [doi:10.1086/111605](https://doi.org/10.1086/111605)
3. Sage, D. et al. (2017). "DeconvolutionLab2: An open-source software for deconvolution microscopy." *Methods* **115**, 28–41. [doi:10.1016/j.ymeth.2016.12.015](https://doi.org/10.1016/j.ymeth.2016.12.015)
4. Descloux, A. et al. (2019). "Combined multi-plane phase retrieval and super-resolution…" *Nat Methods* **16**, 918–924. [doi:10.1038/s41592-019-0515-9](https://doi.org/10.1038/s41592-019-0515-9)
5. Chen, B.-C. et al. (2014). "Lattice light-sheet microscopy." *Science* **346**(6208), 1257998. [doi:10.1126/science.1257998](https://doi.org/10.1126/science.1257998)
6. Wernersson, E. et al. (2024). "Deconwolf enables high-performance deconvolution of widefield fluorescence microscopy images." *Nat Methods*. [doi:10.1038/s41592-024-02294-7](https://doi.org/10.1038/s41592-024-02294-7)
7. van der Walt, S. et al. (2014). "scikit-image: image processing in Python." *PeerJ* **2**, e453. [doi:10.7717/peerj.453](https://doi.org/10.7717/peerj.453)
8. Gibson, S. F. & Lanni, F. (1992). "Experimental test of an analytical model of aberration in an oil-immersion objective lens…" *JOSA A* **9**(1), 154–166. [doi:10.1364/JOSAA.9.000154](https://doi.org/10.1364/JOSAA.9.000154)

---

*See also:* [README.md](README.md) · [BENCHMARK.md](BENCHMARK.md)
