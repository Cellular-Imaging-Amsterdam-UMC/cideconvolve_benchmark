# DECONVOLVE_CI — GPU-Accelerated Richardson-Lucy Deconvolution & PSF Generation

## Overview

`deconvolve_ci.py` is a PyTorch GPU-accelerated deconvolution and PSF generation
module for CIDeconvolve. It combines the best mathematical ideas from multiple state-of-the-art
implementations into a single, cohesive, fast engine.

**Methods**: `ci_rl` (accelerated RL) and `ci_rl_tv` (accelerated RL with Total Variation)

---

## Methods

### `ci_rl` — Scaled Heavy Ball Accelerated Richardson-Lucy

Standard Richardson-Lucy deconvolution enhanced with **Scaled Heavy Ball (SHB) momentum
acceleration** ([Wang & Miller 2014](#ref3)). The SHB technique applies Nesterov-style
momentum to the RL multiplicative update, achieving 5-10× faster convergence at no extra
per-iteration cost.

**Per-iteration update:**

1. **Momentum extrapolation**:
   ```
   p_k = x_k + α_k · (x_k − x_{k−1})
   ```
   where α_k = min((k−1)/(k+2), 1 − 2/√(k+3))

2. **Forward model**: y = IFFT(FFT(p_k) · H)

3. **Ratio**: r = d / max(y, ε)

4. **Multiplicative correction**: x_{k+1} = p_k · IFFT(FFT(r) · H̄) · W

5. **Positivity**: x_{k+1} = max(x_{k+1}, background)

Where:
- H = OTF (Optical Transfer Function, FFT of PSF)
- H̄ = conjugate of OTF (adjoint/transpose)
- W = Bertero boundary correction weights
- d = observed (noisy) image
- ε = small constant (background level or 1e-6)

**Cost**: 4 FFTs per iteration (identical to vanilla RL), but significantly fewer iterations needed.

### `ci_rl_tv` — SHB-RL with Total Variation Regularization

Same as `ci_rl` but with an additional **Total Variation (TV) penalty** applied after each
RL update. TV regularization suppresses noise amplification (a well-known RL pathology at
high iteration counts) while preserving edges.

**TV factor computation** (following DeconvolutionLab2 RLTV, [Dey et al. 2006](#ref5)):

1. Forward differences: ∇x, ∇y, ∇z
2. Gradient magnitude: |∇| = √(gx² + gy² + gz² + ε)
3. Normalize: ĝ = g / |∇|
4. Divergence (backward differences): div(ĝ)
5. TV factor: tv_i = 1 / (λ · div_i + 1)

Applied multiplicatively: x_{k+1} = x_{k+1} · tv(x_k)

---

### 1. Scaled Heavy Ball Momentum Acceleration
**Source**: deconwolf [`method_shb.c`](https://github.com/elgw/deconwolf/blob/master/src/method_shb.c) ([Wernersson et al. 2024](#ref6)), based on
[Wang & Miller 2014](#ref3).

**What it does**: Adds a momentum term to RL ([Richardson 1972](#ref1); [Lucy 1974](#ref2)) that automatically accelerates convergence.
The acceleration factor α_k starts near 0 and grows toward 1, with an upper bound that
guarantees convergence. This is the same mathematical framework as Nesterov's accelerated
gradient descent, adapted for the multiplicative RL update.

**Improvement**: 5-10× fewer iterations vs. vanilla RL for equivalent quality. A 50-iteration
SHB-RL typically matches 200-500 iterations of standard RL.

### 2. Bertero Boundary Weights
**Source**: deconwolf [`method_shb.c`](https://github.com/elgw/deconwolf/blob/master/src/method_shb.c), based on [Bertero & Boccacci 2005](#ref4).

**What it does**: Instead of reflection padding (used by sdeconv) or zero-padding with
no correction (DeconvolutionLab2), Bertero weights correct for the incomplete PSF energy
at image boundaries. Computed as:

```
W = 1 / (IFFT(FFT(1_padded) · |H|²))
```

where values below threshold σ are zeroed.

**Improvement**: Eliminates the bright/dark edge artifacts common in standard RL. More
principled than reflection padding, which can introduce gradient discontinuities.

### 3. I-Divergence Convergence Monitoring
**Source**: deconwolf, based on [Csiszár 1991](#ref10).

**What it does**: Monitors the I-divergence (generalized Kullback-Leibler divergence):

```
I_div = (1/N) Σ [d_i · ln(d_i / ŷ_i) − d_i + ŷ_i]
```

This is the natural loss function for Poisson noise (the RL algorithm is maximum likelihood
for Poisson statistics). When I-divergence plateaus, further iterations are amplifying noise.

**Modes**:
- `convergence="fixed"`: Run exactly `niter` iterations (default)
- `convergence="auto"`: Stop when relative I-divergence change < `rel_threshold`

### 4. Total Variation Regularization
**Source**: DeconvolutionLab2 [`RichardsonLucyTV.java`](https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2/blob/master/src/main/java/deconvolution/algorithm/RichardsonLucyTV.java), [Dey et al. 2006](#ref5).

**What it does**: Applies a TV penalty as a multiplicative correction after each RL step.
TV penalizes the total gradient magnitude, promoting piecewise-constant reconstructions
that suppress noise while preserving sharp edges.

**Parameter**: `tv_lambda` (default 0, i.e. off). Typical values: 0.001–0.01.

### 5. Automatic Background Estimation
**What it does**: Estimates the image background level from the mode of the lowest 10%
of voxel intensities. This is used as:
- The positivity floor (clamp values ≥ background instead of ≥ 0)
- The safe-division epsilon

Can be overridden with `background=<float>` or disabled with `background=0`.

### 6. Real-Valued FFT (Memory Optimization)
**What it does**: Uses `torch.fft.rfftn` / `irfftn` throughout, which exploits the
Hermitian symmetry of real-valued data to store only half the complex spectrum. This
halves the memory cost of all frequency-domain buffers compared to sdeconv's `torch.fft.fftn`.

### 7. Noise-Gated Damping
**Source**: [Laasmaa et al. 2011](#ref11); [Conchello 1998](#ref12).

**What it does**: Attenuates the RL update in background (noisy) regions while leaving
bright structures untouched. The multiplicative correction factor is raised to a per-voxel
power γ ∈ (0, 1]:

```
correction_damped = correction ^ γ(x)
γ(x) = 1 − exp(−(x − bg) / (damping · σ))
```

Where:
- `x` is the current estimate at each voxel
- `bg` is the estimated background level
- `σ` is the noise standard deviation (estimated via MAD of the lowest 25% of voxels)
- `damping` is the user-controlled strength parameter

Pixels near the background floor are heavily damped (γ → 0, correction → 1, i.e. no
change) while bright signal regions get the full RL update (γ → 1). The noise σ is
estimated once using a robust MAD estimator: σ ≈ 1.4826 · median(|x − median(x)|)
computed on the lowest quartile of voxels.

**Modes**:
- `damping=0` — disabled (default)
- `damping="auto"` — automatic strength of 3.0
- `damping=<float>` — manual strength (typical range: 1.0–10.0)

### 8. XY Tiling (Large Image Support)
**What it does**: Splits large images into overlapping XY tiles that are deconvolved
independently and stitched back together. This allows processing images that are too
large to fit in GPU memory as a single volume.

**How it works**:
1. The image is divided into a grid of tiles (`ny × nx`) in the YX plane
2. Each tile is extracted with an overlap margin (default 16 pixels) on each side
3. Each tile is deconvolved independently (including PSF generation, background
   estimation, and full RL iteration)
4. Results are stitched by taking only the core (non-overlap) region of each tile

The overlap margin prevents boundary artefacts at tile edges. The tile grid is chosen
to minimise the aspect ratio of individual tiles.

**Modes**:
- `tiling="none"` — no tiling, process the full image at once
- `tiling="custom"` — automatic tiling based on `max_tile_xy` and `max_tile_z`
  (default: 512 × 512 × 64)

**Parameters**:
- `max_tile_xy` (int, default 512): Maximum tile size in the XY dimensions
- `max_tile_z` (int, default 64): Maximum tile size in the Z dimension

---

## PSF Generation

### `ci_generate_psf` — Vectorial/Scalar PSF with Gibson-Lanni Correction

Generates a physically accurate Point Spread Function using either the **vectorial
Richards-Wolf** model ([Richards & Wolf 1959](#ref7), for high-NA objectives, NA ≥ 0.9) or the **scalar Kirchhoff**
model (for lower NA), with **Gibson-Lanni refractive index mismatch correction** ([Gibson & Lanni 1991](#ref8)).

### PSF Models

**Vectorial Richards-Wolf** (high NA):
```
I₀(ρ,z) = ∫₀^θmax √cosθ · sinθ · (1+cosθ) · J₀(kρsinθ) · exp(ikz·cosθ) · exp(ikW(θ)) dθ
I₁(ρ,z) = ∫₀^θmax √cosθ · sin²θ · J₁(kρsinθ) · exp(ikz·cosθ) · exp(ikW(θ)) dθ
I₂(ρ,z) = ∫₀^θmax √cosθ · sinθ · (1−cosθ) · J₂(kρsinθ) · exp(ikz·cosθ) · exp(ikW(θ)) dθ

PSF(ρ,z) = |I₀|² + 2|I₁|² + |I₂|²
```

**Scalar Kirchhoff** (lower NA):
```
I(ρ,z) = ∫₀^θmax √cosθ · sinθ · J₀(kρsinθ) · exp(ikz·cosθ) · exp(ikW(θ)) dθ

PSF(ρ,z) = |I|²
```

### Gibson-Lanni Aberration
Optical path difference accounting for refractive index mismatches ([Gibson & Lanni 1991](#ref8); [Aguet 2009](#ref9)):
```
W(θ) = z_p · √(n_s² − NA²sin²θ)
      + t_i · √(n_i² − NA²sin²θ)
      − t_i0 · √(n_i0² − NA²sin²θ)
      + t_g · √(n_g² − NA²sin²θ)
      − t_g0 · √(n_g0² − NA²sin²θ)
```

Parameters: sample RI (n_s), immersion RI (n_i), coverslip RI (n_g), and design values.

### Pixel Integration
Instead of sampling the PSF only at pixel centers, integrates over the pixel area by
evaluating on an N×N sub-grid (default N=3) and averaging. This improves accuracy when
the pixel size is not much smaller than the diffraction limit.

Controllable via `integrate_pixels=True` (default) or `False` for speed.

### Confocal PSF
For confocal microscopy:
```
PSF_confocal = PSF_emission × PSF_excitation
```
With separate wavelengths for excitation and emission. Falls back to PSF² when
excitation wavelength is not provided.

### Numerical Integration
Simpson's rule quadrature with configurable number of pupil samples (default 129).

---

---

## Design Choices

| Choice | Decision | Rationale |
|--------|----------|-----------|
| Framework | PyTorch | Native CUDA GPU support, automatic differentiation available, widely installed in scientific Python |
| FFT type | `rfftn`/`irfftn` | 2× memory savings over full complex FFT for real data |
| Padding | Full (image+PSF-1) | Best quality (Bertero bq=2 equivalent); combined with boundary weights eliminates need for reflection padding |
| Acceleration | SHB momentum | Proven convergence guarantees ([Wang 2014](#ref3)), no extra FFTs per iteration, 5-10× speedup |
| Boundary handling | Bertero weights | Mathematically rigorous, superior to reflection/zero padding |
| Convergence metric | I-divergence | Natural for Poisson noise model underlying RL |
| Regularization | TV (optional) | Well-understood, preserves edges, single parameter |
| PSF model | Vectorial + Scalar | Auto-select based on NA; vectorial essential for high-NA (immersion) objectives |
| PSF Aberration | Gibson-Lanni | Accounts for real-world RI mismatches; critical for accurate PSF at depth |
| Pixel integration | Sub-pixel grid | Measurable accuracy gain, optional for speed |
| Background | Auto-estimate | Mode of lowest 10% voxels — robust to outliers and signal |
| Damping | Noise-gated (optional) | Per-voxel γ attenuates background noise; single parameter, auto mode available |
| Tiling | XY overlap-stitch | Process arbitrarily large images within GPU memory; 16 px overlap removes boundary artefacts |
| dtype | float32 (GPU) / float64 (CPU) | Optimal throughput on GPU; CPU can afford double precision |

---

## API Reference

### `ci_rl_deconvolve(image, psf, **kwargs) → dict`

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | ndarray | required | Input image (2D or 3D, any integer or float dtype) |
| `psf` | ndarray | required | Point Spread Function (same dimensionality as image) |
| `niter` | int | 50 | Maximum number of iterations |
| `tv_lambda` | float | 0.0 | TV regularization strength (0 = disabled) |
| `damping` | str/float | 0.0 | Noise-gated damping: `0` (disabled), `"auto"` (strength 3.0), or float > 0 |
| `background` | str/float | "auto" | Background level: "auto", float, or 0 |
| `convergence` | str | "auto" | "fixed" (exact niter) or "auto" (I-divergence stop) |
| `rel_threshold` | float | 0.005 | Relative I-divergence change threshold for auto-stop |
| `check_every` | int | 5 | Check convergence every N iterations |
| `device` | str/None | None | PyTorch device ("cuda", "cpu", or None=auto) |
| `tiling` | str | "custom" | Tiling mode: "none" or "custom" (auto tile to fit GPU) |
| `max_tile_xy` | int | 512 | Maximum tile size in XY (pixels) when tiling="custom" |
| `max_tile_z` | int | 64 | Maximum tile size in Z (slices) when tiling="custom" |

**Returns**: dict with keys:
- `"result"` — deconvolved image as numpy ndarray (same shape as input)
- `"convergence"` — list of I-divergence values at check points
- `"iterations_used"` — actual number of iterations performed

### `ci_generate_psf(na, wavelength_nm, ...) → ndarray`

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `na` | float | required | Numerical aperture |
| `wavelength_nm` | float | required | Emission wavelength in nanometers |
| `pixel_size_xy_nm` | float | required | Lateral pixel size in nm |
| `pixel_size_z_nm` | float | required | Axial pixel size in nm |
| `n_xy` | int | required | Number of lateral pixels (should be odd) |
| `n_z` | int | required | Number of axial planes (should be odd) |
| `ri_immersion` | float | 1.515 | Immersion medium refractive index |
| `ri_sample` | float | 1.33 | Sample refractive index |
| `ri_coverslip` | float | 1.5 | Coverslip refractive index |
| `microscope_type` | str | "widefield" | "widefield" or "confocal" |
| `excitation_nm` | float/None | None | Excitation wavelength (confocal) |
| `integrate_pixels` | bool | True | Enable sub-pixel integration |
| `n_subpixels` | int | 3 | Sub-pixel grid size per dimension |
| `n_pupil` | int | 129 | Number of pupil integration samples |
| `device` | str/None | None | PyTorch device |

**Returns**: ndarray — normalized PSF (sum = 1.0)

---

## References

1. <a id="ref1"></a>Richardson, W.H. (1972). "Bayesian-based iterative method of image restoration." *JOSA* 62(1):55–59.\
   [doi:10.1364/JOSA.62.000055](https://doi.org/10.1364/JOSA.62.000055)

2. <a id="ref2"></a>Lucy, L.B. (1974). "An iterative technique for the rectification of observed distributions." *AJ* 79:745.\
   [doi:10.1086/111605](https://doi.org/10.1086/111605)

3. <a id="ref3"></a>Wang, C. & Miller, E.L. (2014). "Scaled Heavy-Ball Acceleration of the Richardson-Lucy Algorithm for 3D Microscopy Image Restoration." *IEEE TIP* 23(2):848–854.\
   [doi:10.1109/TIP.2013.2291324](https://doi.org/10.1109/TIP.2013.2291324)

4. <a id="ref4"></a>Bertero, M. & Boccacci, P. (2005). "A simple method for the reduction of boundary effects in the Richardson-Lucy approach to image deconvolution." *A&A* 437:369–374.\
   [doi:10.1051/0004-6361:20052717](https://doi.org/10.1051/0004-6361:20052717)

5. <a id="ref5"></a>Dey, N. et al. (2006). "Richardson-Lucy algorithm with total variation regularization for 3D confocal microscope deconvolution." *Microscopy Research and Technique* 69(4):260–266.\
   [doi:10.1002/jemt.20294](https://doi.org/10.1002/jemt.20294)

6. <a id="ref6"></a>Wernersson, E. et al. (2024). "Deconwolf enables high-performance deconvolution of widefield fluorescence microscopy images." *Nature Methods* 21(7):1245–1256.\
   [doi:10.1038/s41592-024-02294-7](https://doi.org/10.1038/s41592-024-02294-7)

7. <a id="ref7"></a>Richards, B. & Wolf, E. (1959). "Electromagnetic diffraction in optical systems, II." *Proc. Royal Society A* 253:358–379.\
   [doi:10.1098/rspa.1959.0200](https://doi.org/10.1098/rspa.1959.0200)

8. <a id="ref8"></a>Gibson, S.F. & Lanni, F. (1991). "Experimental test of an analytical model of aberration in an oil-immersion objective lens used in three-dimensional light microscopy." *JOSA A* 8(10):1601–1613.\
   [doi:10.1364/JOSAA.8.001601](https://doi.org/10.1364/JOSAA.8.001601)

9. <a id="ref9"></a>Aguet, F. (2009). "Super-Resolution Fluorescence Microscopy Based on Physical Models." *EPFL Thesis* no. 4418.\
   [doi:10.5075/epfl-thesis-4418](https://doi.org/10.5075/epfl-thesis-4418)

10. <a id="ref10"></a>Csiszár, I. (1991). "Why least squares and maximum entropy? An axiomatic approach to inference for linear inverse problems." *Annals of Statistics* 19(4):2032–2066.\
    [doi:10.1214/aos/1176348385](https://doi.org/10.1214/aos/1176348385)

11. <a id="ref11"></a>Laasmaa, M. et al. (2011). "Application of regularized Richardson-Lucy algorithm for deconvolution of confocal microscopy images." *Journal of Microscopy* 243(2):124–140.\
    [doi:10.1111/j.1365-2818.2011.03486.x](https://doi.org/10.1111/j.1365-2818.2011.03486.x)

12. <a id="ref12"></a>Conchello, J.-A. (1998). "Superresolution and convergence properties of the expectation-maximization algorithm for maximum-likelihood deconvolution of incoherent images." *JOSA A* 15(10):2609–2619.\
    [doi:10.1364/JOSAA.15.002609](https://doi.org/10.1364/JOSAA.15.002609)

13. <a id="ref13"></a>Sage, D., Donati, L., Soulez, F., Fortun, D., Schmit, G., Seitz, A., Guiet, R., Vonesch, C. & Unser, M. (2017). "DeconvolutionLab2: An Open-Source Software for Deconvolution Microscopy." *Methods* 115:28–41.\
    [doi:10.1016/j.ymeth.2016.12.015](https://doi.org/10.1016/j.ymeth.2016.12.015)

14. <a id="ref14"></a>Sibarita, J.-B. (2005). "Deconvolution Microscopy." *Advances in Biochemical Engineering/Biotechnology* 95:201–243.\
    [doi:10.1007/b102215](https://doi.org/10.1007/b102215)

15. <a id="ref15"></a>van Kempen, G.M.P., van Vliet, L.J., Verveer, P.J. & van der Voort, H.T.M. (1997). "A quantitative comparison of image restoration methods for confocal microscopy." *Journal of Microscopy* 185(3):354–365.\
    [doi:10.1046/j.1365-2818.1997.d01-629.x](https://doi.org/10.1046/j.1365-2818.1997.d01-629.x)

16. <a id="ref16"></a>Prigent, S., Nguyen, H.N., Leconte, L. et al. (2023). "SPITFIR(e): a supermaneuverable algorithm for fast denoising and deconvolution of 3D fluorescence microscopy images and videos." *Scientific Reports* 13:1489.\
    [doi:10.1038/s41598-022-26178-y](https://doi.org/10.1038/s41598-022-26178-y)

17. <a id="ref17"></a>McNally, J.G., Karpova, T., Cooper, J. & Conchello, J.A. (1999). "Three-Dimensional Imaging by Deconvolution Microscopy." *Methods* 19(3):373–385.\
    [doi:10.1006/meth.1999.0873](https://doi.org/10.1006/meth.1999.0873)

18. <a id="ref18"></a>Wallace, W., Schaefer, L.H. & Swedlow, J.R. (2001). "A workingperson's guide to deconvolution in light microscopy." *BioTechniques* 31(5):1076–1097.\
    [doi:10.2144/01315bi01](https://doi.org/10.2144/01315bi01)

19. <a id="ref19"></a>Sarder, P. & Nehorai, A. (2006). "Deconvolution methods for 3-D fluorescence microscopy images." *IEEE Signal Processing Magazine* 23(3):32–45.\
    [doi:10.1109/MSP.2006.1628876](https://doi.org/10.1109/MSP.2006.1628876)

20. <a id="ref20"></a>Bertero, M., Boccacci, P., Desiderà, G. & Vicidomini, G. (2009). "Image deblurring with Poisson data: from cells to galaxies." *Inverse Problems* 25(12):123006.\
    [doi:10.1088/0266-5611/25/12/123006](https://doi.org/10.1088/0266-5611/25/12/123006)

21. <a id="ref21"></a>Kirshner, H., Aguet, F., Sage, D. & Unser, M. (2013). "3-D PSF fitting for fluorescence microscopy: implementation and localization application." *Journal of Microscopy* 249(1):13–25.\
    [doi:10.1111/j.1365-2818.2012.03675.x](https://doi.org/10.1111/j.1365-2818.2012.03675.x)

22. <a id="ref22"></a>Vonesch, C. & Unser, M. (2008). "A fast thresholded Landweber algorithm for wavelet-regularized multidimensional deconvolution." *IEEE Transactions on Image Processing* 17(4):539–549.\
    [doi:10.1109/TIP.2008.917106](https://doi.org/10.1109/TIP.2008.917106)

### Source Code Repositories

| Project | Repository |
|---------|------------|
| deconwolf | [github.com/elgw/deconwolf](https://github.com/elgw/deconwolf) |
| DeconvolutionLab2 | [github.com/Biomedical-Imaging-Group/DeconvolutionLab2](https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2) |
| sdeconv | [github.com/sylvainprigent/sdeconv](https://github.com/sylvainprigent/sdeconv) |

---

## File Layout

```
deconvolve_ci.py
├── ci_generate_psf()              — public PSF generation
│   ├── _scalar_psf()              — Kirchhoff model (NA < 0.9)
│   ├── _vectorial_psf()           — Richards-Wolf model (NA ≥ 0.9)
│   ├── _gibson_lanni_opd()        — aberration phase
│   ├── _simpson_integrate()       — numerical quadrature
│   └── _pixel_integrate()         — sub-pixel averaging
│
├── ci_rl_deconvolve()             — public deconvolution
│   ├── _prepare_otf()             — PSF → OTF with padding
│   ├── _bertero_weights()         — boundary correction
│   ├── _estimate_noise_sigma()    — MAD-based noise σ (lowest 25%)
│   ├── _damping_map()             — per-voxel γ for noise-gated damping
│   ├── _resolve_tiling()          — tiling mode → tile count
│   ├── _ci_deconvolve_tiled()     — tiled deconvolution dispatcher
│   │   ├── _compute_tile_grid()   — optimal ny×nx layout
│   │   └── _compute_tile_slices() — tile extents with overlap
│   ├── _shb_rl_loop()            — main iteration loop
│   │   ├── _rl_update()          — single RL step on momentum point
│   │   └── _tv_penalty()         — optional TV correction
│   ├── _i_divergence()           — convergence metric
│   └── _estimate_background()    — auto background
│
└── _to_device() / _to_numpy()    — device transfer helpers
```
