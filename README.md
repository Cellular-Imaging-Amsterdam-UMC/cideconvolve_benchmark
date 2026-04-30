# CIDeconvolve Benchmark

Benchmark and test workflow for microscopy deconvolution methods.

This repository wraps the CIDeconvolve deconvolution engine in a
BIAFLOWS/BIOMERO-compatible workflow and adds tooling for comparing multiple
deconvolution backends on the same input image. It can run as a Docker
container, through the PyQt6 launcher, or directly from `wrapper.py` in a local
Python environment.

The core CI deconvolution and PSF code is kept in [deconvolve_ci.py](deconvolve_ci.py)
and is synced from the companion
[cideconvolve](https://github.com/Cellular-Imaging-Amsterdam-UMC/cideconvolve)
repository when needed. This benchmark repository adds descriptor handling,
Docker/BIAFLOWS integration, benchmark CSV/provenance output, and local test
image generation.

| | |
|---|---|
| Workflow name | `W_CIDeconvolve_benchmark` |
| Docker image | `cellularimagingcf/w_cideconvolve_benchmark` |
| Version | `v1.0.0` |
| Single-method selector | 16 methods in `descriptor.json` |
| Benchmark sets | include `ci_sparse_hessian` in addition to the public selector methods |
| Input format | OME-TIFF / OME-Zarr via `bioio`, with TIFF fallback |

## What It Produces

In normal mode, the workflow writes deconvolved OME-TIFF results to the output
folder. In benchmark mode, it writes:

- `benchmark_metrics_<image>.csv`: one row per method/iteration attempt
- `benchmark_provenance_<image>.json`: platform, version, PSF, metadata, and CLI provenance
- `decon_benchmark_<image>.png`: optional montage for visual comparison
- deconvolved OME-TIFF outputs for successful method runs

Benchmark rows include timing, CPU/RAM/GPU usage, status/skip reason, tiling
mode, crop mode, and no-reference image metrics.

## Benchmark Metrics

The benchmark currently uses no-reference image metrics:

- `detail_energy_mean`: high-frequency FFT power after normalization
- `bright_detail_energy_mean`: high-frequency detail within bright structures
- `edge_strength_mean`: mean normalized gradient magnitude
- `signal_sparsity_mean`: Gini-style concentration of signal intensity
- `robust_range_mean`: 99.5th minus 0.5th percentile after normalization

These metrics are useful for comparing restoration behavior, but they are not
ground-truth accuracy scores. Interpret them together with the montage images
and the source data.

## Metadata And PSF Behavior

The workflow generates PSFs from image metadata where possible:

- pixel size XY/Z
- numerical aperture
- immersion refractive index
- sample/mounting refractive index
- microscope type (`widefield` or `confocal`)
- excitation and emission wavelengths
- confocal pinhole size

The parameter `overrule_image_metadata` controls how descriptor values are used:

- `False` (default): image metadata wins, descriptor values fill missing fields
- `True`: descriptor values replace image metadata

For confocal PSFs, the current `deconvolve_ci.py` uses finite pinhole size in
the PSF calculation. The descriptor parameter `pinhole_size` is in Airy disk
units. OME channel `PinholeSize` is stored/read in micrometers and converted to
Airy units using emission wavelength, NA, and objective magnification.

## Synthetic 3D Test Images

`create3d_gt.py` creates a deterministic synthetic 3D GFP-like cell volume and
a blurred/noisy image generated from it:

- `synthetic_object_gt.ome.tiff`
- `synthetic_blurred_noisy_snr10.ome.tiff`

The generated OME-TIFF files include structured OME metadata for pixel size,
objective NA and magnification, immersion RI, sample RI, microscope mode,
wavelengths, and pinhole size.

Example:

```bash
python create3d_gt.py --no-gui --output synthetic --z 64 --yx 256
```

Useful options:

```bash
python create3d_gt.py --no-gui \
  --pixel-size-xy-nm 40 \
  --pixel-size-z-nm 100 \
  --na 1.4 \
  --magnification 63 \
  --immersion-ri 1.518 \
  --sample-ri 1.47 \
  --microscope-type confocal \
  --excitation-nm 488 \
  --emission-nm 520 \
  --pinhole-size-airy 1.0 \
  --snr 10
```

Run without `--no-gui` to open the PyQt6 generator interface.

## Local Launcher

`launcher.py` reads `descriptor.json` and builds a PyQt6 GUI for running the
Docker container. Parameters are grouped like the main cideconvolve launcher:
essential parameters first and advanced parameters in a collapsible section.
The parameter grid uses two columns.

```bash
python launcher.py
```

The launcher mounts local folders:

- `infolder` -> `/data/in`
- `outfolder` -> `/data/out`
- `gtfolder` -> `/data/gt`

## Build Docker Image

On Windows:

```bat
builddocker.cmd
```

This reads the local image name from `descriptor.json` and the tag from
`version.txt`, then builds:

- `w_cideconvolve_benchmark:v1.0.0`
- `w_cideconvolve_benchmark:latest`

Equivalent manual command:

```bash
docker build -t w_cideconvolve_benchmark:v1.0.0 -t w_cideconvolve_benchmark:latest .
```

The Docker image includes Python 3.11, CUDA runtime support, pycudadecon,
deconwolf, DeconvolutionLab2, Java, vendored `sdeconv`, and the benchmark
workflow code.

## Run With Docker

On Windows:

```bat
run.cmd --benchmark True
```

Manual Docker example:

```bash
docker run --rm --gpus all \
  -v /path/to/input:/data/in \
  -v /path/to/output:/data/out \
  -v /path/to/gt:/data/gt \
  w_cideconvolve_benchmark:latest \
  --infolder /data/in \
  --outfolder /data/out \
  --gtfolder /data/gt \
  --local \
  --benchmark True \
  --bench_iterations "20, 40, 60"
```

For a single deconvolution method:

```bash
docker run --rm --gpus all \
  -v /path/to/input:/data/in \
  -v /path/to/output:/data/out \
  -v /path/to/gt:/data/gt \
  w_cideconvolve_benchmark:latest \
  --infolder /data/in \
  --outfolder /data/out \
  --gtfolder /data/gt \
  --local \
  --benchmark False \
  --method sdeconv_rl \
  --iterations 40
```

## Run Directly In Python

Example:

```bash
python wrapper.py \
  --infolder ./infolder \
  --outfolder ./outfolder \
  --gtfolder ./gtfolder \
  --local \
  --benchmark True \
  --bench_iterations "20, 40, 60"
```

For local development, use a Python environment with the dependencies from
`requirements_docker.txt` installed.

## Parameters

The user-facing parameters are defined in [descriptor.json](descriptor.json).

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--method` | `sdeconv_rl` | Single deconvolution backend |
| `--iterations` | `40` | Iteration count for iterative methods |
| `--overrule_image_metadata` | `False` | If false, descriptor metadata fills only missing image metadata |
| `--na` | `1.4` | Objective NA fallback/override |
| `--emission_wl` | `520` | Emission wavelength(s), nm |
| `--pixel_size_xy` | `65.0` | XY pixel size, nm |
| `--pixel_size_z` | `200.0` | Z step, nm |
| `--microscope_type` | `confocal` | `widefield` or `confocal` |
| `--excitation_wl` | `488` | Excitation wavelength(s), nm |
| `--pinhole_size` | `1.0` | Confocal pinhole in Airy disk units |
| `--refractive_index` | `oil (1.515)` | Immersion medium |
| `--sample_ri` | `prolong gold (1.47)` | Sample/mounting medium |
| `--projection` | `none` | `none`, `mip`, or `sum` |
| `--benchmark` | `True` | Run benchmark instead of one method |
| `--bench_iterations` | `20, 40, 60` | Iterations used in benchmark mode |
| `--bench_methods` | base CI set | Preset or comma-separated method list |
| `--bench_crop` | `True` | Center-crop before benchmarking |
| `--bench_one_image` | `True` | Benchmark only first input image |
| `--bench_montage` | `True` | Write PNG comparison montages |
| `--tiling` | `custom` | Advanced; `none` or `custom` |
| `--tile_limits` | `512, 64` | Advanced; max XY and Z tile size |
| `--device` | `auto` | Advanced; `auto`, `cpu`, `cuda` |
| `--log` | `False` | Advanced; save console log |

## Methods

See [METHODS.md](METHODS.md) for method details and platform notes.

Single-method choices in `descriptor.json` currently include:

- `sdeconv_rl`, `sdeconv_wiener`, `sdeconv_spitfire`
- `ci_rl`, `ci_rl_tv`
- `pycudadecon_rl_cuda`
- `deconwolf_rl`, `deconwolf_shb`
- `deconvlab2_rl`, `deconvlab2_rltv`, `deconvlab2_landweber`, `deconvlab2_ista`
- `redlionfish_rl`
- `skimage_rl`, `skimage_unsupervised_wiener`, `skimage_cucim_rl`

Benchmark presets also include `ci_sparse_hessian`.

## Project Structure

```text
wrapper.py               BIAFLOWS entrypoint and benchmark runner
deconvolve.py            Metadata parsing, PSF generation, method dispatch, saving
deconvolve_ci.py         CI deconvolution and finite-pinhole PSF implementation
create3d_gt.py           Local synthetic 3D ground-truth generator
launcher.py              PyQt6 Docker launcher
descriptor.json          BIAFLOWS/BIOMERO parameter descriptor
bioflows_local.py        Local BIAFLOWS compatibility shim
Dockerfile               GPU-capable benchmark container
requirements_docker.txt  Docker Python dependencies
vendor/                  Vendored Python libraries
```

## History Note

The generic README was not introduced by the current working tree. In local git
history, `README.md` was first added in commit `d4e57d6` on 2026-03-31 with
generic CIDeconvolve wording. Later commits adjusted method counts and benchmark
details. No older benchmark-specific README exists in the local branches or
tags, so this file has been reconstructed for the benchmark repository.

## References

- [CIDeconvolve](https://github.com/Cellular-Imaging-Amsterdam-UMC/cideconvolve)
- [BIOMERO](https://github.com/NL-BioImaging/biomero)
- [BIAFLOWS](https://biaflows.neubias.org/)
- [deconwolf](https://github.com/elgw/deconwolf)
- [pycudadecon](https://github.com/tlambert03/pycudadecon)
- [RedLionfish](https://github.com/rosalindfranklininstitute/RedLionfish)
- [DeconvolutionLab2](http://bigwww.epfl.ch/deconvolution/deconvolutionlab2/)
