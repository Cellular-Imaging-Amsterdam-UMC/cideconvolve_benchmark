# CIDeconvolve

**3-D / 2-D microscopy deconvolution with 11 algorithm backends.**

CIDeconvolve is a [BIAFLOWS](https://biaflows.neubias.org/)-compatible
workflow that deconvolves widefield and confocal fluorescence microscopy
images.  It generates a theoretically correct PSF from OME-TIFF metadata
and applies a user-selected deconvolution method — ranging from fast CUDA
GPU implementations to portable CPU-only fall-backs.

| | |
|---|---|
| **Docker image** | `cellularimagingcf/w_cideconvolve` |
| **Version** | v0.1.0 |
| **Container type** | Singularity (pulled from Docker Hub) |
| **Available methods** | 11 — see [METHODS.md](METHODS.md) |
| **Benchmark** | built-in multi-method benchmark — see [BENCHMARK.md](BENCHMARK.md) |

---

## Using CIDeconvolve with BIOMERO

[BIOMERO](https://github.com/NL-BioImaging/biomero) (BioImage Analysis in
OMERO) lets you run FAIR bioimage-analysis workflows from an OMERO server
on a SLURM-based HPC cluster.  CIDeconvolve is designed to plug directly
into this framework.

### How it works

1. The OMERO admin configures the workflow in
   **`slurm-config.ini`** on the SLURM submission host by adding a section
   for `W_CIDeconvolve`:

   ```ini
   [SLURM]
   # ... global SLURM settings ...

   [W_CIDeconvolve]
   # Override default SLURM resources for this workflow
   job_cpus=8
   job_memory=52G
   job_gres=gpu:2g.24gb
   ```

2. BIOMERO reads **`descriptor.json`** from the container to discover
   input parameters (method, iterations, device, PSF settings, benchmark
   options, etc.) and presents them in the OMERO web UI.

3. On submission, BIOMERO pulls the Singularity image from Docker Hub,
   transfers the selected images, and executes the workflow on the cluster.

4. Results (deconvolved images, benchmark montages, metrics CSV) are
   automatically uploaded back into OMERO.

> For full BIOMERO setup instructions see the
> [BIOMERO documentation](https://nl-bioimaging.github.io/biomero/)
> and the [NL-BIOMERO deployment repo](https://github.com/NL-BioImaging/NL-BIOMERO).

### SLURM job script

A ready-made SLURM script is provided for manual cluster submission
(outside of BIOMERO):

```bash
sbatch cideconvolve.slurm \
    --infolder /data/myimages \
    --outfolder /data/results \
    -- --method sdeconv_rl --iterations 40 --benchmark True \
       --bench_methods all
```

See `cideconvolve.slurm` for full usage and resource settings.

---

## Building the Docker image locally

```bash
docker build -t w_cideconvolve:v0.1.0 -t w_cideconvolve:latest .
```

The Dockerfile uses a multi-stage build:

1. **Builder stage** — compiles [deconwolf](https://github.com/elgw/deconwolf)
   from source with OpenCL GPU support on the NVIDIA CUDA 12.6 devel image.
2. **Runtime stage** — NVIDIA CUDA 12.6 runtime + Java 17 +
   Miniforge (conda) with `pycudadecon`, all Python dependencies, and
   the application code.

### Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
  (for GPU pass-through at runtime)
- A working `docker build` environment (Docker Desktop on Windows/macOS,
  or Docker Engine on Linux)

---

## Running locally with Docker

```bash
docker run --rm --gpus all \
    -v /path/to/input:/data/in \
    -v /path/to/output:/data/out \
    -v /tmp/gt:/data/gt \
    cellularimagingcf/w_cideconvolve \
    --infolder /data/in --outfolder /data/out --gtfolder /data/gt \
    --method sdeconv_rl --iterations 40 \
    --na auto --refractive_index auto --sample_ri auto \
    --microscope_type auto --emission_wl auto --excitation_wl auto
```

Replace paths as needed.  The `--gpus all` flag enables NVIDIA GPU
pass-through.  Omit it to force CPU-only execution.

### Benchmark mode

```bash
docker run --rm --gpus all \
    -v /path/to/input:/data/in \
    -v /path/to/output:/data/out \
    -v /tmp/gt:/data/gt \
    cellularimagingcf/w_cideconvolve \
    --infolder /data/in --outfolder /data/out --gtfolder /data/gt \
    --benchmark True --bench_methods all --bench_iterations "20, 40, 60"
```

---

## Running locally without Docker

### Requirements

- Python 3.10 or 3.11
- PyTorch with CUDA support (see `requirements.txt` for install instructions)
- Java 17+ (for DeconvolutionLab2 methods)
- Optional: pycudadecon via conda, deconwolf binary on `$PATH`

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### CLI

```bash
python wrapper.py \
    --infolder ./infolder --outfolder ./outfolder --gtfolder ./gtfolder \
    --method sdeconv_rl --iterations 40 \
    --na auto --refractive_index auto --sample_ri auto
```

### Launcher (GUI)

A PyQt6-based launcher provides a graphical interface with parameter
controls, folder pickers, and a live command preview:

```bash
pip install PyQt6
python launcher.py
```

The launcher saves your last-used settings and can restore them on next
launch via the **Restore Last Settings** button.

---

## Parameters

All parameters are defined in `descriptor.json` and exposed on the
command line via `wrapper.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iterations` | 40 | Number of RL iterations |
| `--blocks` | auto | Tile block mode (`auto`, `0`/`1` = no tiling, `>1` = explicit) |
| `--method` | sdeconv_rl | Deconvolution backend — see [METHODS.md](METHODS.md) |
| `--device` | auto | Compute device: `auto`, `cpu`, `cuda` |
| `--na` | auto | Numerical aperture override |
| `--refractive_index` | auto | Immersion medium RI (`air`, `water`, `oil`) |
| `--sample_ri` | auto | Sample/mounting medium RI (named presets available) |
| `--microscope_type` | auto | `widefield` or `confocal` |
| `--emission_wl` | auto | Emission wavelengths in nm (comma-separated) |
| `--excitation_wl` | auto | Excitation wavelengths in nm (for confocal PSF) |
| `--projection` | none | Z-projection: `none`, `mip`, `sum` |
| `--benchmark` | false | Run multi-method benchmark |
| `--bench_iterations` | 20, 40, 60 | Iteration counts for benchmark |
| `--bench_methods` | sdeconv_rl, pycudadecon_rl_cuda | Method set for benchmark |
| `--bench_crop` | true | Centre-crop images before benchmarking |

---

## Project structure

```
wrapper.py            BIAFLOWS entrypoint — parameter parsing, benchmark runner, montage
deconvolve.py         Core deconvolution engine with all 11 backends + PSF generation
launcher.py           PyQt6 GUI launcher
descriptor.json       BIAFLOWS/BIOMERO parameter descriptor
bioflows_local.py     Local BIAFLOWS compatibility shim
cideconvolve.slurm    SLURM job script for HPC execution
Dockerfile            Multi-stage Docker build
requirements.txt      Python dependencies (local install)
requirements_docker.txt  Pinned dependencies for Docker
vendor/               Vendored libraries (sdeconv, psf_generator)
docs/                 Benchmark output images and CSV
```

---

## References

- **BIOMERO:** Luik, T. T., Rosas-Bertolini, R., Reits, E. A. J., Hoebe, R. A. & Krawczyk, P. M. (2024). "BIOMERO: A scalable and extensible image analysis framework." *Patterns* **5**(8), 101024. [doi:10.1016/j.patter.2024.101024](https://doi.org/10.1016/j.patter.2024.101024) · [GitHub](https://github.com/NL-BioImaging/biomero) · [Documentation](https://nl-bioimaging.github.io/biomero/)
- **BIAFLOWS:** Rubens, U. et al. (2020). "BIAFLOWS: A Collaborative Framework to Reproducibly Deploy and Benchmark Bioimage Analysis Workflows." *Patterns* **1**(3), 100040. [doi:10.1016/j.patter.2020.100040](https://doi.org/10.1016/j.patter.2020.100040)
- **PSF Generator:** Kirshner, H. et al. — [EPFL PSF Generator](https://bigwww.epfl.ch/algorithms/psfgenerator/)
- **Gibson–Lanni model:** Gibson, S. F. & Lanni, F. (1992). [doi:10.1364/JOSAA.9.000154](https://doi.org/10.1364/JOSAA.9.000154)
- **OMERO:** Allan, C. et al. (2012). "OMERO: flexible, model-driven data management for experimental biology." *Nat Methods* **9**, 245–253. [doi:10.1038/nmeth.1896](https://doi.org/10.1038/nmeth.1896)

For method-specific references see [METHODS.md](METHODS.md).
For benchmark results and metrics see [BENCHMARK.md](BENCHMARK.md).

---

## License

See individual library licenses in `vendor/` and `gitclones/`.
