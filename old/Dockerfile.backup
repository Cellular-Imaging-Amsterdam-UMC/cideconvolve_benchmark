# ===========================================================================
# CIDeconvolve Benchmark — GPU-enabled Docker image
# ===========================================================================
# Base: NVIDIA CUDA 12.6 devel (needed for OpenCL headers to build deconwolf
#        with GPU support and for redlionfish OpenCL runtime).
# Includes: Python 3.11, Java 17 (for DeconvolutionLab2), deconwolf (built
#           from source), and all Python dependencies.
# ===========================================================================

FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG DECONWOLF_VERSION=v0.4.5

# --- System packages for building deconwolf ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
        libfftw3-dev libgsl-dev libtiff-dev libpng-dev \
        opencl-headers ocl-icd-opencl-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Build deconwolf from source ---
RUN git clone --depth 1 --branch ${DECONWOLF_VERSION} \
        https://github.com/elgw/deconwolf.git /tmp/deconwolf \
    && mkdir /tmp/deconwolf/build \
    && cd /tmp/deconwolf/build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build . --parallel "$(nproc)" \
    && cmake --install .

# ===========================================================================
# Runtime stage
# ===========================================================================
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# --- Runtime system packages ---
# pocl-opencl-icd: CPU OpenCL backend for deconwolf SHB
#   (WSL2 NVIDIA driver does not provide libnvidia-opencl.so.1)
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        libfftw3-3 libgsl27 libtiff5 libpng16-16 \
        ocl-icd-libopencl1 pocl-opencl-icd \
        wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- Register NVIDIA OpenCL ICD (driver is mounted by nvidia-container-toolkit) ---
RUN mkdir -p /etc/OpenCL/vendors \
    && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# --- Install Miniforge (conda-forge only, no Anaconda ToS) ---
RUN wget -q -O /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda clean -afy

ENV PATH="/opt/conda/bin:${PATH}"

# --- Create conda env with Python 3.11 + pycudadecon ---
RUN conda create -n deconv python=3.11 -y \
    && conda install -n deconv -c conda-forge pycudadecon=0.5.1 -y \
    && conda clean -afy

# Activate the conda env for all subsequent RUN commands
ENV PATH="/opt/conda/envs/deconv/bin:${PATH}"
ENV CONDA_DEFAULT_ENV=deconv

# --- Copy deconwolf binaries and libraries from builder ---
COPY --from=builder /usr/local/bin/dw /usr/local/bin/dw
COPY --from=builder /usr/local/bin/dw_bw /usr/local/bin/dw_bw
COPY --from=builder /usr/lib/x86_64-linux-gnu/libtrafo.so /usr/lib/x86_64-linux-gnu/libtrafo.so
RUN ldconfig

# --- Create VkFFT kernel cache directory for deconwolf ---
RUN mkdir -p /root/.config/deconwolf

# --- Download ImageJ JAR (required by DeconvolutionLab2) ---
RUN mkdir -p /app/bin \
    && wget -q -O /app/bin/ij-1.51h.jar \
       "https://repo1.maven.org/maven2/net/imagej/ij/1.51h/ij-1.51h.jar"

# --- Copy DeconvolutionLab2 JAR ---
COPY bin/DeconvolutionLab_2.jar /app/bin/DeconvolutionLab_2.jar

WORKDIR /app

# --- Python dependencies (pip installs into the conda env) ---
COPY requirements_docker.txt /app/requirements_docker.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements_docker.txt

# --- Application code ---
COPY vendor/ /app/vendor/
COPY deconvolve.py /app/deconvolve.py
COPY benchmark.py /app/benchmark.py

# --- Set ImageJ JAR path for deconvolve.py (it checks ~/.m2 by default) ---
# We place ij JAR in /app/bin/ alongside DL2 JAR. deconvolve.py resolves
# _IJ_JAR via $HOME/.m2/...; create a symlink so it's found.
RUN mkdir -p /root/.m2/repository/net/imagej/ij/1.51h \
    && ln -s /app/bin/ij-1.51h.jar /root/.m2/repository/net/imagej/ij/1.51h/ij-1.51h.jar

# Volumes for data (input) and output (results)
VOLUME ["/app/data", "/app/output"]

# Expose all NVIDIA driver capabilities so OpenCL libraries are mounted
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

CMD ["python", "benchmark.py"]
