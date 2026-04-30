# ===========================================================================
# CIDeconvolve Benchmark — BIAFLOWS-compatible GPU-enabled Docker image
# ===========================================================================
# This benchmark image needs more runtimes than the main cideconvolve image:
# Java for DeconvolutionLab2, deconwolf, pycudadecon, OpenCL, cuCIM/CuPy, and
# CUDA-enabled PyTorch.  To keep the image smaller, CUDA libraries come from
# Python/conda wheels and the host NVIDIA driver mounted by NVIDIA Container
# Toolkit; the final image no longer uses an NVIDIA CUDA base image.
#
# BIAFLOWS convention: images in /data/in, results in /data/out, ground truth
# in /data/gt.  The entrypoint is wrapper.py.
# ===========================================================================

FROM ubuntu:22.04 AS deconwolf_builder

ARG DEBIAN_FRONTEND=noninteractive
ARG DECONWOLF_VERSION=v0.4.5

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        cmake \
        git \
        libfftw3-dev \
        libgsl-dev \
        libpng-dev \
        libtiff-dev \
        ocl-icd-opencl-dev \
        opencl-headers \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${DECONWOLF_VERSION}" \
        https://github.com/elgw/deconwolf.git /tmp/deconwolf \
    && cmake -S /tmp/deconwolf -B /tmp/deconwolf/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build /tmp/deconwolf/build --parallel "$(nproc)" \
    && cmake --install /tmp/deconwolf/build \
    && strip /usr/local/bin/dw /usr/local/bin/dw_bw /usr/lib/x86_64-linux-gnu/libtrafo.so

# ===========================================================================
# Runtime stage
# ===========================================================================
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH="/opt/conda/envs/deconv/bin:/opt/conda/bin:${PATH}"
ENV CONDA_DEFAULT_ENV=deconv

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        bzip2 \
        wget \
        fonts-dejavu-core \
        libfftw3-3 \
        libgomp1 \
        libgsl27 \
        libpng16-16 \
        libtiff5 \
        ocl-icd-libopencl1 \
        openjdk-17-jre-headless \
        pocl-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Let NVIDIA Container Toolkit mount host CUDA/OpenCL driver libraries.
RUN mkdir -p /etc/OpenCL/vendors \
    && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Install micromamba instead of full Miniforge/conda to avoid carrying the base
# environment and package cache in the final image.
RUN wget -q -O /tmp/micromamba.tar.bz2 \
        "https://micro.mamba.pm/api/micromamba/linux-64/latest" \
    && mkdir -p /tmp/micromamba /opt/conda/bin \
    && tar -xjf /tmp/micromamba.tar.bz2 -C /tmp/micromamba \
    && mv /tmp/micromamba/bin/micromamba /opt/conda/bin/micromamba \
    && rm -rf /tmp/micromamba /tmp/micromamba.tar.bz2

# pycudadecon is installed from conda-forge; most other dependencies are pip
# wheels.  The conda package cache is removed immediately.
RUN micromamba create -y -n deconv -c conda-forge \
        python=3.11 \
        pip \
        pycudadecon=0.5.1 \
    && micromamba clean -a -f -y

# deconwolf binaries and runtime helper library from the build stage.
COPY --from=deconwolf_builder /usr/local/bin/dw /usr/local/bin/dw
COPY --from=deconwolf_builder /usr/local/bin/dw_bw /usr/local/bin/dw_bw
COPY --from=deconwolf_builder /usr/lib/x86_64-linux-gnu/libtrafo.so /usr/lib/x86_64-linux-gnu/libtrafo.so
RUN ldconfig \
    && mkdir -p /root/.config/deconwolf /app/bin /data/in /data/out /data/gt

WORKDIR /app

# ImageJ support jars for DeconvolutionLab2.
RUN wget -q -O /app/bin/ij-1.51h.jar \
        "https://repo1.maven.org/maven2/net/imagej/ij/1.51h/ij-1.51h.jar" \
    && wget -q -O /app/bin/JTransforms-3.1.jar \
        "https://repo1.maven.org/maven2/com/github/wendykierp/JTransforms/3.1/JTransforms-3.1.jar" \
    && wget -q -O /app/bin/JLargeArrays-1.5.jar \
        "https://repo1.maven.org/maven2/pl/edu/icm/JLargeArrays/1.5/JLargeArrays-1.5.jar"

COPY bin/DeconvolutionLab_2.jar /app/bin/DeconvolutionLab_2.jar

COPY requirements_docker.txt /app/requirements_docker.txt
RUN python -m pip install --no-cache-dir --no-compile --upgrade pip \
    && python -m pip install --no-cache-dir --no-compile -r requirements_docker.txt \
    && find /opt/conda -type d -name "__pycache__" -prune -exec rm -rf {} + \
    && find /opt/conda -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.a" \) -delete \
    && micromamba clean -a -f -y

COPY vendor/ /app/vendor/
COPY deconvolve.py /app/deconvolve.py
COPY deconvolve_ci.py /app/deconvolve_ci.py
COPY bioflows_local.py /app/bioflows_local.py
COPY wrapper.py /app/wrapper.py
COPY descriptor.json /app/descriptor.json

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

ENTRYPOINT ["python", "/app/wrapper.py"]
