"""Build the production CUDA extension for inclusion in a binary distribution."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parent
    build = args.output / "build"
    build.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6;8.9;9.0;10.0;12.0+PTX")
    module = load(
        name="_optimized_cuda",
        sources=[
            str(root / "cuda" / "fused_ops.cpp"),
            str(root / "cuda" / "fused_ops.cu"),
            str(root / "cuda" / "direct_fft.cpp"),
        ],
        build_directory=str(build),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lcufft"],
        verbose=True,
    )
    target = args.output / Path(module.__file__).name
    shutil.copy2(module.__file__, target)
    print(f"Built {target} with torch={torch.__version__} CUDA={torch.version.cuda}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
