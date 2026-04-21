"""
bioflows_local.py — Local BIAFLOWS job helper for CIDeconvolve.

Provides a BiaflowsJob class and helper functions that mirror the
Cytomine/BIAFLOWS runner API so that the workflow can run locally
(inside Docker or on the host) without any Cytomine dependencies.

Based on the pattern from W_CellExpansionAdvanced.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence

CLASS_SPTCNT = "LOCAL_CLASS_SPTCNT"

KNOWN_JOB_ATTRS = {
    "input_dir",
    "output_dir",
    "gt_dir",
    "temp_dir",
    "suffixes",
    "local",
    "parameters",
}

DEFAULT_SUFFIXES = (
    ".tif",
    ".tiff",
    ".ome.tif",
    ".ome.tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".npy",
)


def _str_to_bool(value: str) -> bool:
    """Convert a string to a boolean for argparse."""
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


def _load_descriptor_inputs() -> List[dict]:
    """Return parameter definitions declared in descriptor.json if available."""
    descriptor_path = Path(__file__).with_name("descriptor.json")
    try:
        with descriptor_path.open("r", encoding="utf-8") as stream:
            descriptor = json.load(stream)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as exc:
        print(f"Warning: descriptor.json could not be parsed ({exc}); ignoring parameter metadata.")
        return []
    inputs = descriptor.get("inputs", [])
    if not isinstance(inputs, list):
        return []
    return inputs


@dataclass
class ImageResource:
    """Minimal image representation compatible with the BIAFLOWS wrapper."""

    filename: str
    filename_original: str
    filepath: Path

    def __post_init__(self) -> None:
        self.filepath = Path(self.filepath)
        self.path = str(self.filepath)


class BiaflowsJob:
    """Local stand-in for the Cytomine/BIAFLOWS job helper."""

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        parameters: Optional[SimpleNamespace] = None,
    ) -> None:
        if parameters is None:
            parameters = getattr(args, "parameters", None)
        if parameters is None:
            param_values = {
                key: value
                for key, value in vars(args).items()
                if key not in KNOWN_JOB_ATTRS
            }
            parameters = SimpleNamespace(**param_values)

        self.parameters = parameters
        self.flags = {}
        self.input_dir = Path(args.input_dir)
        self.output_dir = Path(args.output_dir)
        self.gt_dir = Path(args.gt_dir)

        temp_dir_value = getattr(args, "temp_dir", None)
        if temp_dir_value is None:
            temp_dir_value = self.output_dir / "tmp"
        self.temp_dir = Path(temp_dir_value)
        self.suffixes = self._normalise_suffixes(args.suffixes)

    def __enter__(self) -> "BiaflowsJob":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    @classmethod
    def from_cli(
        cls,
        argv: Sequence[str],
        **overrides,
    ) -> "BiaflowsJob":
        args = _parse_args(argv)
        parameters = overrides.pop(
            "parameters",
            getattr(args, "parameters", None),
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return cls(args, parameters=parameters)

    @staticmethod
    def _normalise_suffixes(
        suffixes: Optional[Sequence[str]],
    ) -> Optional[List[str]]:
        if not suffixes:
            return list(DEFAULT_SUFFIXES)
        normalised: List[str] = []
        for suffix in suffixes:
            clean = suffix.strip().lower()
            if not clean:
                continue
            if not clean.startswith("."):
                clean = f".{clean}"
            normalised.append(clean)
        return normalised or list(DEFAULT_SUFFIXES)


def prepare_data(
    discipline: str,
    job: BiaflowsJob,
    *,
    is_2d: bool = True,
    **flags,
):
    """Prepare input/output directories and enumerate available images."""
    del discipline, flags
    if not is_2d:
        print("Warning: local workflow set is_2d=False; proceeding regardless.")

    job.input_dir.mkdir(parents=True, exist_ok=True)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.temp_dir.mkdir(parents=True, exist_ok=True)

    in_imgs = _collect_images(job.input_dir, job.suffixes)
    gt_imgs = _collect_images(job.gt_dir, job.suffixes)

    return (
        in_imgs,
        gt_imgs,
        str(job.input_dir),
        str(job.gt_dir),
        str(job.output_dir),
        str(job.temp_dir),
    )


def get_discipline(job: BiaflowsJob, default: Optional[str] = None) -> Optional[str]:
    """Return the requested default discipline (placeholder for compatibility)."""
    del job
    return default


def _collect_images(directory: Path, suffixes: Optional[Sequence[str]]) -> List[ImageResource]:
    if not directory.exists():
        return []
    records: List[ImageResource] = []
    for entry in sorted(directory.iterdir()):
        # OME-Zarr stores are directories ending in .zarr
        if entry.is_dir() and entry.suffix.lower() == ".zarr":
            records.append(
                ImageResource(
                    filename=entry.name,
                    filename_original=entry.name,
                    filepath=entry,
                )
            )
            continue
        if not entry.is_file():
            continue
        if suffixes and entry.suffix.lower() not in suffixes:
            continue
        records.append(
            ImageResource(
                filename=entry.name,
                filename_original=entry.name,
                filepath=entry,
            )
        )
    return records


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local BIAFLOWS runner for CIDeconvolve."
    )
    # BIAFLOWS standard directory arguments
    parser.add_argument("--input-dir", dest="input_dir")
    parser.add_argument(
        "--infolder", dest="input_dir",
        help="Compatibility alias for --input-dir.",
    )
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument(
        "--outfolder", dest="output_dir",
        help="Compatibility alias for --output-dir.",
    )
    parser.add_argument("--gt-dir", dest="gt_dir", default="")
    parser.add_argument(
        "--gtfolder", dest="gt_dir",
        help="Compatibility alias for --gt-dir.",
    )
    parser.add_argument("--temp-dir", dest="temp_dir", default=None)
    parser.add_argument(
        "--local", action="store_true",
        help="Run locally without Cytomine.",
    )
    parser.add_argument(
        "--suffixes", nargs="*", default=None,
        help="File suffixes to process (default: .tif .tiff .ome.tif .ome.tiff .png).",
    )

    # Descriptor-defined parameters (loaded from descriptor.json)
    descriptor_inputs = _load_descriptor_inputs()
    for inp in descriptor_inputs:
        param_id = inp.get("id")
        if not param_id:
            continue
        flag = inp.get("command-line-flag", f"--{param_id}")
        param_type = inp.get("type", "String")
        default = inp.get("default-value")

        kwargs = {"default": default, "help": inp.get("description", "")}

        if param_type == "Boolean":
            kwargs["nargs"] = "?"
            kwargs["const"] = True
            kwargs["default"] = bool(default) if default is not None else False
            kwargs["type"] = _str_to_bool
            kwargs["metavar"] = "BOOL"
        elif param_type == "Number":
            is_int = inp.get("integer", False)
            kwargs["type"] = int if is_int else float
        else:
            # Use nargs="+" so Docker CLI space-separated values like
            # --bench_methods sdeconv_rl, ci_rl, ci_rl_tv  are all captured
            # instead of only the first token.
            kwargs["type"] = str
            kwargs["nargs"] = "+"

        parser.add_argument(flag, dest=param_id, **kwargs)

    args, unknown = parser.parse_known_args(argv)

    # Join multi-word string arguments back into single strings.
    # nargs="+" produces a list; join them so downstream code sees
    # the same comma-separated string it expects.
    for inp in descriptor_inputs:
        param_id = inp.get("id")
        param_type = inp.get("type", "String")
        if param_id and param_type not in ("Boolean", "Number"):
            val = getattr(args, param_id, None)
            if isinstance(val, list):
                setattr(args, param_id, " ".join(val))

    # Default directories for Docker convention
    if not args.input_dir:
        args.input_dir = "/data/in"
    if not args.output_dir:
        args.output_dir = "/data/out"
    if not args.gt_dir:
        args.gt_dir = "/data/gt"

    return args
