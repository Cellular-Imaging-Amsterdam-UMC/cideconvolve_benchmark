"""Small Bilayers-compatible CLI helper for CIDeconvolve.

This intentionally keeps the project independent from the external bilayers
package at runtime while supporting the same useful parse/generate/validate
workflow for the local Bilayers ``config.yaml`` file.
"""
from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in incomplete envs
    yaml = None


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")

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

_IMMERSION_RI = {
    "air": 1.0003,
    "water": 1.333,
    "oil": 1.515,
}

_SAMPLE_RI = {
    "water": 1.333,
    "pbs": 1.334,
    "culture medium": 1.337,
    "vectashield": 1.45,
    "prolong gold": 1.47,
    "glycerol": 1.474,
    "oil": 1.515,
    "prolong glass": 1.52,
}

_DEFAULT_NA = 1.4
_DEFAULT_EMISSION_WL = "520"
_DEFAULT_PIXEL_SIZE_XY_NM = 65.0
_DEFAULT_PIXEL_SIZE_Z_NM = 200.0
_DEFAULT_MICROSCOPE_TYPE = "confocal"
_DEFAULT_EXCITATION_WL = "488"
_DEFAULT_PINHOLE_AIRY = 1.0
_DEFAULT_IMMERSION_RI_CHOICE = "oil (1.515)"
_DEFAULT_SAMPLE_RI_CHOICE = "prolong gold (1.47)"
_SAMPLE_RI_DEFAULT = 1.47
_START_MODES = (
    "auto",
    "flat",
    "percentile_flat",
    "observed",
    "observed_bgsub",
    "lowpass",
    "lowpass_bgsub",
    "hybrid",
)


def _str_to_bool(value: str) -> bool:
    """Convert a string to a boolean for argparse."""
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1", "yes"):
        return True
    if str(value).lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


def _to_bool(value: Any) -> bool:
    """Convert CLI, JSON, and YAML boolean values to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _parse_ri_choice(raw: str, lookup: dict[str, float]) -> float | None:
    """Parse an RI preset string like ``oil (1.515)`` or a bare float."""
    text = str(raw).strip().lower()
    if not text:
        return None
    name = text.split("(")[0].strip()
    if name in lookup:
        return lookup[name]
    try:
        return float(text)
    except ValueError:
        return None


def _parse_float_or_default(raw: Any, default: float) -> float:
    """Parse a finite float, accepting non-numeric values as the default."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return value


def _parse_float_list_or_default(raw: Any, default: str) -> list[float]:
    """Parse comma- or semicolon-separated floats."""
    text = str(raw if raw is not None else default).strip()
    if not text or text.lower() == "auto":
        text = default
    values: list[float] = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values or [float(default)]


def _parse_tile_limits(raw: Any, default: tuple[int, int] = (0, 64)) -> tuple[int, int]:
    """Parse tile limits as max_xy,max_z; XY <= 0 means auto tile sizing."""
    text = str(raw or "").strip()
    if not text or text.lower() == "auto":
        return default
    parts = [p.strip() for p in text.replace("x", ",").split(",") if p.strip()]
    try:
        max_xy = int(parts[0]) if parts else default[0]
        max_z = int(parts[1]) if len(parts) > 1 else default[1]
    except ValueError:
        return default
    if max_xy <= 0:
        max_xy = 0
    return (max_xy if max_xy == 0 else max(max_xy, 64)), max(max_z, 1)


def _load_config(config_path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip install PyYAML")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Bilayers config must be a YAML mapping")
    return config


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load a Bilayers YAML configuration from disk."""
    return _load_config(Path(config_path))


def _iter_cli_items(config: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for source in ("inputs", "outputs", "parameters"):
        for spec in config.get(source, []) or []:
            item = dict(spec)
            item["source"] = source[:-1]
            items.append(item)
    for spec in config.get("exec_function", {}).get("hidden_args", []) or []:
        item = dict(spec)
        item["source"] = "hidden"
        items.append(item)
    return sorted(items, key=lambda item: int(item.get("cli_order", 0)))


def _default_folder(config: dict[str, Any], section: str, fallback: str) -> str:
    """Return the first folder_name in a Bilayers section, or a Docker fallback."""
    entries = config.get(section, []) or []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("folder_name"):
            return str(entry["folder_name"])
    return fallback


def _parameter_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return workflow parameters from config.yaml."""
    return [dict(item) for item in config.get("parameters", []) or [] if isinstance(item, dict)]


def _normalise_suffixes(suffixes: Optional[Sequence[str]]) -> list[str]:
    """Return lower-case suffixes with a leading dot."""
    if not suffixes:
        return list(DEFAULT_SUFFIXES)
    normalised: list[str] = []
    for suffix in suffixes:
        clean = str(suffix).strip().lower()
        if not clean:
            continue
        if not clean.startswith("."):
            clean = f".{clean}"
        normalised.append(clean)
    return normalised or list(DEFAULT_SUFFIXES)


@dataclass
class ImageResource:
    """Small file record used by wrapper.py."""

    filename: str
    filename_original: str
    filepath: Path

    def __post_init__(self) -> None:
        self.filepath = Path(self.filepath)
        self.path = str(self.filepath)


class BilayersJob:
    """Parsed Bilayers wrapper invocation."""

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        parameters: SimpleNamespace | None = None,
    ) -> None:
        self.parameters = parameters or getattr(args, "parameters", SimpleNamespace())
        self.input_dir = Path(args.input_dir)
        self.output_dir = Path(args.output_dir)
        temp_dir_value = getattr(args, "temp_dir", None) or self.output_dir / "tmp"
        self.temp_dir = Path(temp_dir_value)
        self.suffixes = _normalise_suffixes(getattr(args, "suffixes", None))

    def __enter__(self) -> "BilayersJob":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    @classmethod
    def from_cli(cls, argv: Sequence[str], **overrides: Any) -> "BilayersJob":
        args = _parse_wrapper_args(argv)
        parameters = overrides.pop("parameters", getattr(args, "parameters", None))
        for key, value in overrides.items():
            setattr(args, key, value)
        return cls(args, parameters=parameters)


def _collect_images(directory: Path, suffixes: Sequence[str]) -> list[ImageResource]:
    """Enumerate input files and OME-Zarr folders."""
    if not directory.exists():
        return []
    records: list[ImageResource] = []
    for entry in sorted(directory.iterdir()):
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
        if entry.suffix.lower() not in suffixes:
            continue
        records.append(
            ImageResource(
                filename=entry.name,
                filename_original=entry.name,
                filepath=entry,
            )
        )
    return records


def prepare_bilayers_data(job: BilayersJob) -> tuple[list[ImageResource], str, str, str]:
    """Create runtime directories and enumerate input images."""
    job.input_dir.mkdir(parents=True, exist_ok=True)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.temp_dir.mkdir(parents=True, exist_ok=True)
    return (
        _collect_images(job.input_dir, job.suffixes),
        str(job.input_dir),
        str(job.output_dir),
        str(job.temp_dir),
    )


def _add_parameter_argument(parser: argparse.ArgumentParser, spec: dict[str, Any]) -> str | None:
    """Add one config.yaml parameter as an argparse option."""
    param_id = spec.get("name")
    cli_tag = spec.get("cli_tag")
    if not param_id or cli_tag in (None, "", "None"):
        return None

    param_type = str(spec.get("type", "textbox")).lower()
    kwargs: dict[str, Any] = {
        "default": argparse.SUPPRESS,
        "help": spec.get("description", ""),
    }

    if param_type == "checkbox":
        kwargs["nargs"] = "?"
        kwargs["const"] = True
        kwargs["type"] = _str_to_bool
        kwargs["metavar"] = "BOOL"
    elif param_type in {"float", "number"}:
        kwargs["type"] = float
    elif param_type in {"integer", "int"}:
        kwargs["type"] = int
    else:
        kwargs["type"] = str

    parser.add_argument(str(cli_tag), dest=str(param_id), **kwargs)
    return str(param_id)


def _parse_wrapper_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse the wrapper CLI from config.yaml metadata."""
    config = _load_config(DEFAULT_CONFIG)
    parser = argparse.ArgumentParser(description="CIDeconvolve Bilayers runner.")
    parser.add_argument("--input-dir", "--infolder", dest="input_dir")
    parser.add_argument("--output-dir", "--outfolder", dest="output_dir")
    parser.add_argument("--temp-dir", dest="temp_dir", default=None)
    parser.add_argument("--local", action="store_true", help="Run as a local container job.")
    parser.add_argument(
        "--suffixes",
        nargs="*",
        default=None,
        help="File suffixes to process.",
    )
    parser.add_argument(
        "--parameters",
        dest="parameters_json",
        default=None,
        help="JSON object with parameter defaults/values.",
    )

    parameter_ids: list[str] = []
    parameter_defaults: dict[str, Any] = {}
    for spec in _parameter_specs(config):
        name = spec.get("name")
        if not name:
            continue
        parameter_defaults[str(name)] = spec.get("default")
        param_id = _add_parameter_argument(parser, spec)
        if param_id:
            parameter_ids.append(param_id)

    args, _unknown = parser.parse_known_args(argv)
    parameter_values: dict[str, Any] = dict(parameter_defaults)
    if args.parameters_json:
        try:
            loaded = json.loads(args.parameters_json)
        except json.JSONDecodeError as exc:
            parser.error(f"--parameters must be a JSON object: {exc}")
        if not isinstance(loaded, dict):
            parser.error("--parameters must be a JSON object")
        parameter_values.update(loaded)

    for param_id in parameter_ids:
        if hasattr(args, param_id):
            parameter_values[param_id] = getattr(args, param_id)
    args.parameters = SimpleNamespace(**parameter_values)

    if not args.input_dir:
        args.input_dir = _default_folder(config, "inputs", "/data/in")
    if not args.output_dir:
        args.output_dir = _default_folder(config, "outputs", "/data/out")
    return args


def resolve_workflow_parameters(parameters: object | None) -> SimpleNamespace:
    """Resolve raw CLI/Bilayers parameters into wrapper-ready values."""
    if parameters is None:
        parameters = SimpleNamespace()

    iter_raw = str(getattr(parameters, "iterations", "40")).strip()
    niter_list: list[int] = []
    for item in iter_raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            niter_list.append(max(1, int(float(item))))
        except ValueError:
            continue
    if not niter_list:
        niter_list = [40]

    method = str(getattr(parameters, "method", "ci_rl") or "ci_rl").strip()
    if method not in ("ci_rl", "ci_rl_tv", "ci_sparse_hessian"):
        method = "ci_rl"

    device_param = getattr(parameters, "device", "auto")
    device = None if device_param in (None, "auto") else device_param

    overrule_metadata = _to_bool(getattr(parameters, "overrule_image_metadata", False))
    na_value = _parse_float_or_default(getattr(parameters, "na", _DEFAULT_NA), _DEFAULT_NA)
    ri_raw = str(getattr(parameters, "refractive_index", _DEFAULT_IMMERSION_RI_CHOICE))
    ri_value = _parse_ri_choice(ri_raw, _IMMERSION_RI) or 1.515
    sample_ri_raw = str(getattr(parameters, "sample_ri", _DEFAULT_SAMPLE_RI_CHOICE))
    sample_ri_value = _parse_ri_choice(sample_ri_raw, _SAMPLE_RI) or _SAMPLE_RI_DEFAULT
    micro_value = str(getattr(parameters, "microscope_type", _DEFAULT_MICROSCOPE_TYPE)).strip().lower()
    if micro_value == "auto":
        micro_value = _DEFAULT_MICROSCOPE_TYPE
    em_raw = str(getattr(parameters, "emission_wl", _DEFAULT_EMISSION_WL)).strip()
    em_value = _parse_float_list_or_default(em_raw, _DEFAULT_EMISSION_WL)
    ex_raw = str(getattr(parameters, "excitation_wl", _DEFAULT_EXCITATION_WL)).strip()
    ex_value = _parse_float_list_or_default(ex_raw, _DEFAULT_EXCITATION_WL)
    pinhole_airy = _parse_float_list_or_default(
        getattr(parameters, "pinhole_airy", str(_DEFAULT_PINHOLE_AIRY)),
        str(_DEFAULT_PINHOLE_AIRY),
    )

    tv_lambda = _parse_float_or_default(getattr(parameters, "tv_lambda", 0.0001), 0.0001)

    bg_raw = str(getattr(parameters, "background", "auto")).strip()
    background: float | str = "auto" if bg_raw.lower() == "auto" else _parse_float_or_default(bg_raw, 0.0)
    offset_raw = str(getattr(parameters, "offset", "auto")).strip().lower()
    if offset_raw in ("none", "0", "0.0"):
        offset: float | str = 0.0
    elif offset_raw == "auto":
        offset = "auto"
    else:
        offset = _parse_float_or_default(offset_raw, 0.0)

    prefilter_sigma = max(0.0, _parse_float_or_default(getattr(parameters, "prefilter_sigma", 0.0), 0.0))
    legacy_snr = getattr(parameters, "snr", None)
    if legacy_snr is not None:
        snr_raw = str(legacy_snr or "off").strip().lower()
        if snr_raw in ("off", "none", ""):
            snr: float | str | None = None
        elif snr_raw == "auto":
            snr = "auto"
        else:
            snr = _parse_float_or_default(snr_raw, 0.0)
            if snr <= 0.0:
                snr = None
    else:
        snr_mode = str(getattr(parameters, "snr_mode", "none") or "none").strip().lower()
        if snr_mode == "auto":
            snr = "auto"
        elif snr_mode == "manual":
            snr = _parse_float_or_default(getattr(parameters, "snr_value", 4.0), 4.0)
            if snr <= 0.0:
                snr = 4.0
        else:
            snr = None
    acuity = min(max(_parse_float_or_default(getattr(parameters, "acuity", 0.0), 0.0), -100.0), 100.0)
    start = str(getattr(parameters, "start", "auto")).strip().lower()
    if start not in _START_MODES:
        start = "flat"
    sparse_hessian_weight = min(
        max(_parse_float_or_default(getattr(parameters, "sparse_hessian_weight", 0.6), 0.6), 0.0),
        1.0,
    )
    sparse_hessian_reg = min(
        max(_parse_float_or_default(getattr(parameters, "sparse_hessian_reg", 0.98), 0.98), 0.0),
        1.0,
    )
    convergence = str(getattr(parameters, "convergence", "auto")).strip().lower()
    if convergence in ("none", "fixed"):
        convergence = "fixed"
    elif convergence != "auto":
        convergence = "auto"
    rel_threshold = min(
        max(_parse_float_or_default(getattr(parameters, "rel_threshold", 0.005), 0.005), 1e-8),
        1.0,
    )
    check_every = 5

    t_g = 170000.0
    t_g0 = 170000.0
    t_i0 = 100000.0
    z_p = 0.0

    px_xy_raw = str(getattr(parameters, "pixel_size_xy", _DEFAULT_PIXEL_SIZE_XY_NM)).strip()
    px_xy_nm = _parse_float_or_default(px_xy_raw, _DEFAULT_PIXEL_SIZE_XY_NM)
    px_xy_value = px_xy_nm / 1000.0
    px_z_raw = str(getattr(parameters, "pixel_size_z", _DEFAULT_PIXEL_SIZE_Z_NM)).strip()
    px_z_nm = _parse_float_or_default(px_z_raw, _DEFAULT_PIXEL_SIZE_Z_NM)
    px_z_value = px_z_nm / 1000.0

    projection = str(getattr(parameters, "projection", "none")).lower()
    benchmark = _to_bool(getattr(parameters, "benchmark", False))
    bench_crop = _to_bool(getattr(parameters, "bench_crop", False))
    compute_metrics = _to_bool(getattr(parameters, "compute_metrics", False))
    output_format = str(getattr(parameters, "output_format", "ome-zarr")).strip().lower()
    if output_format in ("ome_zarr", "zarr"):
        output_format = "ome-zarr"
    output_dtype = str(getattr(parameters, "output_dtype", "float32")).strip().lower()
    if output_dtype in ("uint16", "ushort", "u16"):
        output_dtype = "uint16"
    else:
        output_dtype = "float32"
    streaming_mode = str(getattr(parameters, "streaming", "auto")).strip().lower()
    tile_limits = _parse_tile_limits(getattr(parameters, "tile_limits", "auto"))
    streaming_threshold_gb = max(
        _parse_float_or_default(getattr(parameters, "streaming_threshold_gb", 2.0), 2.0),
        0.01,
    )
    t_start = max(int(_parse_float_or_default(getattr(parameters, "t_start", 1), 1)), 1)
    t_stop_raw = getattr(parameters, "t_stop", 0)
    t_stop = int(_parse_float_or_default(t_stop_raw, 0))
    t_step = max(int(_parse_float_or_default(getattr(parameters, "t_step", 1), 1)), 1)
    hcs_field = getattr(parameters, "hcs_field", None)
    hcs_field = None if hcs_field in (None, "", "auto") else str(hcs_field)

    two_d_mode = str(getattr(parameters, "two_d_mode", "auto")).strip().lower()
    two_d_wf_aggressiveness = str(getattr(parameters, "two_d_wf_aggressiveness", "Balanced")).strip()
    two_d_wf_bg_radius_um = max(
        _parse_float_or_default(getattr(parameters, "two_d_wf_bg_radius_um", 0.5), 0.5),
        0.1,
    )
    two_d_wf_bg_scale = max(
        _parse_float_or_default(getattr(parameters, "two_d_wf_bg_scale", 1.0), 1.0),
        0.1,
    )

    return SimpleNamespace(
        niter_list=niter_list,
        method=method,
        device_param=device_param,
        device=device,
        overrule_metadata=overrule_metadata,
        na_value=na_value,
        ri_raw=ri_raw,
        ri_value=ri_value,
        sample_ri_raw=sample_ri_raw,
        sample_ri_value=sample_ri_value,
        micro_value=micro_value,
        em_value=em_value,
        ex_value=ex_value,
        pinhole_airy=pinhole_airy,
        tv_lambda=tv_lambda,
        background=background,
        offset=offset,
        prefilter_sigma=prefilter_sigma,
        snr=snr,
        acuity=acuity,
        start=start,
        sparse_hessian_weight=sparse_hessian_weight,
        sparse_hessian_reg=sparse_hessian_reg,
        convergence=convergence,
        rel_threshold=rel_threshold,
        check_every=check_every,
        t_g=t_g,
        t_g0=t_g0,
        t_i0=t_i0,
        z_p=z_p,
        px_xy_nm=px_xy_nm,
        px_z_nm=px_z_nm,
        na_override=na_value,
        ri_override=ri_value,
        sample_ri=sample_ri_value,
        micro_override=micro_value,
        em_override=em_value,
        ex_override=ex_value,
        pinhole_airy_override=pinhole_airy,
        px_xy_override=px_xy_value,
        px_z_override=px_z_value,
        projection=projection,
        benchmark=benchmark,
        bench_crop=bench_crop,
        compute_metrics=compute_metrics,
        output_format=output_format,
        output_dtype=output_dtype,
        streaming_mode=streaming_mode,
        tile_limits=tile_limits,
        streaming_threshold_gb=streaming_threshold_gb,
        t_start=t_start,
        t_stop=t_stop,
        t_step=t_step,
        hcs_field=hcs_field,
        two_d_mode=two_d_mode,
        two_d_wf_aggressiveness=two_d_wf_aggressiveness,
        two_d_wf_bg_radius_um=two_d_wf_bg_radius_um,
        two_d_wf_bg_scale=two_d_wf_bg_scale,
    )


def _format_cli_arg(cli_tag: str, value: Any) -> str:
    if cli_tag in ("", None):
        return shlex.quote(str(value))
    if "=" in cli_tag:
        return f"{cli_tag}{shlex.quote(str(value))}"
    return f"{cli_tag} {shlex.quote(str(value))}"


def generate_cli_command(config: dict[str, Any]) -> str:
    command = [str(config.get("exec_function", {}).get("cli_command", "python wrapper.py")).strip()]
    for item in _iter_cli_items(config):
        cli_tag = item.get("cli_tag")
        if cli_tag in (None, "None"):
            continue
        if item["source"] == "hidden":
            value = item.get("value")
            append_value = bool(item.get("append_value", True))
            if not append_value:
                if _to_bool(value):
                    command.append(str(cli_tag))
                continue
        else:
            value = item.get("default")
            append_value = bool(item.get("append_value", False))
            if item.get("type") in {"image", "file", "directory"} and item.get("folder_name"):
                value = item["folder_name"]

        if item.get("type") == "checkbox" or isinstance(value, bool):
            if append_value:
                command.append(_format_cli_arg(str(cli_tag), value))
            elif value:
                command.append(str(cli_tag))
            continue

        if value not in (None, ""):
            command.append(_format_cli_arg(str(cli_tag), value))
    return " ".join(part for part in command if part)


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("citations", "docker_image", "algorithm_folder_name", "exec_function", "inputs", "outputs", "parameters", "display_only"):
        if key not in config:
            errors.append(f"Missing top-level key: {key}")
    for section in ("inputs", "outputs", "parameters"):
        value = config.get(section, [])
        if not isinstance(value, list):
            errors.append(f"{section} must be a list")
            continue
        seen: set[str] = set()
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                errors.append(f"{section}[{index}] must be a mapping")
                continue
            name = item.get("name")
            if not name:
                errors.append(f"{section}[{index}] is missing name")
            elif name in seen:
                errors.append(f"{section} has duplicate name: {name}")
            seen.add(str(name))
            if section != "outputs" and not item.get("cli_tag"):
                errors.append(f"{section}.{name} is missing cli_tag")
            mode = item.get("mode")
            if mode not in ("beginner", "advanced"):
                errors.append(f"{section}.{name} mode must be beginner or advanced")
            if section in ("inputs", "outputs") and item.get("type") == "image":
                for field_name in ("subtype", "depth", "timepoints", "tiled", "pyramidal"):
                    if field_name not in item:
                        errors.append(f"{section}.{name} image entry is missing {field_name}")
            if section in ("inputs", "outputs") and "unique_string" not in item:
                errors.append(f"{section}.{name} is missing unique_string")
    return errors


def validate_config_strict(config: dict[str, Any]) -> list[str]:
    """Validate config.yaml with upstream Bilayers LinkML schema packages."""
    try:
        import linkml.validator as linkml_validator
        from bilayers_schema import schema
    except ImportError as exc:
        return [
            "Strict validation dependency missing. Install with: "
            "pip install linkml git+https://github.com/bilayer-containers/bilayers-schema.git"
            f" ({exc})"
        ]

    report = linkml_validator.validate(config, schema)
    return [f"[{result.severity.value}] {result.message}" for result in report.results]


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CIDeconvolve Bilayers YAML helper.",
    )
    parser.add_argument("-v", "--version", action="version", version="bilayers_cli 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="Parse a Bilayers YAML config file.")
    parse_parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG))

    generate_parser = subparsers.add_parser("generate", help="Generate outputs from a Bilayers YAML config file.")
    generate_parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG))
    generate_parser.add_argument("--cli", action="store_true", help="Generate the default CLI command.")

    validate_parser = subparsers.add_parser("validate", help="Validate a Bilayers YAML config file.")
    validate_parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG))
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="Also validate against the upstream Bilayers LinkML schema.",
    )

    args = parser.parse_args(argv)
    try:
        config = _load_config(Path(args.config))
        if args.command == "parse":
            image = config["docker_image"]
            print(f"Inputs: {len(config.get('inputs', []))}")
            print(f"Outputs: {len(config.get('outputs', []))}")
            print(f"Parameters: {len(config.get('parameters', []))}")
            print(f"Docker Image: {image['org']}/{image['name']}:{image['tag']} ({image['platform']})")
            print(f"CLI Sequence Order: {[item.get('name', item.get('cli_tag')) for item in _iter_cli_items(config)]}")
        elif args.command == "generate":
            if not args.cli:
                print("Only --cli generation is implemented for this local helper.")
                return 1
            print("Generated CLI Command:")
            print(generate_cli_command(config))
        elif args.command == "validate":
            errors = validate_config(config)
            if getattr(args, "strict", False) and not errors:
                errors.extend(validate_config_strict(config))
            if errors:
                for error in errors:
                    print(f"[ERROR] {error}")
                return 1
            print("No issues found")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(cli())
