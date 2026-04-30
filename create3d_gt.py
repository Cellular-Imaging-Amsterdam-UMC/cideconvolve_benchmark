"""
Create deterministic synthetic 3D GFP ground-truth and blurred/noisy input.

This local helper writes two OME-TIFF files:
  - synthetic_object_gt.ome.tiff
  - synthetic_blurred_noisy_snr<N>.ome.tiff

It can run from a small PyQt6 GUI or from the command line with --no-gui.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "synthetic"
PINHOLE_UNIT = "Airy Disk"


@dataclass(frozen=True)
class SyntheticConfig:
    output: Path = DEFAULT_OUTPUT
    z: int = 64
    yx: int = 256
    pixel_size_xy_nm: float = 40.0
    pixel_size_z_nm: float = 100.0
    na: float = 1.4
    magnification: float = 63.0
    immersion_ri: float = 1.518
    sample_ri: float = 1.47
    microscope_type: str = "confocal"
    excitation_nm: float = 488.0
    emission_nm: float = 520.0
    pinhole_size_airy: float = 1.0
    snr: float = 10.0
    seed: int = 12345
    psf_xy: int = 129
    psf_z: int = 65
    n_pupil: int = 129


def positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return value


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return value


def nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def odd_positive_int(text: str) -> int:
    value = positive_int(text)
    if value % 2 == 0:
        raise argparse.ArgumentTypeError("value must be an odd positive integer")
    return value


def _normalise(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def pinhole_size_um(config: SyntheticConfig) -> float:
    """Convert pinhole size from Airy Disk units to micrometers at the pinhole."""
    emission_um = config.emission_nm / 1000.0
    return config.pinhole_size_airy * 1.22 * emission_um * config.magnification / config.na


def _grid(config: SyntheticConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.linspace(-1.0, 1.0, config.z, dtype=np.float32)[:, None, None]
    y = np.linspace(-1.0, 1.0, config.yx, dtype=np.float32)[None, :, None]
    x = np.linspace(-1.0, 1.0, config.yx, dtype=np.float32)[None, None, :]
    return z, y, x


def _cell_mask(config: SyntheticConfig) -> np.ndarray:
    z, y, x = _grid(config)
    shell = (x / 0.82) ** 2 + (y / 0.72) ** 2 + (z / 0.56) ** 2
    mask = shell <= 1.0
    nucleus = ((x + 0.10) / 0.28) ** 2 + ((y - 0.05) / 0.24) ** 2 + (z / 0.30) ** 2
    return mask & (nucleus > 1.0)


def _add_gaussian_spot(
    img: np.ndarray,
    center: tuple[float, float, float],
    sigma: tuple[float, float, float],
    amplitude: float,
) -> None:
    zz = np.arange(img.shape[0], dtype=np.float32)[:, None, None]
    yy = np.arange(img.shape[1], dtype=np.float32)[None, :, None]
    xx = np.arange(img.shape[2], dtype=np.float32)[None, None, :]
    cz, cy, cx = center
    sz, sy, sx = sigma
    spot = np.exp(
        -0.5
        * (
            ((zz - cz) / sz) ** 2
            + ((yy - cy) / sy) ** 2
            + ((xx - cx) / sx) ** 2
        )
    )
    img += amplitude * spot.astype(np.float32)


def _add_tube(
    img: np.ndarray,
    points: np.ndarray,
    radius_px: float,
    amplitude: float,
) -> None:
    """Rasterize a soft 3D tube around a deterministic polyline."""
    rz = max(radius_px * 0.65, 0.8)
    ryx = max(radius_px, 1.0)
    margin_z = int(np.ceil(3.0 * rz))
    margin_yx = int(np.ceil(3.0 * ryx))

    for p0, p1 in zip(points[:-1], points[1:]):
        segment = p1 - p0
        seg_len_sq = float(np.dot(segment, segment))
        if seg_len_sq <= 1e-12:
            continue

        z0 = max(int(np.floor(min(p0[0], p1[0]) - margin_z)), 0)
        z1 = min(int(np.ceil(max(p0[0], p1[0]) + margin_z)) + 1, img.shape[0])
        y0 = max(int(np.floor(min(p0[1], p1[1]) - margin_yx)), 0)
        y1 = min(int(np.ceil(max(p0[1], p1[1]) + margin_yx)) + 1, img.shape[1])
        x0 = max(int(np.floor(min(p0[2], p1[2]) - margin_yx)), 0)
        x1 = min(int(np.ceil(max(p0[2], p1[2]) + margin_yx)) + 1, img.shape[2])
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue

        zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1].astype(np.float32)
        coords = np.stack((zz, yy, xx), axis=-1)
        rel = coords - p0.astype(np.float32)
        t = np.clip(np.sum(rel * segment.astype(np.float32), axis=-1) / seg_len_sq, 0.0, 1.0)
        nearest = p0 + t[..., None] * segment
        dz = (coords[..., 0] - nearest[..., 0]) / rz
        dy = (coords[..., 1] - nearest[..., 1]) / ryx
        dx = (coords[..., 2] - nearest[..., 2]) / ryx
        tube = np.exp(-0.5 * (dz * dz + dy * dy + dx * dx)).astype(np.float32)
        img[z0:z1, y0:y1, x0:x1] += amplitude * tube


def _deterministic_tubes(config: SyntheticConfig) -> list[np.ndarray]:
    cz = (config.z - 1) / 2.0
    cy = (config.yx - 1) / 2.0
    cx = (config.yx - 1) / 2.0
    tubes: list[np.ndarray] = []
    t = np.linspace(0.0, 1.0, 96, dtype=np.float32)
    golden = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(18):
        angle = i * golden
        sweep = 1.3 + 0.35 * np.sin(i)
        radius = (0.16 + 0.028 * i) * config.yx
        wobble = 0.055 * config.yx
        x = cx + radius * t * np.cos(angle + sweep * t) + wobble * np.sin(5.0 * np.pi * t + i)
        y = cy + radius * t * np.sin(angle + sweep * t) + wobble * np.cos(4.0 * np.pi * t + 0.4 * i)
        z = cz + (0.34 * config.z) * (t - 0.5) * np.sin(angle * 0.7) + 2.0 * np.sin(2.0 * np.pi * t + i)
        tubes.append(np.stack((z, y, x), axis=1).astype(np.float32))

    for i in range(8):
        angle = i * np.pi / 4.0
        helix = 2.0 * np.pi * (1.0 + 0.12 * i) * t
        x = cx + (0.48 * config.yx) * np.cos(angle) * (t - 0.5) + 9.0 * np.sin(helix)
        y = cy + (0.48 * config.yx) * np.sin(angle) * (t - 0.5) + 9.0 * np.cos(helix)
        z = cz + (0.20 * config.z) * np.sin(2.0 * np.pi * t + angle)
        tubes.append(np.stack((z, y, x), axis=1).astype(np.float32))

    return tubes


def create_object(config: SyntheticConfig) -> np.ndarray:
    """Create deterministic GFP-like cellular structures."""
    obj = np.zeros((config.z, config.yx, config.yx), dtype=np.float32)
    mask = _cell_mask(config)

    z, y, x = _grid(config)
    cytoplasm = 0.045 + 0.035 * (1.0 - ((x / 0.82) ** 2 + (y / 0.72) ** 2 + (z / 0.56) ** 2))
    obj += np.where(mask, cytoplasm, 0.012).astype(np.float32)

    for i, points in enumerate(_deterministic_tubes(config)):
        tube = np.zeros_like(obj)
        _add_tube(tube, points, radius_px=1.35 + 0.12 * (i % 4), amplitude=0.55 + 0.08 * (i % 3))
        obj += tube * mask

    cz = (config.z - 1) / 2.0
    cy = (config.yx - 1) / 2.0
    cx = (config.yx - 1) / 2.0
    golden = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(70):
        f = (i + 0.5) / 70.0
        r = np.sqrt(f) * 0.68
        angle = i * golden
        x_pos = cx + r * 0.80 * config.yx * np.cos(angle)
        y_pos = cy + r * 0.66 * config.yx * np.sin(angle)
        z_pos = cz + 0.46 * config.z * np.sin(2.4 * angle) * (1.0 - 0.35 * f)
        center = (z_pos, y_pos, x_pos)
        sigma = (
            1.0 + 0.25 * ((i + 1) % 3),
            2.2 + 0.45 * (i % 4),
            2.0 + 0.35 * ((i + 2) % 4),
        )
        amplitude = 0.25 + 0.28 * ((i % 5) / 4.0)
        spot = np.zeros_like(obj)
        _add_gaussian_spot(spot, center, sigma, amplitude)
        obj += spot * mask

    return _normalise(np.clip(obj, 0.0, None))


def generate_psf(config: SyntheticConfig) -> np.ndarray:
    from deconvolve_ci import ci_generate_psf

    psf = ci_generate_psf(
        na=config.na,
        wavelength_nm=config.emission_nm,
        pixel_size_xy_nm=config.pixel_size_xy_nm,
        pixel_size_z_nm=config.pixel_size_z_nm,
        n_xy=config.psf_xy,
        n_z=config.psf_z,
        ri_immersion=config.immersion_ri,
        ri_sample=config.sample_ri,
        ri_immersion_design=config.immersion_ri,
        microscope_type=config.microscope_type,
        excitation_nm=config.excitation_nm,
        pinhole_airy_units=config.pinhole_size_airy,
        n_pupil=config.n_pupil,
        device=None,
    )
    psf = np.asarray(psf, dtype=np.float32)
    return psf / max(float(psf.sum()), 1e-12)


def blur_and_noise(obj: np.ndarray, psf: np.ndarray, config: SyntheticConfig) -> np.ndarray:
    from scipy.signal import fftconvolve

    blurred = fftconvolve(obj, psf, mode="same").astype(np.float32)
    blurred = np.clip(blurred + 0.015, 0.0, None)
    peak = max(float(blurred.max()), 1e-12)
    photon_peak = config.snr ** 2
    scaled = blurred / peak * photon_peak
    rng = np.random.default_rng(config.seed)
    noisy = rng.poisson(scaled).astype(np.float32) / photon_peak * peak
    return np.clip(noisy, 0.0, None).astype(np.float32)


def _immersion_name(ri: float) -> str:
    if abs(ri - 1.515) < 0.02 or abs(ri - 1.518) < 0.02:
        return "oil"
    if abs(ri - 1.333) < 0.02:
        return "water"
    if abs(ri - 1.0) < 0.02:
        return "air"
    if abs(ri - 1.474) < 0.02:
        return "glycerol"
    return "other"


def _ome_xml(config: SyntheticConfig, image_kind: str, data: np.ndarray) -> bytes:
    from ome_types import to_xml
    from ome_types.model import (
        AnnotationRef,
        Channel,
        Detector,
        DetectorSettings,
        Image,
        Instrument,
        InstrumentRef,
        Map,
        MapAnnotation,
        Microscope,
        Objective,
        ObjectiveSettings,
        OME,
        Pixels,
        StructuredAnnotations,
        TiffData,
        UnitsLength,
    )

    px_xy_um = config.pixel_size_xy_nm / 1000.0
    px_z_um = config.pixel_size_z_nm / 1000.0
    pinhole_um = pinhole_size_um(config)
    image_name = f"synthetic {image_kind}"
    description = (
        f"Synthetic {image_kind}; NA={config.na}; "
        f"Magnification={config.magnification:g}x; "
        f"ImmersionRI={config.immersion_ri}; SampleRI={config.sample_ri}; "
        f"Microscope={config.microscope_type}; "
        f"Excitation={config.excitation_nm:g} nm; Emission={config.emission_nm:g} nm; "
        f"Pinhole={config.pinhole_size_airy:.2f} {PINHOLE_UNIT}; "
        f"PinholeSize={pinhole_um:.3f} µm; "
        f"PinholeSizeUnit=µm; PeakSNR={config.snr:g}; Seed={config.seed}"
    )

    channel_acquisition_mode = get_args(Channel.model_fields["acquisition_mode"].annotation)[0]
    objective_immersion = get_args(Objective.model_fields["immersion"].annotation)[0]
    microscope_type = get_args(Microscope.model_fields["type"].annotation)[0]
    objective_medium = get_args(ObjectiveSettings.model_fields["medium"].annotation)[0]

    immersion = _immersion_name(config.immersion_ri)
    immersion_enum = {
        "air": objective_immersion.AIR,
        "water": objective_immersion.WATER,
        "oil": objective_immersion.OIL,
        "glycerol": objective_immersion.GLYCEROL,
    }.get(immersion, objective_immersion.OTHER)
    medium_enum = {
        "air": objective_medium.AIR,
        "water": objective_medium.WATER,
        "oil": objective_medium.OIL,
        "glycerol": objective_medium.GLYCEROL,
    }.get(immersion, objective_medium.OTHER)
    acquisition_mode = (
        channel_acquisition_mode.LASER_SCANNING_CONFOCAL_MICROSCOPY
        if config.microscope_type == "confocal"
        else channel_acquisition_mode.WIDE_FIELD
    )

    annotations = MapAnnotation(
        id="Annotation:0",
        namespace="create3d_gt.py/synthetic-metadata",
        value=Map(
            ms=[
                Map.M(k="SampleRefractiveIndex", value=f"{config.sample_ri:g}"),
                Map.M(k="PinholeAiryUnits", value=f"{config.pinhole_size_airy:g}"),
                Map.M(k="PinholeSizeUnitAiry", value=PINHOLE_UNIT),
                Map.M(k="PeakSNR", value=f"{config.snr:g}"),
                Map.M(k="Seed", value=str(config.seed)),
            ]
        ),
    )

    ome = OME(
        creator="create3d_gt.py",
        instruments=[
            Instrument(
                id="Instrument:0",
                microscope=Microscope(
                    type=microscope_type.OTHER,
                    model=config.microscope_type,
                ),
                detectors=[Detector(id="Detector:0", model="synthetic detector")],
                objectives=[
                    Objective(
                        id="Objective:0",
                        lens_na=config.na,
                        nominal_magnification=config.magnification,
                        immersion=immersion_enum,
                    )
                ],
            )
        ],
        images=[
            Image(
                id="Image:0",
                name=image_name,
                description=description,
                instrument_ref=InstrumentRef(id="Instrument:0"),
                objective_settings=ObjectiveSettings(
                    id="Objective:0",
                    medium=medium_enum,
                    refractive_index=config.immersion_ri,
                ),
                annotation_refs=[AnnotationRef(id="Annotation:0")],
                pixels=Pixels(
                    id="Pixels:0",
                    dimension_order="XYZCT",
                    type="float",
                    size_x=int(data.shape[2]),
                    size_y=int(data.shape[1]),
                    size_z=int(data.shape[0]),
                    size_c=1,
                    size_t=1,
                    physical_size_x=px_xy_um,
                    physical_size_x_unit=UnitsLength.MICROMETER,
                    physical_size_y=px_xy_um,
                    physical_size_y_unit=UnitsLength.MICROMETER,
                    physical_size_z=px_z_um,
                    physical_size_z_unit=UnitsLength.MICROMETER,
                    channels=[
                        Channel(
                            id="Channel:0:0",
                            name="GFP",
                            samples_per_pixel=1,
                            acquisition_mode=acquisition_mode,
                            excitation_wavelength=config.excitation_nm,
                            excitation_wavelength_unit=UnitsLength.NANOMETER,
                            emission_wavelength=config.emission_nm,
                            emission_wavelength_unit=UnitsLength.NANOMETER,
                            pinhole_size=pinhole_um,
                            pinhole_size_unit=UnitsLength.MICROMETER,
                            detector_settings=DetectorSettings(id="Detector:0"),
                            fluor="GFP",
                        )
                    ],
                    tiff_data_blocks=[
                        TiffData(ifd=0, first_z=0, first_t=0, first_c=0, plane_count=int(data.shape[0]))
                    ],
                ),
            )
        ],
        structured_annotations=StructuredAnnotations(map_annotations=[annotations]),
    )
    return to_xml(ome).encode("utf-8")


def write_ome(path: Path, data: np.ndarray, config: SyntheticConfig, image_kind: str) -> None:
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(path),
        data.astype(np.float32),
        description=_ome_xml(config, image_kind, data),
        photometric="minisblack",
        compression="zlib",
        metadata={"axes": "ZYX"},
    )


def generate_pair(config: SyntheticConfig) -> tuple[Path, Path]:
    validate_config(config)
    obj = create_object(config)
    psf = generate_psf(config)
    noisy = blur_and_noise(obj, psf, config)

    gt_path = config.output / "synthetic_object_gt.ome.tiff"
    noisy_path = config.output / f"synthetic_blurred_noisy_snr{config.snr:g}.ome.tiff"
    write_ome(gt_path, obj, config, "ground truth object")
    write_ome(noisy_path, noisy, config, "blurred noisy input")
    return gt_path, noisy_path


def validate_config(config: SyntheticConfig) -> None:
    if config.pinhole_size_airy <= 0:
        raise ValueError("pinhole_size_airy must be > 0")
    if config.magnification <= 0:
        raise ValueError("magnification must be > 0")
    if config.snr <= 0:
        raise ValueError("snr must be > 0")
    if config.seed < 0:
        raise ValueError("seed must be >= 0")
    for name in ("z", "yx", "psf_xy", "psf_z", "n_pupil"):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be > 0")
    for name in ("psf_xy", "psf_z", "n_pupil"):
        if getattr(config, name) % 2 == 0:
            raise ValueError(f"{name} must be odd")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gui", action="store_true", help="Run generation directly without opening PyQt6 GUI.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output folder for the two OME-TIFF files.")
    parser.add_argument("--z", type=positive_int, default=64, help="Number of Z planes.")
    parser.add_argument("--yx", type=positive_int, default=256, help="Y and X size in pixels.")
    parser.add_argument("--pixel-size-xy-nm", type=positive_float, default=40.0)
    parser.add_argument("--pixel-size-z-nm", type=positive_float, default=100.0)
    parser.add_argument("--na", type=positive_float, default=1.4)
    parser.add_argument("--magnification", type=positive_float, default=63.0)
    parser.add_argument("--immersion-ri", type=positive_float, default=1.518)
    parser.add_argument("--sample-ri", type=positive_float, default=1.47)
    parser.add_argument("--microscope-type", choices=("confocal", "widefield"), default="confocal")
    parser.add_argument("--excitation-nm", type=positive_float, default=488.0)
    parser.add_argument("--emission-nm", type=positive_float, default=520.0)
    parser.add_argument("--pinhole-size-airy", type=positive_float, default=1.0, help="Pinhole size in Airy Disk units.")
    parser.add_argument("--snr", type=positive_float, default=10.0, help="Peak SNR for Poisson scaling.")
    parser.add_argument("--seed", type=nonnegative_int, default=12345)
    parser.add_argument("--psf-xy", type=odd_positive_int, default=129)
    parser.add_argument("--psf-z", type=odd_positive_int, default=65)
    parser.add_argument("--n-pupil", type=odd_positive_int, default=129)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> SyntheticConfig:
    return SyntheticConfig(
        output=args.output,
        z=args.z,
        yx=args.yx,
        pixel_size_xy_nm=args.pixel_size_xy_nm,
        pixel_size_z_nm=args.pixel_size_z_nm,
        na=args.na,
        magnification=args.magnification,
        immersion_ri=args.immersion_ri,
        sample_ri=args.sample_ri,
        microscope_type=args.microscope_type,
        excitation_nm=args.excitation_nm,
        emission_nm=args.emission_nm,
        pinhole_size_airy=args.pinhole_size_airy,
        snr=args.snr,
        seed=args.seed,
        psf_xy=args.psf_xy,
        psf_z=args.psf_z,
        n_pupil=args.n_pupil,
    )


def run_gui(initial: SyntheticConfig) -> int:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QSpinBox,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    class Worker(QThread):
        finished_ok = pyqtSignal(str)
        failed = pyqtSignal(str)

        def __init__(self, config: SyntheticConfig):
            super().__init__()
            self.config = config

        def run(self) -> None:
            try:
                gt_path, noisy_path = generate_pair(self.config)
            except Exception as exc:
                self.failed.emit(str(exc))
                return
            self.finished_ok.emit(f"Wrote:\n{gt_path}\n{noisy_path}")

    class Window(QMainWindow):
        def __init__(self, config: SyntheticConfig):
            super().__init__()
            self.worker: Worker | None = None
            self.setWindowTitle("Create synthetic 3D GFP ground truth")
            self.setMinimumWidth(620)
            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)

            folder_group = QGroupBox("Output")
            folder_layout = QHBoxLayout(folder_group)
            self.output_edit = QLineEdit(str(config.output))
            browse = QPushButton("Browse...")
            browse.clicked.connect(self._browse)
            folder_layout.addWidget(self.output_edit)
            folder_layout.addWidget(browse)
            layout.addWidget(folder_group)

            form_group = QGroupBox("Synthetic image and microscope parameters")
            form = QFormLayout(form_group)
            self.z = self._spin(config.z, 1, 512)
            self.yx = self._spin(config.yx, 16, 2048)
            self.pixel_xy = self._double(config.pixel_size_xy_nm, 1.0, 1000.0, 2)
            self.pixel_z = self._double(config.pixel_size_z_nm, 1.0, 5000.0, 2)
            self.na = self._double(config.na, 0.1, 2.0, 3)
            self.magnification = self._double(config.magnification, 1.0, 200.0, 1)
            self.immersion = self._double(config.immersion_ri, 1.0, 2.0, 4)
            self.sample = self._double(config.sample_ri, 1.0, 2.0, 4)
            self.microscope = QComboBox()
            self.microscope.addItems(["confocal", "widefield"])
            self.microscope.setCurrentText(config.microscope_type)
            self.excitation = self._double(config.excitation_nm, 200.0, 1000.0, 1)
            self.emission = self._double(config.emission_nm, 200.0, 1000.0, 1)
            self.pinhole = self._double(config.pinhole_size_airy, 0.1, 10.0, 2)
            self.snr = self._double(config.snr, 0.1, 1000.0, 2)
            self.seed = self._spin(config.seed, 0, 2147483647)
            self.psf_xy = self._spin(config.psf_xy, 3, 511)
            self.psf_z = self._spin(config.psf_z, 3, 511)
            self.n_pupil = self._spin(config.n_pupil, 3, 511)

            form.addRow("Z planes", self.z)
            form.addRow("Y/X pixels", self.yx)
            form.addRow("Pixel size XY (nm)", self.pixel_xy)
            form.addRow("Pixel size Z (nm)", self.pixel_z)
            form.addRow("NA", self.na)
            form.addRow("Magnification", self.magnification)
            form.addRow("Immersion RI", self.immersion)
            form.addRow("Sample RI", self.sample)
            form.addRow("Microscope type", self.microscope)
            form.addRow("Excitation (nm)", self.excitation)
            form.addRow("Emission (nm)", self.emission)
            form.addRow("Pinhole size (Airy Disk)", self.pinhole)
            form.addRow("Peak SNR", self.snr)
            form.addRow("Noise seed", self.seed)
            form.addRow("PSF XY size (odd)", self.psf_xy)
            form.addRow("PSF Z size (odd)", self.psf_z)
            form.addRow("Pupil samples (odd)", self.n_pupil)
            layout.addWidget(form_group)

            self.log = QTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumHeight(140)
            layout.addWidget(self.log)

            buttons = QHBoxLayout()
            buttons.addStretch()
            self.run_btn = QPushButton("Create images")
            self.run_btn.clicked.connect(self._run)
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self.close)
            buttons.addWidget(self.run_btn)
            buttons.addWidget(close_btn)
            layout.addLayout(buttons)

        def _spin(self, value: int, minimum: int, maximum: int) -> QSpinBox:
            box = QSpinBox()
            box.setRange(minimum, maximum)
            box.setValue(value)
            return box

        def _double(self, value: float, minimum: float, maximum: float, decimals: int) -> QDoubleSpinBox:
            box = QDoubleSpinBox()
            box.setRange(minimum, maximum)
            box.setDecimals(decimals)
            box.setValue(value)
            return box

        def _browse(self) -> None:
            folder = QFileDialog.getExistingDirectory(self, "Select output folder", self.output_edit.text())
            if folder:
                self.output_edit.setText(folder)

        def _config(self) -> SyntheticConfig:
            psf_xy = self.psf_xy.value()
            psf_z = self.psf_z.value()
            n_pupil = self.n_pupil.value()
            if psf_xy % 2 == 0:
                psf_xy += 1
            if psf_z % 2 == 0:
                psf_z += 1
            if n_pupil % 2 == 0:
                n_pupil += 1
            return SyntheticConfig(
                output=Path(self.output_edit.text()),
                z=self.z.value(),
                yx=self.yx.value(),
                pixel_size_xy_nm=self.pixel_xy.value(),
                pixel_size_z_nm=self.pixel_z.value(),
                na=self.na.value(),
                magnification=self.magnification.value(),
                immersion_ri=self.immersion.value(),
                sample_ri=self.sample.value(),
                microscope_type=self.microscope.currentText(),
                excitation_nm=self.excitation.value(),
                emission_nm=self.emission.value(),
                pinhole_size_airy=self.pinhole.value(),
                snr=self.snr.value(),
                seed=self.seed.value(),
                psf_xy=psf_xy,
                psf_z=psf_z,
                n_pupil=n_pupil,
            )

        def _run(self) -> None:
            try:
                config = self._config()
                validate_config(config)
            except Exception as exc:
                self.log.setPlainText(f"Invalid settings: {exc}")
                return
            self.run_btn.setEnabled(False)
            self.log.setPlainText("Generating synthetic object, PSF, blur, and Poisson noise...")
            self.worker = Worker(config)
            self.worker.finished_ok.connect(self._done)
            self.worker.failed.connect(self._failed)
            self.worker.start()

        def _done(self, message: str) -> None:
            self.run_btn.setEnabled(True)
            self.log.setPlainText(message)

        def _failed(self, message: str) -> None:
            self.run_btn.setEnabled(True)
            self.log.setPlainText(f"Generation failed:\n{message}")

    app = QApplication(sys.argv)
    app.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar)
    window = Window(initial)
    window.show()
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    if not args.no_gui:
        return run_gui(config)

    try:
        gt_path, noisy_path = generate_pair(config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {gt_path}")
    print(f"Wrote {noisy_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
