"""gui_deconvolve_ci.py — Standalone PyQt6 GUI for CI deconvolution.

Provides a graphical interface to the ``ci_rl_deconvolve`` and
``ci_generate_psf`` functions from ``deconvolve_ci.py``.  Supports
multi-channel 3-D OME-TIFF input with per-channel PSF generation,
side-by-side input/output viewing with a shared Z-slider, and
MIP / SUM projection toggle.

Usage:
    python gui_deconvolve_ci.py
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

# Windows taskbar: set AppUserModelID so the taskbar shows our icon
if sys.platform == "win32":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        "ci.gui_deconvolve_ci"
    )

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
ICON_PATH = SCRIPT_DIR / "icon.svg"

# ---------------------------------------------------------------------------
# Channel colour helpers (same scheme as deconvolve.save_mip_png)
# ---------------------------------------------------------------------------

_FALLBACK_COLORS = [
    (0, 255, 0),      # Green
    (255, 0, 255),     # Magenta
    (0, 255, 255),     # Cyan
    (255, 0, 0),       # Red
    (0, 0, 255),       # Blue
    (255, 255, 0),     # Yellow
]

_BGRCYM = [
    (0, 0, 255),      # Blue
    (0, 255, 0),      # Green
    (255, 0, 0),      # Red
    (0, 255, 255),    # Cyan
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
]


def _emission_to_rgb(wavelength_nm: Optional[float]) -> tuple[int, int, int]:
    """Map an emission wavelength (nm) to an approximate RGB colour."""
    if wavelength_nm is None:
        return (255, 255, 255)
    wl = wavelength_nm
    r = g = b = 0.0
    if 380 <= wl < 440:
        r = -(wl - 440) / (440 - 380)
        b = 1.0
    elif 440 <= wl < 490:
        g = (wl - 440) / (490 - 440)
        b = 1.0
    elif 490 <= wl < 510:
        g = 1.0
        b = -(wl - 510) / (510 - 490)
    elif 510 <= wl < 580:
        r = (wl - 510) / (580 - 510)
        g = 1.0
    elif 580 <= wl < 645:
        r = 1.0
        g = -(wl - 645) / (645 - 580)
    elif 645 <= wl <= 780:
        r = 1.0
    else:
        return (255, 255, 255)
    return (int(r * 255), int(g * 255), int(b * 255))


def _channel_color(metadata: dict, ch_idx: int) -> tuple[int, int, int]:
    """Determine display colour for a channel from metadata."""
    channels = metadata.get("channels", [])
    em = None
    if ch_idx < len(channels):
        em = channels[ch_idx].get("emission_wavelength")
    rgb = _emission_to_rgb(em)
    if rgb == (255, 255, 255):
        rgb = _FALLBACK_COLORS[ch_idx % len(_FALLBACK_COLORS)]
    return rgb


def _resolve_channel_colors(
    metadata: dict, n_ch: int
) -> list[tuple[int, int, int]]:
    """Return a list of display colours, de-duplicating if needed."""
    colors = [_channel_color(metadata, i) for i in range(n_ch)]
    if n_ch > 1 and len(set(colors)) == 1:
        colors = [_BGRCYM[i % len(_BGRCYM)] for i in range(n_ch)]
    return colors


# ---------------------------------------------------------------------------
# Helpers: numpy 2-D arrays → QPixmap
# ---------------------------------------------------------------------------

def _composite_to_pixmap(
    slices: list[tuple[np.ndarray, tuple[int, int, int]]],
    width: int = 0,
) -> QPixmap:
    """Build an RGB composite from (2-D array, colour) pairs → QPixmap."""
    if not slices:
        return QPixmap()
    h, w_ = slices[0][0].shape
    canvas = np.zeros((h, w_, 3), dtype=np.float64)
    for arr, rgb in slices:
        ch_img = arr.astype(np.float64)
        lo, hi = float(ch_img.min()), float(ch_img.max())
        if hi - lo > 0:
            ch_img = (ch_img - lo) / (hi - lo)
        else:
            ch_img = np.zeros_like(ch_img)
        for c in range(3):
            canvas[:, :, c] += ch_img * (rgb[c] / 255.0)
    canvas = np.clip(canvas, 0, 1)
    canvas = (canvas * 255).astype(np.uint8)
    canvas = np.ascontiguousarray(canvas)
    qimg = QImage(canvas.data, w_, h, w_ * 3, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg.copy())
    if width > 0:
        pix = pix.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
    return pix


# ---------------------------------------------------------------------------
# Lightweight image loader (avoids importing deconvolve.py / torch)
# ---------------------------------------------------------------------------


def _apply_metadata_defaults(images: list, meta: dict) -> dict:
    """Track metadata source and apply defaults for missing fields."""
    meta_keys_from_file = set(meta.keys())
    ch_list = meta.get("channels", [])
    if ch_list:
        meta_keys_from_file.add("channels")
        if all("emission_wavelength" in c for c in ch_list):
            meta_keys_from_file.add("emission_wavelength")
        if all("excitation_wavelength" in c for c in ch_list):
            meta_keys_from_file.add("excitation_wavelength")

    meta.setdefault("na", 1.4)
    meta.setdefault("pixel_size_x", 0.065)
    meta.setdefault("pixel_size_z", 0.2)
    meta.setdefault("refractive_index", 1.515)
    meta.setdefault("microscope_type", "widefield")
    if "channels" not in meta:
        meta["channels"] = [{"emission_wavelength": 520.0}] * len(images)
    meta["n_channels"] = len(images)
    meta["_from_file"] = meta_keys_from_file
    return meta


def _load_image(path_str: str) -> dict:
    """Load any supported microscopy image via BioImage (bioio).

    Supports OME-TIFF, TIFF, ND2, CZI, OME-Zarr, and other bioio-
    supported formats.  Returns dict with ``'images'`` (list[ndarray])
    and ``'metadata'`` (dict).
    """
    from bioio import BioImage

    _ACQ_MODE_MAP = {
        "LASER_SCANNING_CONFOCAL_MICROSCOPY": "confocal",
        "SPINNING_DISK_CONFOCAL": "confocal",
        "SLIT_SCAN_CONFOCAL": "confocal",
        "MULTI_PHOTON_MICROSCOPY": "confocal",
        "WIDE_FIELD": "widefield",
        "OTHER": "widefield",
    }
    _IMM_RI = {
        "OIL": 1.515,
        "WATER": 1.333,
        "GLYCEROL": 1.47,
        "AIR": 1.0,
        "MULTI": 1.515,
    }

    img = BioImage(str(path_str))
    meta: dict = {}

    # Physical pixel sizes (µm)
    pps = img.physical_pixel_sizes
    if pps.X:
        meta["pixel_size_x"] = pps.X
    if pps.Z:
        meta["pixel_size_z"] = pps.Z

    # OME metadata (unified across formats via ome-types)
    try:
        ome = img.ome_metadata
        if ome and hasattr(ome, "images") and ome.images:
            im0 = ome.images[0]

            # Per-channel metadata
            ch_list = []
            for c in (im0.pixels.channels or []):
                ch_d: dict = {}
                if c.emission_wavelength is not None:
                    ch_d["emission_wavelength"] = float(c.emission_wavelength)
                if c.excitation_wavelength is not None:
                    ch_d["excitation_wavelength"] = float(c.excitation_wavelength)
                ch_list.append(ch_d)
                # Acquisition mode (use first channel's)
                if c.acquisition_mode and "microscope_type" not in meta:
                    name = getattr(c.acquisition_mode, "name",
                                   str(c.acquisition_mode))
                    meta["microscope_type"] = _ACQ_MODE_MAP.get(
                        name, "widefield")
            if ch_list:
                meta["channels"] = ch_list

            # Objective (NA, immersion → RI)
            if ome.instruments:
                for inst in ome.instruments:
                    for obj in (inst.objectives or []):
                        if obj.lens_na and "na" not in meta:
                            meta["na"] = float(obj.lens_na)
                        if obj.immersion and "refractive_index" not in meta:
                            imm_name = getattr(
                                obj.immersion, "name",
                                str(obj.immersion)).upper()
                            if imm_name in _IMM_RI:
                                meta["refractive_index"] = _IMM_RI[imm_name]

            # ObjectiveSettings — may contain explicit RI (overrides above)
            if hasattr(im0, "objective_settings") and im0.objective_settings:
                os_ = im0.objective_settings
                if os_.refractive_index:
                    meta["refractive_index"] = float(os_.refractive_index)
    except Exception:
        pass

    # Fallback: try to parse channel names as emission wavelengths
    # (e.g. OME-Zarr stores channel names like '520.0', '600.0')
    if "channels" not in meta:
        try:
            ch_names = img.channel_names or []
            ch_list = []
            for nm in ch_names:
                val = float(str(nm))
                if 300 < val < 900:
                    ch_list.append({"emission_wavelength": val})
            if ch_list:
                meta["channels"] = ch_list
        except (ValueError, TypeError):
            pass

    # ND2 fallback: native nd2 metadata for RI when ome_metadata lacks it
    ext = Path(path_str).suffix.lower()
    if ext == ".nd2" and "refractive_index" not in meta:
        try:
            import nd2
            with nd2.ND2File(str(path_str)) as f:
                chs = f.metadata.channels if f.metadata else []
                if chs and chs[0].microscope:
                    ri = chs[0].microscope.immersionRefractiveIndex
                    if ri is not None and ri > 0:
                        meta["refractive_index"] = ri
        except Exception:
            pass

    # Image data — request CZYX, first timepoint
    raw = img.get_image_data("CZYX", T=0).astype(np.float32)
    images: list[np.ndarray] = []
    if raw.ndim == 4:
        for c in range(raw.shape[0]):
            images.append(raw[c])
    elif raw.ndim == 3:
        images.append(raw)
    else:
        images.append(raw.squeeze())

    meta = _apply_metadata_defaults(images, meta)
    return {"images": images, "metadata": meta}


# ---------------------------------------------------------------------------
# Worker thread for deconvolution
# ---------------------------------------------------------------------------

class _DeconvolveWorker(QThread):
    """Run deconvolution in a background thread."""
    finished = pyqtSignal(object)  # emits list[np.ndarray] or Exception
    progress = pyqtSignal(str)

    def __init__(
        self,
        channels: list[np.ndarray],
        metadata: dict,
        params: dict,
        parent=None,
    ):
        super().__init__(parent)
        self.channels = channels
        self.metadata = metadata
        self.params = params

    def run(self):
        try:
            try:
                from deconvolve_ci import ci_generate_psf, ci_rl_deconvolve
            except OSError as e:
                raise RuntimeError(
                    f"Failed to load deconvolve_ci (torch DLL error).\n\n"
                    f"Your PyTorch installation appears broken. Try:\n"
                    f"  conda install pytorch torchvision torchaudio "
                    f"pytorch-cuda=12.1 -c pytorch -c nvidia\n\n"
                    f"Original error: {e}"
                ) from e

            results: list[np.ndarray] = []
            n_ch = len(self.channels)
            meta = self.metadata
            p = self.params

            for ci, ch_data in enumerate(self.channels):
                self.progress.emit(f"Processing channel {ci + 1}/{n_ch} …")

                # Per-channel wavelengths from GUI
                em_list = p["emission_wavelengths"]
                em_wl = em_list[ci] if ci < len(em_list) else em_list[-1] if em_list else 520.0
                ex_list = p["excitation_wavelengths"]
                ex_wl = ex_list[ci] if ci < len(ex_list) else ex_list[-1] if ex_list else None
                if p["microscope_type"] != "confocal":
                    ex_wl = None

                # PSF size: match image lateral, use odd axial
                if ch_data.ndim == 3:
                    nz_img, ny, nx = ch_data.shape
                    n_xy_psf = max(ny, nx) | 1  # make odd
                    n_z_psf = nz_img | 1
                else:
                    ny, nx = ch_data.shape
                    n_xy_psf = max(ny, nx) | 1
                    n_z_psf = 1

                self.progress.emit(
                    f"  Generating PSF (ch {ci + 1}, λ={em_wl:.0f} nm) …"
                )
                psf = ci_generate_psf(
                    na=p["na"],
                    wavelength_nm=em_wl,
                    pixel_size_xy_nm=p["pixel_size_xy_nm"],
                    pixel_size_z_nm=p["pixel_size_z_nm"],
                    n_xy=n_xy_psf,
                    n_z=n_z_psf,
                    ri_immersion=p["ri_immersion"],
                    ri_sample=p["ri_sample"],
                    ri_coverslip=p["ri_coverslip"],
                    ri_coverslip_design=p["ri_coverslip_design"],
                    ri_immersion_design=p["ri_immersion_design"],
                    t_g=p["t_g"],
                    t_g0=p["t_g0"],
                    t_i0=p["t_i0"],
                    z_p=p["z_p"],
                    microscope_type=p["microscope_type"],
                    excitation_nm=ex_wl,
                    integrate_pixels=p["integrate_pixels"],
                    n_subpixels=p["n_subpixels"],
                    n_pupil=p["n_pupil"],
                    device=p["device"],
                )

                # Crop PSF axial dimension to match image
                if psf.ndim == 3 and ch_data.ndim == 3:
                    pz = psf.shape[0]
                    iz = ch_data.shape[0]
                    if pz > iz:
                        start = (pz - iz) // 2
                        psf = psf[start : start + iz]

                self.progress.emit(
                    f"  Deconvolving (ch {ci + 1}, {p['niter']} iter) …"
                )
                out = ci_rl_deconvolve(
                    ch_data,
                    psf,
                    niter=p["niter"],
                    tv_lambda=p["tv_lambda"],
                    background=p["background"],
                    convergence=p["convergence"],
                    rel_threshold=p["rel_threshold"],
                    check_every=p["check_every"],
                    device=p["device"],
                    tiling=p["tiling"],
                    max_tile_xy=p["max_tile_xy"],
                    max_tile_z=p["max_tile_z"],
                )
                results.append(out["result"].copy())

            self.finished.emit(results)
        except Exception as exc:
            traceback.print_exc()
            self.finished.emit(exc)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

def _detect_gpu_info() -> str:
    """Return a short string with torch + GPU version info for the title bar."""
    try:
        import torch
        parts = [f"torch {torch.__version__}"]
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda or "?"
            parts.append(f"{name}  CUDA {cuda_ver}")
        else:
            parts.append("CPU only")
        return "  |  ".join(parts)
    except Exception:
        return "torch not available"


class DeconvolveCIWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        gpu_info = _detect_gpu_info()
        self.setWindowTitle(f"CI Deconvolve — {gpu_info}")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.setMinimumSize(1430, 700)

        # State
        self._input_channels: list[np.ndarray] = []
        self._output_channels: list[np.ndarray] = []
        self._metadata: dict = {}
        self._worker: Optional[_DeconvolveWorker] = None
        self._input_path: Optional[Path] = None

        self._build_ui()
        self._update_viewer()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        # ---- Left: controls (scrollable) ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(440)
        scroll.setMaximumWidth(500)
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setSpacing(6)
        scroll.setWidget(ctrl_widget)
        splitter.addWidget(scroll)

        # Title
        title = QLabel("CI Deconvolve")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        ctrl_layout.addWidget(title)

        # --- Method ---
        method_group = QGroupBox("Method")
        ml = QFormLayout()
        method_group.setLayout(ml)

        self._method_combo = QComboBox()
        self._method_combo.addItems(["ci_rl", "ci_rl_tv"])
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        ml.addRow("Method:", self._method_combo)

        self._sp_niter = QSpinBox()
        self._sp_niter.setRange(1, 10000)
        self._sp_niter.setValue(50)
        ml.addRow("Iterations:", self._sp_niter)

        self._sp_tv_lambda = QDoubleSpinBox()
        self._sp_tv_lambda.setRange(0.0, 1.0)
        self._sp_tv_lambda.setDecimals(6)
        self._sp_tv_lambda.setSingleStep(0.0001)
        self._sp_tv_lambda.setValue(0.001)
        self._tv_lambda_label = ml.labelForField(self._sp_tv_lambda)  # type: ignore
        ml.addRow("TV lambda:", self._sp_tv_lambda)
        self._tv_lambda_row_label = "TV lambda:"

        self._bg_combo = QComboBox()
        self._bg_combo.addItems(["auto", "manual"])
        self._bg_combo.currentTextChanged.connect(self._on_bg_changed)
        ml.addRow("Background:", self._bg_combo)

        self._sp_bg_value = QDoubleSpinBox()
        self._sp_bg_value.setRange(0.0, 1e9)
        self._sp_bg_value.setDecimals(2)
        self._sp_bg_value.setValue(0.0)
        self._sp_bg_value.setEnabled(False)
        ml.addRow("BG value:", self._sp_bg_value)

        self._conv_combo = QComboBox()
        self._conv_combo.addItems(["fixed", "auto"])
        self._conv_combo.currentTextChanged.connect(self._on_conv_changed)
        ml.addRow("Convergence:", self._conv_combo)

        self._sp_rel_thresh = QDoubleSpinBox()
        self._sp_rel_thresh.setRange(1e-8, 1.0)
        self._sp_rel_thresh.setDecimals(6)
        self._sp_rel_thresh.setSingleStep(0.0001)
        self._sp_rel_thresh.setValue(0.001)
        self._sp_rel_thresh.setEnabled(False)
        ml.addRow("Rel. threshold:", self._sp_rel_thresh)

        self._sp_check_every = QSpinBox()
        self._sp_check_every.setRange(1, 1000)
        self._sp_check_every.setValue(5)
        self._sp_check_every.setEnabled(False)
        ml.addRow("Check every:", self._sp_check_every)

        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "cuda", "cpu"])
        ml.addRow("Device:", self._device_combo)

        ctrl_layout.addWidget(method_group)

        # --- Tiling ---
        tiling_group = QGroupBox("Tiling")
        tl = QFormLayout()
        tiling_group.setLayout(tl)

        self._tiling_combo = QComboBox()
        self._tiling_combo.addItems(["custom", "none"])
        self._tiling_combo.currentTextChanged.connect(self._on_tiling_changed)
        tl.addRow("Tiling:", self._tiling_combo)

        self._sp_max_tile_xy = QSpinBox()
        self._sp_max_tile_xy.setRange(64, 4096)
        self._sp_max_tile_xy.setSingleStep(64)
        self._sp_max_tile_xy.setValue(512)
        tl.addRow("Max tile XY:", self._sp_max_tile_xy)

        self._sp_max_tile_z = QSpinBox()
        self._sp_max_tile_z.setRange(8, 512)
        self._sp_max_tile_z.setSingleStep(8)
        self._sp_max_tile_z.setValue(64)
        tl.addRow("Max tile Z:", self._sp_max_tile_z)

        ctrl_layout.addWidget(tiling_group)

        # --- Optics / PSF ---
        optics_group = QGroupBox("Optics / PSF")
        ol = QFormLayout()
        optics_group.setLayout(ol)

        self._sp_na = QDoubleSpinBox()
        self._sp_na.setRange(0.1, 2.0)
        self._sp_na.setDecimals(3)
        self._sp_na.setSingleStep(0.05)
        self._sp_na.setValue(1.4)
        ol.addRow("NA:", self._sp_na)

        self._le_emission = QLineEdit("520")
        self._le_emission.setToolTip(
            "Emission wavelength(s) in nm, comma-separated per channel."
        )
        ol.addRow("Emission (nm):", self._le_emission)

        self._sp_px_xy = QDoubleSpinBox()
        self._sp_px_xy.setRange(1.0, 10000.0)
        self._sp_px_xy.setDecimals(3)
        self._sp_px_xy.setSingleStep(1.0)
        self._sp_px_xy.setValue(65.0)
        ol.addRow("Pixel XY (nm):", self._sp_px_xy)

        self._sp_px_z = QDoubleSpinBox()
        self._sp_px_z.setRange(1.0, 50000.0)
        self._sp_px_z.setDecimals(3)
        self._sp_px_z.setSingleStep(10.0)
        self._sp_px_z.setValue(200.0)
        ol.addRow("Pixel Z (nm):", self._sp_px_z)

        self._micro_combo = QComboBox()
        self._micro_combo.addItems(["widefield", "confocal"])
        self._micro_combo.currentTextChanged.connect(self._on_micro_changed)
        ol.addRow("Microscope:", self._micro_combo)

        self._le_excitation = QLineEdit("488")
        self._le_excitation.setToolTip(
            "Excitation wavelength(s) in nm, comma-separated per channel.\n"
            "Used only for confocal PSF generation."
        )
        self._le_excitation.setEnabled(False)
        ol.addRow("Excitation (nm):", self._le_excitation)

        ctrl_layout.addWidget(optics_group)

        # --- Refractive Indices ---
        ri_group = QGroupBox("Refractive Indices")
        rl = QFormLayout()
        ri_group.setLayout(rl)

        self._sp_ri_imm = QDoubleSpinBox()
        self._sp_ri_imm.setRange(1.0, 2.0)
        self._sp_ri_imm.setDecimals(4)
        self._sp_ri_imm.setSingleStep(0.001)
        self._sp_ri_imm.setValue(1.515)
        rl.addRow("RI immersion:", self._sp_ri_imm)

        self._sp_ri_sample = QDoubleSpinBox()
        self._sp_ri_sample.setRange(1.0, 2.0)
        self._sp_ri_sample.setDecimals(4)
        self._sp_ri_sample.setSingleStep(0.001)
        self._sp_ri_sample.setValue(1.33)
        rl.addRow("RI sample:", self._sp_ri_sample)

        self._sp_ri_cover = QDoubleSpinBox()
        self._sp_ri_cover.setRange(1.0, 2.0)
        self._sp_ri_cover.setDecimals(4)
        self._sp_ri_cover.setSingleStep(0.001)
        self._sp_ri_cover.setValue(1.5255)
        rl.addRow("RI coverslip:", self._sp_ri_cover)

        self._sp_ri_cover_d = QDoubleSpinBox()
        self._sp_ri_cover_d.setRange(1.0, 2.0)
        self._sp_ri_cover_d.setDecimals(4)
        self._sp_ri_cover_d.setSingleStep(0.001)
        self._sp_ri_cover_d.setValue(1.5255)
        rl.addRow("RI coverslip (design):", self._sp_ri_cover_d)

        self._sp_ri_imm_d = QDoubleSpinBox()
        self._sp_ri_imm_d.setRange(1.0, 2.0)
        self._sp_ri_imm_d.setDecimals(4)
        self._sp_ri_imm_d.setSingleStep(0.001)
        self._sp_ri_imm_d.setValue(1.515)
        rl.addRow("RI immersion (design):", self._sp_ri_imm_d)

        ctrl_layout.addWidget(ri_group)

        # --- Coverslip / depths ---
        cov_group = QGroupBox("Coverslip / Depth")
        cl = QFormLayout()
        cov_group.setLayout(cl)

        self._sp_tg = QDoubleSpinBox()
        self._sp_tg.setRange(0, 1e7)
        self._sp_tg.setDecimals(0)
        self._sp_tg.setSingleStep(1000)
        self._sp_tg.setValue(170000)
        self._sp_tg.setSuffix(" nm")
        cl.addRow("Coverslip thickness:", self._sp_tg)

        self._sp_tg0 = QDoubleSpinBox()
        self._sp_tg0.setRange(0, 1e7)
        self._sp_tg0.setDecimals(0)
        self._sp_tg0.setSingleStep(1000)
        self._sp_tg0.setValue(170000)
        self._sp_tg0.setSuffix(" nm")
        cl.addRow("Coverslip thickness (design):", self._sp_tg0)

        self._sp_ti0 = QDoubleSpinBox()
        self._sp_ti0.setRange(0, 1e7)
        self._sp_ti0.setDecimals(0)
        self._sp_ti0.setSingleStep(1000)
        self._sp_ti0.setValue(100000)
        self._sp_ti0.setSuffix(" nm")
        cl.addRow("Immersion thickness (design):", self._sp_ti0)

        self._sp_zp = QDoubleSpinBox()
        self._sp_zp.setRange(0, 1e7)
        self._sp_zp.setDecimals(0)
        self._sp_zp.setSingleStep(100)
        self._sp_zp.setValue(0)
        self._sp_zp.setSuffix(" nm")
        cl.addRow("Particle depth (z_p):", self._sp_zp)

        ctrl_layout.addWidget(cov_group)

        # --- PSF advanced ---
        psf_group = QGroupBox("PSF Advanced")
        pl = QFormLayout()
        psf_group.setLayout(pl)

        self._cb_integrate = QCheckBox()
        self._cb_integrate.setChecked(True)
        pl.addRow("Pixel integration:", self._cb_integrate)

        self._sp_subpixels = QSpinBox()
        self._sp_subpixels.setRange(1, 9)
        self._sp_subpixels.setValue(3)
        pl.addRow("Sub-pixels:", self._sp_subpixels)

        self._sp_n_pupil = QSpinBox()
        self._sp_n_pupil.setRange(33, 513)
        self._sp_n_pupil.setSingleStep(2)
        self._sp_n_pupil.setValue(129)
        pl.addRow("Pupil samples:", self._sp_n_pupil)

        ctrl_layout.addWidget(psf_group)

        ctrl_layout.addStretch()

        # ---- Right: image viewer ----
        viewer = QWidget()
        vl = QVBoxLayout(viewer)
        vl.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(viewer)

        # Channel toggle buttons (per-channel, coloured)
        self._ch_bar = QHBoxLayout()
        self._ch_bar.addWidget(QLabel("Channels:"))
        self._ch_toggles: list[QPushButton] = []
        self._ch_bar.addStretch()
        vl.addLayout(self._ch_bar)

        # Image panels
        panels = QHBoxLayout()
        vl.addLayout(panels, stretch=1)

        # Input panel
        inp_vl = QVBoxLayout()
        inp_vl.addWidget(QLabel("Input"))
        self._lbl_input = QLabel()
        self._lbl_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_input.setMinimumSize(256, 256)
        self._lbl_input.setStyleSheet("background-color: #1a1a1a;")
        inp_vl.addWidget(self._lbl_input, stretch=1)
        panels.addLayout(inp_vl)

        # Output panel
        out_vl = QVBoxLayout()
        out_vl.addWidget(QLabel("Output"))
        self._lbl_output = QLabel()
        self._lbl_output.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_output.setMinimumSize(256, 256)
        self._lbl_output.setStyleSheet("background-color: #1a1a1a;")
        out_vl.addWidget(self._lbl_output, stretch=1)
        panels.addLayout(out_vl)

        # Z-slider + projection toggle
        nav = QHBoxLayout()

        self._z_slider = QSlider(Qt.Orientation.Horizontal)
        self._z_slider.setMinimum(0)
        self._z_slider.setMaximum(0)
        self._z_slider.valueChanged.connect(self._update_viewer)
        nav.addWidget(QLabel("Z:"))
        nav.addWidget(self._z_slider, stretch=1)

        self._z_label = QLabel("0/0")
        self._z_label.setMinimumWidth(50)
        nav.addWidget(self._z_label)

        self._proj_combo = QComboBox()
        self._proj_combo.addItems(["Slice", "MIP", "SUM"])
        self._proj_combo.currentTextChanged.connect(self._on_proj_changed)
        nav.addWidget(QLabel("View:"))
        nav.addWidget(self._proj_combo)

        vl.addLayout(nav)

        # --- Bottom panel: Open / Run / Save ---
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 4, 0, 0)

        btn_open = QPushButton("Open\u2026")
        btn_open.clicked.connect(self._on_open)
        bottom.addWidget(btn_open)

        btn_open_zarr = QPushButton("Open Zarr\u2026")
        btn_open_zarr.clicked.connect(self._on_open_zarr)
        bottom.addWidget(btn_open_zarr)

        self._file_label = QLabel("No file loaded")
        self._file_label.setWordWrap(False)
        bottom.addWidget(self._file_label, stretch=1)

        self._btn_run = QPushButton("Run Deconvolution")
        self._btn_run.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px; }"
        )
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._on_run)
        bottom.addWidget(self._btn_run)

        self._btn_save = QPushButton("Save Result\u2026")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save)
        bottom.addWidget(self._btn_save)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setVisible(False)
        self._progress.setMaximumWidth(120)
        bottom.addWidget(self._progress)

        root.addLayout(bottom)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)

        # Splitter proportions
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Initial method state
        self._on_method_changed(self._method_combo.currentText())

    # -----------------------------------------------------------------------
    # Slots — control panel
    # -----------------------------------------------------------------------

    def _on_method_changed(self, text: str):
        is_tv = text == "ci_rl_tv"
        self._sp_tv_lambda.setEnabled(is_tv)
        if not is_tv:
            self._sp_tv_lambda.setValue(0.0)
        else:
            if self._sp_tv_lambda.value() == 0.0:
                self._sp_tv_lambda.setValue(0.001)

    def _on_bg_changed(self, text: str):
        self._sp_bg_value.setEnabled(text == "manual")

    def _on_conv_changed(self, text: str):
        auto = text == "auto"
        self._sp_rel_thresh.setEnabled(auto)
        self._sp_check_every.setEnabled(auto)

    def _on_micro_changed(self, text: str):
        self._le_excitation.setEnabled(text == "confocal")

    def _on_tiling_changed(self, text: str):
        enabled = text == "custom"
        self._sp_max_tile_xy.setEnabled(enabled)
        self._sp_max_tile_z.setEnabled(enabled)

    def _on_proj_changed(self, text: str):
        self._z_slider.setEnabled(text == "Slice")
        self._update_viewer()

    # -----------------------------------------------------------------------
    # File open
    # -----------------------------------------------------------------------

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.ome.tiff *.ome.tif *.tiff *.tif *.nd2 *.czi);;All Files (*)",
        )
        if path:
            self._do_load(path)

    def _on_open_zarr(self):
        path = QFileDialog.getExistingDirectory(
            self, "Open OME-Zarr Folder", "",
        )
        if path:
            self._do_load(path)

    def _do_load(self, path: str):
        self._status.showMessage(f"Loading {Path(path).name} …")
        QApplication.processEvents()

        try:
            data = _load_image(path)
            self._input_channels = data["images"]
            self._metadata = data["metadata"]
            self._output_channels = []
            self._input_path = Path(path)

            # Populate UI from metadata
            meta = self._metadata
            from_file = meta.get("_from_file", set())

            def _bg(found: bool) -> str:
                """Stylesheet snippet: green if from metadata, orange if default."""
                if found:
                    return "background-color: #e6ffe6; color: black;"   # pastel green
                return "background-color: #fff0d0; color: black;"       # pastel orange

            if meta.get("na"):
                self._sp_na.setValue(float(meta["na"]))
            self._sp_na.setStyleSheet(_bg("na" in from_file))
            px_x = meta.get("pixel_size_x")
            if px_x:
                self._sp_px_xy.setValue(float(px_x) * 1000.0)  # µm → nm
            self._sp_px_xy.setStyleSheet(_bg("pixel_size_x" in from_file))
            px_z = meta.get("pixel_size_z")
            if px_z:
                self._sp_px_z.setValue(float(px_z) * 1000.0)
            self._sp_px_z.setStyleSheet(_bg("pixel_size_z" in from_file))
            ri = meta.get("refractive_index")
            if ri:
                self._sp_ri_imm.setValue(float(ri))
                self._sp_ri_imm_d.setValue(float(ri))
            self._sp_ri_imm.setStyleSheet(_bg("refractive_index" in from_file))
            self._sp_ri_imm_d.setStyleSheet(_bg("refractive_index" in from_file))
            micro = meta.get("microscope_type")
            if micro:
                idx = self._micro_combo.findText(micro)
                if idx >= 0:
                    self._micro_combo.setCurrentIndex(idx)
            self._micro_combo.setStyleSheet(_bg("microscope_type" in from_file))

            # Per-channel wavelengths
            ch_info = meta.get("channels", [])
            if ch_info:
                em_vals = [str(c.get("emission_wavelength", 520))
                           for c in ch_info]
                self._le_emission.setText(", ".join(em_vals))
                ex_vals = [str(c.get("excitation_wavelength", 488))
                           for c in ch_info]
                self._le_excitation.setText(", ".join(ex_vals))
            self._le_emission.setStyleSheet(
                _bg("emission_wavelength" in from_file))
            self._le_excitation.setStyleSheet(
                _bg("excitation_wavelength" in from_file))

            # RI sample is never in metadata — red (needs user input)
            self._sp_ri_sample.setStyleSheet(
                "background-color: #ffe0e0; color: black;")  # pastel red
            # Coverslip RI and design params — always defaults, orange
            self._sp_ri_cover.setStyleSheet(_bg(False))
            self._sp_ri_cover_d.setStyleSheet(_bg(False))

            # Channel toggle buttons
            self._rebuild_channel_toggles()

            # Z-slider
            first = self._input_channels[0]
            if first.ndim == 3:
                self._z_slider.setMaximum(first.shape[0] - 1)
                self._z_slider.setValue(first.shape[0] // 2)
            else:
                self._z_slider.setMaximum(0)
                self._z_slider.setValue(0)

            stem = Path(path).name
            shape = self._input_channels[0].shape
            n_ch = len(self._input_channels)
            self._file_label.setText(
                f"{stem}\n{n_ch} ch, shape={shape}"
            )
            self._btn_run.setEnabled(True)
            self._btn_save.setEnabled(False)
            self._update_viewer()
            self._status.showMessage(f"Loaded {stem}", 5000)

        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            self._status.showMessage("Load failed", 5000)

    # -----------------------------------------------------------------------
    # Run deconvolution
    # -----------------------------------------------------------------------

    def _collect_params(self) -> dict:
        device_text = self._device_combo.currentText()
        device = None if device_text == "auto" else device_text

        bg_text = self._bg_combo.currentText()
        background: str | float = "auto"
        if bg_text == "manual":
            background = self._sp_bg_value.value()

        em_list = [float(s.strip()) for s in self._le_emission.text().split(",")
                   if s.strip()]
        ex_list = [float(s.strip()) for s in self._le_excitation.text().split(",")
                   if s.strip()]

        return {
            "niter": self._sp_niter.value(),
            "tv_lambda": self._sp_tv_lambda.value(),
            "background": background,
            "convergence": self._conv_combo.currentText(),
            "rel_threshold": self._sp_rel_thresh.value(),
            "check_every": self._sp_check_every.value(),
            "device": device,
            "na": self._sp_na.value(),
            "emission_wavelengths": em_list,
            "excitation_wavelengths": ex_list,
            "pixel_size_xy_nm": self._sp_px_xy.value(),
            "pixel_size_z_nm": self._sp_px_z.value(),
            "ri_immersion": self._sp_ri_imm.value(),
            "ri_sample": self._sp_ri_sample.value(),
            "ri_coverslip": self._sp_ri_cover.value(),
            "ri_coverslip_design": self._sp_ri_cover_d.value(),
            "ri_immersion_design": self._sp_ri_imm_d.value(),
            "t_g": self._sp_tg.value(),
            "t_g0": self._sp_tg0.value(),
            "t_i0": self._sp_ti0.value(),
            "z_p": self._sp_zp.value(),
            "microscope_type": self._micro_combo.currentText(),
            "integrate_pixels": self._cb_integrate.isChecked(),
            "n_subpixels": self._sp_subpixels.value(),
            "n_pupil": self._sp_n_pupil.value(),
            "tiling": self._tiling_combo.currentText(),
            "max_tile_xy": self._sp_max_tile_xy.value(),
            "max_tile_z": self._sp_max_tile_z.value(),
        }

    def _on_run(self):
        if not self._input_channels:
            return

        self._btn_run.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._progress.setVisible(True)
        self._status.showMessage("Running deconvolution …")

        params = self._collect_params()
        self._worker = _DeconvolveWorker(
            self._input_channels, self._metadata, params, parent=self
        )
        self._worker.progress.connect(lambda msg: self._status.showMessage(msg))
        self._worker.finished.connect(self._on_deconv_done)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_deconv_done(self, result):
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)

        if isinstance(result, Exception):
            self._worker = None
            QMessageBox.critical(self, "Deconvolution Error", str(result))
            self._status.showMessage("Deconvolution failed", 5000)
            return

        try:
            self._output_channels = result
            self._btn_save.setEnabled(True)
            self._update_viewer()
            self._status.showMessage("Deconvolution complete", 5000)
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(self, "Viewer Error", str(exc))
        finally:
            self._worker = None

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    def _on_save(self):
        if not self._output_channels:
            return

        stem = self._input_path.stem if self._input_path else "deconvolved"
        method = self._method_combo.currentText()
        niter = self._sp_niter.value()
        suggested = f"{stem}_{method}_{niter}i.ome.tiff"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Deconvolved Image",
            suggested,
            "OME-TIFF (*.ome.tiff);;TIFF (*.tiff *.tif)",
        )
        if not path:
            return

        try:
            from bioio.writers import OmeTiffWriter

            channels = self._output_channels
            # OmeTiffWriter expects TCZYX (5-D)
            stack = np.stack(channels, axis=0)  # CZYX
            data = stack[np.newaxis, ...].astype(np.float32)  # TCZYX
            if data.ndim == 4:
                data = data[:, :, np.newaxis, :, :]  # ensure 5-D (add Z)

            # Collect physical pixel sizes from current GUI state
            meta = self._metadata or {}
            px_x = meta.get("pixel_size_x")
            px_z = meta.get("pixel_size_z")
            physical_pixel_sizes = None
            if px_x or px_z:
                from bioio_base.types import PhysicalPixelSizes
                physical_pixel_sizes = PhysicalPixelSizes(
                    Z=px_z or 1.0, Y=px_x or 1.0, X=px_x or 1.0
                )

            # Channel names
            ch_names = meta.get("channel_names")
            if not ch_names:
                ch_info = meta.get("channels", [])
                ch_names = [
                    f"Ch{i} em={c.get('emission_wavelength', '?')}"
                    for i, c in enumerate(ch_info)
                ]

            OmeTiffWriter.save(
                data,
                path,
                dim_order="TCZYX",
                physical_pixel_sizes=physical_pixel_sizes,
                channel_names=ch_names[:len(channels)],
            )
            self._status.showMessage(f"Saved → {Path(path).name}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    # -----------------------------------------------------------------------
    # Viewer
    # -----------------------------------------------------------------------

    def _rebuild_channel_toggles(self):
        """Create one toggle button per channel in the viewer bar."""
        # Remove old toggles
        for btn in self._ch_toggles:
            self._ch_bar.removeWidget(btn)
            btn.deleteLater()
        self._ch_toggles.clear()

        n_ch = len(self._input_channels)
        colors = _resolve_channel_colors(self._metadata, n_ch)
        ch_names = self._metadata.get("channel_names", [])

        for i in range(n_ch):
            name = ch_names[i] if i < len(ch_names) else f"Ch {i}"
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setChecked(True)
            r, g, b = colors[i]
            btn.setStyleSheet(
                f"QPushButton {{ color: rgb({r},{g},{b}); font-weight: bold; "
                f"border: 2px solid rgb({r},{g},{b}); padding: 2px 8px; }}"
                f"QPushButton:checked {{ background-color: rgba({r},{g},{b},60); }}"
            )
            btn.toggled.connect(self._update_viewer)
            # Insert before the trailing stretch
            self._ch_bar.insertWidget(self._ch_bar.count() - 1, btn)
            self._ch_toggles.append(btn)

    def _get_slice(
        self, channels: list[np.ndarray], ch_idx: int
    ) -> Optional[np.ndarray]:
        """Return the 2-D slice/projection to display."""
        if not channels or ch_idx >= len(channels):
            return None
        vol = channels[ch_idx]
        mode = self._proj_combo.currentText()

        if vol.ndim == 2:
            return vol

        if mode == "MIP":
            return vol.max(axis=0)
        elif mode == "SUM":
            return vol.sum(axis=0)
        else:  # Slice
            z = self._z_slider.value()
            z = min(z, vol.shape[0] - 1)
            return vol[z]

    def _composite_pixmap(
        self, channels: list[np.ndarray], width: int
    ) -> Optional[QPixmap]:
        """Build an RGB composite from enabled channels."""
        if not channels:
            return None
        n_ch = len(channels)
        colors = _resolve_channel_colors(self._metadata, n_ch)
        slices: list[tuple[np.ndarray, tuple[int, int, int]]] = []
        for i in range(n_ch):
            if i < len(self._ch_toggles) and not self._ch_toggles[i].isChecked():
                continue
            s = self._get_slice(channels, i)
            if s is not None:
                slices.append((s, colors[i]))
        if not slices:
            return None
        return _composite_to_pixmap(slices, width)

    def _update_viewer(self):
        # Panel size for scaling
        pw = max(self._lbl_input.width(), 256)

        # Input composite
        pix_in = self._composite_pixmap(self._input_channels, pw)
        if pix_in is not None:
            self._lbl_input.setPixmap(pix_in)
        else:
            self._lbl_input.clear()

        # Output composite
        pix_out = self._composite_pixmap(self._output_channels, pw)
        if pix_out is not None:
            self._lbl_output.setPixmap(pix_out)
        else:
            self._lbl_output.clear()

        # Z label
        if self._input_channels and self._input_channels[0].ndim == 3:
            nz = self._input_channels[0].shape[0]
            z = self._z_slider.value()
            self._z_label.setText(f"{z}/{nz - 1}")
        else:
            self._z_label.setText("—")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_viewer()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _excepthook(exc_type, exc_value, exc_tb):
    """Show uncaught exceptions in a dialog instead of silently crashing."""
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    log.critical("Uncaught exception:\n%s", msg)
    print(msg, file=sys.stderr, flush=True)
    QMessageBox.critical(None, "Fatal Error", msg)


def main():
    sys.excepthook = _excepthook

    app = QApplication(sys.argv)
    app.setApplicationName("CI Deconvolve")
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))

    window = DeconvolveCIWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
