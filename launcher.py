"""
launcher.py — PyQt6 GUI frontend for W_CIDeconvolve_benchmark descriptor.json.

Dynamically reads descriptor.json and builds a form with appropriate
widgets for each parameter. On "Run" it executes the Docker container
in the console that launched this script.

Usage:
    python launcher.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Windows taskbar: set AppUserModelID so the taskbar shows our icon, not Python's
if sys.platform == "win32":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ci.w_cideconvolve_benchmark.launcher")

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QLineEdit,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
DESCRIPTOR_PATH = SCRIPT_DIR / "descriptor.json"
ICON_PATH = SCRIPT_DIR / "icon.svg"
LAST_SETTINGS_PATH = SCRIPT_DIR / ".last_settings.json"


class ToggleSwitch(QCheckBox):
    """Styled toggle switch using a QCheckBox with a stylesheet."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            """
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 40px;
                height: 22px;
                border-radius: 11px;
                background-color: #888;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
            }
            QCheckBox::indicator:unchecked {
                background-color: #888;
            }
            """
        )


def load_descriptor() -> dict:
    with open(DESCRIPTOR_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_docker_command(descriptor: dict, values: dict, folders: dict) -> list[str]:
    """Build the docker run command from descriptor and current widget values."""
    # Derive from descriptor, strip namespace (e.g. "cellularimagingcf/") for local run
    full_image = descriptor.get("container-image", {}).get("image", "w_cideconvolve_benchmark")
    image = full_image.rsplit("/", 1)[-1]
    name = descriptor.get("name", image)

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{folders['infolder']}:/data/in",
        "-v", f"{folders['outfolder']}:/data/out",
        "-v", f"{folders['gtfolder']}:/data/gt",
        image,
        "--infolder", "/data/in",
        "--outfolder", "/data/out",
        "--gtfolder", "/data/gt",
        "--local",
    ]

    for inp in descriptor.get("inputs", []):
        param_id = inp["id"]
        flag = inp.get("command-line-flag", f"--{param_id}")
        val = values.get(param_id)
        if val is None:
            continue
        if inp["type"] == "Boolean":
            default_val = inp.get("default-value", False)
            if default_val is True:
                # Default-true booleans use --no-<id> in argparse (store_false)
                # Only emit flag when user unchecks (wants False)
                if not val:
                    neg_flag = f"--no-{param_id.replace('_', '-')}"
                    cmd.append(neg_flag)
            else:
                # Default-false booleans use --<flag> in argparse (store_true)
                # Only emit flag when user checks (wants True)
                if val:
                    cmd.append(flag)
        else:
            cmd.extend([flag, str(val)])

    return cmd


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.descriptor = load_descriptor()
        self.widgets: dict[str, QWidget] = {}
        self._build_ui()

    def _build_ui(self):
        name = self.descriptor.get("name", "W_CIDeconvolve_benchmark")
        self.setWindowTitle(f"{name} — Launcher")
        self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.setMinimumWidth(676)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)

        # -- Header --
        title = QLabel(name)
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        desc = self.descriptor.get("description", "")
        if desc:
            desc_label = QLabel(desc)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #666; margin-bottom: 8px;")
            layout.addWidget(desc_label)

        # -- Data folders with browse buttons --
        folder_group = QGroupBox("Data Folders")
        folder_layout = QFormLayout()
        folder_group.setLayout(folder_layout)

        self.folder_widgets: dict[str, QLineEdit] = {}
        for key, label_text, default_path in [
            ("infolder",  "Input folder",  str(SCRIPT_DIR / "infolder")),
            ("outfolder", "Output folder", str(SCRIPT_DIR / "outfolder")),
        ]:
            row = QHBoxLayout()
            line = QLineEdit(default_path)
            line.setMinimumWidth(360)
            line.textChanged.connect(self._update_preview)
            browse_btn = QPushButton("Browse…")
            browse_btn.setFixedWidth(80)
            browse_btn.clicked.connect(lambda checked, le=line: self._browse_folder(le))
            row.addWidget(line)
            row.addWidget(browse_btn)
            folder_layout.addRow(label_text + ":", row)
            self.folder_widgets[key] = line

        layout.addWidget(folder_group)

        # -- Parameters from descriptor --
        param_group = QGroupBox("Parameters")
        param_layout = QFormLayout()
        param_group.setLayout(param_layout)

        for inp in self.descriptor.get("inputs", []):
            if inp.get("set-by-server", False):
                continue
            widget = self._create_widget(inp)
            if widget is not None:
                tooltip = inp.get("description", "")
                widget.setToolTip(tooltip)
                label = QLabel(inp.get("name", inp["id"]))
                label.setToolTip(tooltip)
                param_layout.addRow(label, widget)
                self.widgets[inp["id"]] = widget

        layout.addWidget(param_group)

        # -- Command preview --
        self.cmd_preview = QTextEdit()
        self.cmd_preview.setReadOnly(True)
        self.cmd_preview.setMaximumHeight(163)
        self.cmd_preview.setFont(QFont("Consolas", 9))
        self.cmd_preview.setStyleSheet("background: #1e1e1e; color: #dcdcdc;")
        layout.addWidget(QLabel("Command preview:"))
        layout.addWidget(self.cmd_preview)

        # -- Buttons --
        btn_layout = QHBoxLayout()

        restore_btn = QPushButton("Restore Last Settings")
        restore_btn.setStyleSheet("padding: 8px 16px;")
        restore_btn.setToolTip("Restore parameter values from the previous run")
        restore_btn.setEnabled(LAST_SETTINGS_PATH.exists())
        restore_btn.clicked.connect(self._on_restore)
        btn_layout.addWidget(restore_btn)

        load_btn = QPushButton("Load Settings")
        load_btn.setStyleSheet("padding: 8px 16px;")
        load_btn.setToolTip("Load parameter values from a JSON file")
        load_btn.clicked.connect(self._on_load_settings)
        btn_layout.addWidget(load_btn)

        save_btn = QPushButton("Save Settings")
        save_btn.setStyleSheet("padding: 8px 16px;")
        save_btn.setToolTip("Save current parameter values to a JSON file")
        save_btn.clicked.connect(self._on_save_settings)
        btn_layout.addWidget(save_btn)

        btn_layout.addStretch()

        run_btn = QPushButton("Run")
        run_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px 24px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        run_btn.clicked.connect(self._on_run)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("padding: 8px 24px;")
        close_btn.clicked.connect(self.close)

        btn_layout.addWidget(run_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        # Initial preview
        self._update_preview()

        # Connect all widgets for live preview updates
        for inp in self.descriptor.get("inputs", []):
            w = self.widgets.get(inp["id"])
            if w is None:
                continue
            if isinstance(w, QSpinBox):
                w.valueChanged.connect(self._update_preview)
            elif isinstance(w, QComboBox):
                w.currentTextChanged.connect(self._update_preview)
            elif isinstance(w, QCheckBox):
                w.stateChanged.connect(self._update_preview)
            elif isinstance(w, QLineEdit):
                w.textChanged.connect(self._update_preview)

    def _create_widget(self, inp: dict) -> QWidget | None:
        ptype = inp.get("type", "String")
        default = inp.get("default-value")
        choices = inp.get("value-choices")

        if ptype == "Boolean":
            toggle = ToggleSwitch()
            toggle.setChecked(bool(default))
            return toggle

        if choices:
            combo = QComboBox()
            combo.addItems([str(c) for c in choices])
            if default is not None and str(default) in [str(c) for c in choices]:
                combo.setCurrentText(str(default))
            return combo

        if ptype == "Number":
            spin = QSpinBox()
            spin.setMinimum(inp.get("minimum", 0))
            spin.setMaximum(inp.get("maximum", 99999))
            if default is not None:
                spin.setValue(int(default))
            return spin

        # Fallback: plain text
        line = QLineEdit()
        if default is not None:
            line.setText(str(default))
        return line

    def _get_values(self) -> dict:
        values = {}
        for inp in self.descriptor.get("inputs", []):
            w = self.widgets.get(inp["id"])
            if w is None:
                continue
            if isinstance(w, QCheckBox):
                values[inp["id"]] = w.isChecked()
            elif isinstance(w, QSpinBox):
                values[inp["id"]] = w.value()
            elif isinstance(w, QComboBox):
                values[inp["id"]] = w.currentText()
            elif isinstance(w, QLineEdit):
                values[inp["id"]] = w.text()
        return values

    def _get_folders(self) -> dict:
        return {
            "infolder": self.folder_widgets["infolder"].text(),
            "outfolder": self.folder_widgets["outfolder"].text(),
            "gtfolder": str(SCRIPT_DIR / "gtfolder"),
        }

    def _browse_folder(self, line_edit: QLineEdit):
        """Open a folder picker and update the given QLineEdit."""
        current = line_edit.text()
        start = current if Path(current).is_dir() else str(SCRIPT_DIR)
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", start)
        if folder:
            line_edit.setText(folder)

    def _update_preview(self):
        cmd = build_docker_command(
            self.descriptor, self._get_values(), self._get_folders()
        )
        self.cmd_preview.setPlainText(" ".join(cmd))

    def _save_settings(self):
        """Persist current widget values and folders to .last_settings.json."""
        data = {"values": self._get_values(), "folders": self._get_folders()}
        try:
            with open(LAST_SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def _apply_settings(self, data: dict):
        """Apply a settings dict (values + folders) to the widgets."""
        # Restore folders
        for key, line in self.folder_widgets.items():
            saved = data.get("folders", {}).get(key)
            if saved is not None:
                line.setText(str(saved))

        # Restore parameter values
        saved_vals = data.get("values", {})
        for inp in self.descriptor.get("inputs", []):
            w = self.widgets.get(inp["id"])
            val = saved_vals.get(inp["id"])
            if w is None or val is None:
                continue
            if isinstance(w, QCheckBox):
                w.setChecked(bool(val))
            elif isinstance(w, QSpinBox):
                w.setValue(int(val))
            elif isinstance(w, QComboBox):
                idx = w.findText(str(val))
                if idx >= 0:
                    w.setCurrentIndex(idx)
            elif isinstance(w, QLineEdit):
                w.setText(str(val))

    def _on_restore(self):
        """Load settings from .last_settings.json into the widgets."""
        try:
            with open(LAST_SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        self._apply_settings(data)

    def _on_save_settings(self):
        """Save current settings to a user-chosen JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Settings", str(SCRIPT_DIR / "settings.json"),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        data = {"values": self._get_values(), "folders": self._get_folders()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _on_load_settings(self):
        """Load settings from a user-chosen JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", str(SCRIPT_DIR),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        self._apply_settings(data)

    def _on_run(self):
        self._save_settings()
        cmd = build_docker_command(
            self.descriptor, self._get_values(), self._get_folders()
        )
        print("\n" + "=" * 70)
        print("Running:")
        print(" ".join(cmd))
        print("=" * 70 + "\n")

        self.close()

        # Run the docker command in the current console (inherits stdin/stdout)
        subprocess.run(cmd)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(str(ICON_PATH)))
    window = LauncherWindow()
    window.show()
    screen = app.primaryScreen().availableGeometry()
    window.move(
        (screen.width() - window.frameGeometry().width()) // 2,
        (screen.height() - window.frameGeometry().height()) // 2,
    )
    app.exec()


if __name__ == "__main__":
    main()
