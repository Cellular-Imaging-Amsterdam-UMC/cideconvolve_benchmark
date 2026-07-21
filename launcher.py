from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml

if sys.platform == "win32":
    import ctypes

    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        "ci.w_cideconvolve_benchmark.bilayers_launcher"
    )

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.yaml"
ICON = ROOT / "icon.svg"
SETTINGS = ROOT / ".last_launcher_settings.json"


class MultiSelectList(QListWidget):
    def __init__(self, options: list[dict], selected: list[str]):
        super().__init__()
        selected_values = set(selected)
        self.setMinimumHeight(120)
        self.setMaximumHeight(160)
        for option in options:
            item = QListWidgetItem(str(option["label"]))
            item.setData(Qt.ItemDataRole.UserRole, option["value"])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if option["value"] in selected_values
                else Qt.CheckState.Unchecked
            )
            self.addItem(item)

    def values(self) -> list[str]:
        return [
            str(self.item(index).data(Qt.ItemDataRole.UserRole))
            for index in range(self.count())
            if self.item(index).checkState() == Qt.CheckState.Checked
        ]


class NoWheelComboBox(QComboBox):
    """A selector that never changes its value while the form is scrolled."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API name
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    """An integer editor that ignores mouse-wheel value changes."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API name
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """A floating-point editor that ignores mouse-wheel value changes."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API name
        event.ignore()


class CollapsiblePanel(QWidget):
    def __init__(self, title: str):
        super().__init__()
        self._collapsed_window_height: int | None = None
        self.toggle = QToolButton(text=title, checkable=True, checked=False)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.clicked.connect(self._set_open)
        self.content = QWidget()
        self.content.setVisible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle)
        layout.addWidget(self.content)

    def _set_open(self, opened: bool) -> None:
        window = self.window()
        if opened and not window.isMaximized() and not window.isFullScreen():
            self._collapsed_window_height = window.height()
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if opened else Qt.ArrowType.RightArrow
        )
        self.content.setVisible(opened)
        self.content.updateGeometry()
        self.updateGeometry()
        # Visibility changes are not fully reflected in the top-level size hint
        # until Qt has completed the current event. Resizing immediately with
        # adjustSize() can therefore retain the expanded height after collapse,
        # and the vertical layout distributes that surplus inside every panel.
        target_height = None if opened else self._collapsed_window_height
        QTimer.singleShot(
            0, lambda: self._fit_window_to_visible_content(target_height, retry=True)
        )

    def _fit_window_to_visible_content(
        self, target_height: int | None = None, *, retry: bool = False
    ) -> None:
        window = self.window()
        if window.isMaximized() or window.isFullScreen():
            return
        own_layout = self.layout()
        if own_layout is not None:
            own_layout.activate()
        central = getattr(window, "centralWidget", lambda: None)()
        if central is not None and central.layout() is not None:
            central.layout().activate()
            central.updateGeometry()
        window.updateGeometry()
        hint = window.sizeHint()
        if target_height is not None:
            window.resize(window.width(), target_height)
        elif hint.isValid():
            window.resize(max(window.width(), hint.width()), hint.height())
        if retry:
            # QMainWindow's minimum-size cache is updated one event after its
            # central layout. A second queued resize is what permits shrinking
            # all the way back to the pre-expansion height on Windows.
            QTimer.singleShot(
                0, lambda: self._fit_window_to_visible_content(target_height)
            )


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _append_parameters(command: list[str], config: dict, values: dict) -> list[str]:
    for item in sorted(
        config.get("parameters", []), key=lambda entry: int(entry.get("cli_order", 0))
    ):
        value = values.get(item["name"], item.get("default"))
        passes_boolean_value = item.get("type") == "checkbox" and item.get(
            "append_value", False
        )
        if value in (None, "", []) or (value is False and not passes_boolean_value):
            continue
        command.append(str(item["cli_tag"]))
        if item.get("type") != "checkbox" or item.get("append_value", False):
            if isinstance(value, list):
                value = ",".join(value)
            command.append(str(value))
    command.append("--local")
    return command


def build_docker_command(
    config: dict, values: dict, input_dir: str, output_dir: str, gpu: bool = True
) -> list[str]:
    image = config["docker_image"]
    # The desktop launcher runs the image produced by builddocker.cmd. The
    # organization-qualified image in config.yaml remains the BIOMERO registry
    # identity and must not make local Docker runs pull from Docker Hub.
    image_name = f"{image['name']}:{image.get('tag', 'latest')}"
    command = ["docker", "run", "--rm"]
    if gpu:
        command += ["--gpus", "all"]
    command += [
        "-v",
        f"{input_dir}:/data/in",
        "-v",
        f"{output_dir}:/data/out",
        image_name,
    ]
    return _append_parameters(command, config, values)


def build_local_command(
    config: dict,
    values: dict,
    input_dir: str,
    output_dir: str,
    python_executable: str | None = None,
) -> list[str]:
    command = [
        python_executable or sys.executable,
        str(ROOT / "wrapper.py"),
        "--infolder",
        input_dir,
        "--outfolder",
        output_dir,
    ]
    return _append_parameters(command, config, values)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.widgets: dict[str, QWidget] = {}
        self.run_buttons: list[QPushButton] = []
        self.setWindowTitle("CIDeconvolve Benchmark - Bilayers Launcher")
        self.setWindowIcon(QIcon(str(ICON)))
        self.setMinimumWidth(960)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(10)

        title = QLabel("CIDeconvolve Benchmark - Bilayers")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        folders = QGroupBox("Data folders")
        folder_form = QFormLayout(folders)
        self.input_path = QLineEdit(str(ROOT / "infolder"))
        self.output_path = QLineEdit(str(ROOT / "outfolder"))
        folder_form.addRow("Input folder:", self._folder_row(self.input_path))
        folder_form.addRow("Output folder:", self._folder_row(self.output_path))
        layout.addWidget(folders)

        runtime = QGroupBox("Docker runtime")
        runtime_form = QFormLayout(runtime)
        self.gpu = QCheckBox("Expose NVIDIA GPU to container")
        self.gpu.setChecked(True)
        self.gpu.setToolTip("Adds '--gpus all' to the Docker command.")
        runtime_form.addRow("GPU:", self.gpu)
        layout.addWidget(runtime)

        parameters = QGroupBox("Parameters")
        parameter_layout = QVBoxLayout(parameters)
        main = QWidget()
        main_grid = self._parameter_grid(main)
        advanced = CollapsiblePanel("Advanced parameters")
        self.advanced_panel = advanced
        advanced_grid = self._parameter_grid(advanced.content, left_margin=18)
        main_count = advanced_count = 0
        for spec in self.config.get("parameters", []):
            widget = self._widget(spec)
            widget.setToolTip(spec.get("description", ""))
            label = QLabel(spec.get("label", spec["name"]))
            label.setToolTip(spec.get("description", ""))
            label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            if spec.get("mode") == "advanced":
                self._add_two_column_row(advanced_grid, advanced_count, label, widget)
                advanced_count += 1
            else:
                self._add_two_column_row(main_grid, main_count, label, widget)
                main_count += 1
            self.widgets[spec["name"]] = widget
        parameter_layout.addWidget(main)
        if advanced_count:
            parameter_layout.addWidget(advanced)
        layout.addWidget(parameters)

        layout.addWidget(QLabel("Command preview:"))
        self.preview = QTextEdit(readOnly=True)
        self.preview.setMaximumHeight(125)
        self.preview.setFont(QFont("Consolas", 9))
        layout.addWidget(self.preview)

        buttons = QHBoxLayout()
        restore = QPushButton("Restore settings")
        restore.clicked.connect(self.restore)
        buttons.addWidget(restore)
        save = QPushButton("Save settings")
        save.clicked.connect(self.save)
        buttons.addWidget(save)
        buttons.addStretch()
        run_local = QPushButton("Run Locally")
        run_local.setToolTip(
            "Run wrapper.py with the Python environment used to launch this window."
        )
        run_local.clicked.connect(self.run_local)
        buttons.addWidget(run_local)
        self.run_buttons.append(run_local)
        run_docker = QPushButton("Run Docker")
        run_docker.setToolTip("Run the configured container image with Docker.")
        run_docker.clicked.connect(self.run_docker)
        buttons.addWidget(run_docker)
        self.run_buttons.append(run_docker)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        self._connect_signals()
        self.refresh()

    @staticmethod
    def _parameter_grid(parent: QWidget, left_margin: int = 0) -> QGridLayout:
        grid = QGridLayout(parent)
        grid.setContentsMargins(left_margin, 0, 0, 0)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        return grid

    @staticmethod
    def _add_two_column_row(
        grid: QGridLayout, index: int, label: QLabel, widget: QWidget
    ) -> None:
        column = 0 if index % 2 == 0 else 2
        row = index // 2
        grid.addWidget(label, row, column)
        grid.addWidget(widget, row, column + 1)

    def _folder_row(self, edit: QLineEdit) -> QWidget:
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit)
        button = QPushButton("Browse...")
        button.clicked.connect(lambda: self._browse(edit))
        row.addWidget(button)
        return box

    def _browse(self, edit: QLineEdit) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select folder", edit.text())
        if selected:
            edit.setText(selected)

    def _widget(self, spec: dict) -> QWidget:
        if spec.get("multiselect"):
            return MultiSelectList(spec.get("options", []), spec.get("default", []))
        if spec.get("type") == "checkbox":
            widget = QCheckBox()
            widget.setChecked(bool(spec.get("default")))
            return widget
        if spec.get("options"):
            widget = NoWheelComboBox()
            for option in spec["options"]:
                widget.addItem(str(option["label"]), option["value"])
            widget.setCurrentIndex(max(widget.findData(spec.get("default")), 0))
            return widget
        if spec.get("type") == "integer":
            widget = NoWheelSpinBox()
            widget.setRange(
                int(spec.get("minimum", -999999)), int(spec.get("maximum", 999999))
            )
            widget.setValue(int(spec.get("default", 0)))
            return widget
        if spec.get("type") == "float":
            widget = NoWheelDoubleSpinBox()
            widget.setDecimals(6)
            widget.setRange(
                float(spec.get("minimum", -999999)),
                float(spec.get("maximum", 999999)),
            )
            widget.setValue(float(spec.get("default", 0)))
            return widget
        return QLineEdit(str(spec.get("default", "")))

    def _connect_signals(self) -> None:
        self.input_path.textChanged.connect(self.refresh)
        self.output_path.textChanged.connect(self.refresh)
        self.gpu.stateChanged.connect(self.refresh)
        for widget in self.widgets.values():
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.refresh)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(self.refresh)
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.connect(self.refresh)
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(self.refresh)
            elif isinstance(widget, MultiSelectList):
                widget.itemChanged.connect(self.refresh)

    def values(self) -> dict:
        values = {}
        for name, widget in self.widgets.items():
            if isinstance(widget, MultiSelectList):
                values[name] = widget.values()
            elif isinstance(widget, QComboBox):
                values[name] = widget.currentData()
            elif isinstance(widget, QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                values[name] = widget.value()
            else:
                values[name] = widget.text()
        return values

    def docker_command(self) -> list[str]:
        return build_docker_command(
            self.config,
            self.values(),
            self.input_path.text(),
            self.output_path.text(),
            self.gpu.isChecked(),
        )

    def local_command(self) -> list[str]:
        return build_local_command(
            self.config,
            self.values(),
            self.input_path.text(),
            self.output_path.text(),
        )

    def command(self) -> list[str]:
        """Return the Docker command for backward compatibility."""
        return self.docker_command()

    def refresh(self) -> None:
        self.preview.setPlainText(
            "Docker:\n"
            + subprocess.list2cmdline(self.docker_command())
            + "\n\nLocal Python:\n"
            + subprocess.list2cmdline(self.local_command())
        )

    def run_docker(self) -> None:
        self.save()
        subprocess.Popen(self.docker_command(), cwd=ROOT)

    def run_local(self) -> None:
        self.save()
        subprocess.Popen(self.local_command(), cwd=ROOT)

    def save(self) -> None:
        SETTINGS.write_text(
            json.dumps(
                {
                    "values": self.values(),
                    "input": self.input_path.text(),
                    "output": self.output_path.text(),
                    "gpu": self.gpu.isChecked(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def restore(self) -> None:
        if not SETTINGS.exists():
            return
        data = json.loads(SETTINGS.read_text(encoding="utf-8"))
        self.input_path.setText(data.get("input", ""))
        self.output_path.setText(data.get("output", ""))
        self.gpu.setChecked(data.get("gpu", True))
        restored_values = data.get("values", {})
        for name, value in restored_values.items():
            widget = self.widgets.get(name)
            if isinstance(widget, QComboBox):
                widget.setCurrentIndex(max(0, widget.findData(value)))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, MultiSelectList):
                selected = set(value)
                for index in range(widget.count()):
                    item = widget.item(index)
                    item.setCheckState(
                        Qt.CheckState.Checked
                        if item.data(Qt.ItemDataRole.UserRole) in selected
                        else Qt.CheckState.Unchecked
                    )
        self.refresh()


def main() -> int:
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(ICON)))
    window = Window()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
