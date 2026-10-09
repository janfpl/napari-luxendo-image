"""Export panel: write a Luxendo layer to .lux.h5 or BigTIFF.

Ported from shifter's export section. The panel exports image layers opened
by this plugin's reader (they carry their source file paths in
``layer.metadata``); a layer stacked from several files exports one file per
channel. The full volume or a Z/Y/X box (ROI) can be exported, and for
.lux.h5 the resolution pyramid and Imaris/BigDataViewer companion headers
are optional.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._companion_headers import find_companion_header_files
from ._export import (
    FORMAT_LUX_H5,
    FORMAT_TIFF,
    ExportPlan,
    Roi,
    plan_export,
    run_export,
)
from ._pyramids import detect_pyramid_levels
from ._reader import SOURCE_PATHS_KEY

if TYPE_CHECKING:
    import napari

logger = logging.getLogger(__name__)

ROI_LAYER_NAME = "Export ROI"

_HEADER_NOTES = {
    "copy": "",
    "single_resolution": " (rewritten to single resolution)",
    "roi": " (regenerated for the ROI)",
}


class ExportWorker(QThread):
    """Runs :func:`run_export` off the GUI thread."""

    # object (not int): byte counts for large exports can exceed 2**31.
    progress = Signal(object, object)
    export_done = Signal(object)  # summary Path, or None when cancelled
    failed = Signal(str)

    def __init__(self, plan: ExportPlan, ram_percent: int) -> None:
        super().__init__()
        self.plan = plan
        self.ram_percent = ram_percent
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            summary = run_export(
                self.plan,
                ram_percent=self.ram_percent,
                progress_callback=self.progress.emit,
                cancel_check=lambda: self._cancelled,
            )
            self.export_done.emit(summary)
        except MemoryError:
            self.failed.emit(
                "Ran out of memory while exporting.\n\n"
                "Lower the RAM allocation slider, close other programs, or "
                "export a smaller ROI."
            )
        except Exception:  # noqa: BLE001 - reported to the user
            self.failed.emit(traceback.format_exc())


def _layer_shape_zyx(layer: Any) -> tuple[int, int, int]:
    """Full-resolution (Z, Y, X) shape of an image layer."""
    data = layer.data[0] if layer.multiscale else layer.data
    return tuple(int(d) for d in data.shape[-3:])


def _is_luxendo_layer(layer: Any) -> bool:
    return (
        layer.__class__.__name__ == "Image"
        and bool(layer.metadata.get(SOURCE_PATHS_KEY))
        and layer.ndim >= 3
    )


class LuxendoExportWidget(QWidget):
    """Dock widget exporting Luxendo layers to .lux.h5 or BigTIFF."""

    def __init__(self, napari_viewer: napari.viewer.Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self._roi_layer = None
        self._worker: ExportWorker | None = None

        lay = QVBoxLayout()
        lay.addWidget(self._build_source_section())
        lay.addWidget(self._build_region_section())
        lay.addWidget(self._build_output_section())
        lay.addStretch()
        self.setLayout(lay)

        # Keep the layer list in sync with the viewer. Disconnect when the
        # widget is destroyed so the viewer doesn't call into a dead widget.
        events = self.viewer.layers.events
        callbacks = [
            (events.inserted, self._refresh_layers),
            (events.removed, self._refresh_layers),
        ]
        for event, cb in callbacks:
            event.connect(cb)

        def _disconnect(*_args: Any) -> None:
            for event, cb in callbacks:
                event.disconnect(cb)

        self.destroyed.connect(_disconnect)

        self._refresh_layers()
        self._on_layer_changed()
        self._on_format_changed()
        self._on_region_changed()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_source_section(self) -> QGroupBox:
        grp = QGroupBox("Source")
        lay = QFormLayout()

        self.combo_layer = QComboBox()
        self.combo_layer.setToolTip(
            "Image layers opened from .lux.h5 files with this plugin."
        )
        self.combo_layer.currentIndexChanged.connect(self._on_layer_changed)
        lay.addRow("Layer:", self.combo_layer)

        self.lbl_source_info = QLabel("")
        self.lbl_source_info.setWordWrap(True)
        lay.addRow(self.lbl_source_info)

        self.combo_format = QComboBox()
        self.combo_format.addItem("Luxendo H5 (.lux.h5)", FORMAT_LUX_H5)
        self.combo_format.addItem("BigTIFF (.tif)", FORMAT_TIFF)
        self.combo_format.currentIndexChanged.connect(self._on_format_changed)
        lay.addRow("Output format:", self.combo_format)

        grp.setLayout(lay)
        return grp

    def _build_region_section(self) -> QGroupBox:
        grp = QGroupBox("Export region")
        lay = QVBoxLayout()

        row = QHBoxLayout()
        self.radio_full = QRadioButton("Full volume")
        self.radio_roi = QRadioButton("ROI only")
        self.radio_full.setChecked(True)
        self._region_group = QButtonGroup(self)
        self._region_group.addButton(self.radio_full, 0)
        self._region_group.addButton(self.radio_roi, 1)
        self.radio_full.toggled.connect(self._on_region_changed)
        row.addWidget(self.radio_full)
        row.addWidget(self.radio_roi)
        row.addStretch()
        lay.addLayout(row)

        self.roi_box = QWidget()
        roi_lay = QVBoxLayout()
        roi_lay.setContentsMargins(0, 0, 0, 0)

        row_btns = QHBoxLayout()
        self.btn_draw_roi = QPushButton("Draw ROI rectangle")
        self.btn_draw_roi.setToolTip(
            "Draw a rectangle in the viewer to set the Y/X bounds. "
            "The last rectangle drawn is used."
        )
        self.btn_draw_roi.clicked.connect(self._on_draw_roi)
        self.btn_reset_roi = QPushButton("Reset to full volume")
        self.btn_reset_roi.clicked.connect(self._reset_roi_spins)
        row_btns.addWidget(self.btn_draw_roi)
        row_btns.addWidget(self.btn_reset_roi)
        roi_lay.addLayout(row_btns)

        grid = QGridLayout()
        grid.addWidget(QLabel("start"), 0, 1)
        grid.addWidget(QLabel("stop (exclusive)"), 0, 2)
        self.roi_spins: dict[str, tuple[QSpinBox, QSpinBox]] = {}
        for r, axis in enumerate("ZYX", start=1):
            start, stop = QSpinBox(), QSpinBox()
            start.setRange(0, 0)
            stop.setRange(0, 0)
            grid.addWidget(QLabel(f"{axis}:"), r, 0)
            grid.addWidget(start, r, 1)
            grid.addWidget(stop, r, 2)
            self.roi_spins[axis] = (start, stop)
        roi_lay.addLayout(grid)

        row_z = QHBoxLayout()
        self.btn_z_start_here = QPushButton("Z start = current slice")
        self.btn_z_start_here.clicked.connect(
            lambda: self._set_z_from_slice(stop=False)
        )
        self.btn_z_stop_here = QPushButton("Z stop = current slice")
        self.btn_z_stop_here.clicked.connect(
            lambda: self._set_z_from_slice(stop=True)
        )
        row_z.addWidget(self.btn_z_start_here)
        row_z.addWidget(self.btn_z_stop_here)
        roi_lay.addLayout(row_z)

        self.roi_box.setLayout(roi_lay)
        lay.addWidget(self.roi_box)

        grp.setLayout(lay)
        return grp

    def _build_output_section(self) -> QGroupBox:
        grp = QGroupBox("Export")
        lay = QVBoxLayout()

        row_outdir = QHBoxLayout()
        self.btn_select_outdir = QPushButton("Output directory...")
        self.btn_select_outdir.clicked.connect(self._on_select_output_dir)
        self.edit_outdir = QLineEdit()
        self.edit_outdir.setPlaceholderText("No output directory selected")
        row_outdir.addWidget(self.btn_select_outdir)
        row_outdir.addWidget(self.edit_outdir)
        lay.addLayout(row_outdir)

        row_ram = QHBoxLayout()
        row_ram.addWidget(QLabel("RAM allocation:"))
        self.slider_ram = QSlider(Qt.Horizontal)
        self.slider_ram.setRange(50, 95)
        self.slider_ram.setValue(90)
        self.slider_ram.setTickPosition(QSlider.TicksBelow)
        self.slider_ram.setTickInterval(5)
        self.slider_ram.setToolTip(
            "Share of currently available RAM a Z-slab may use."
        )
        self.lbl_ram = QLabel("90%")
        self.slider_ram.valueChanged.connect(
            lambda v: self.lbl_ram.setText(f"{v}%")
        )
        row_ram.addWidget(self.slider_ram)
        row_ram.addWidget(self.lbl_ram)
        lay.addLayout(row_ram)

        # Pyramids are built from each slab in memory (no reread), so they
        # are cheap enough to write by default. Untick for the fastest export
        # when only full resolution is needed.
        self.chk_write_pyramids = QCheckBox(
            "Write low-resolution pyramid layers (H5 only)"
        )
        self.chk_write_pyramids.setChecked(True)
        self.chk_write_pyramids.setToolTip(
            "Regenerate the Data_W_H_D levels found in the source file."
        )
        lay.addWidget(self.chk_write_pyramids)

        self.chk_write_headers = QCheckBox(
            "Write Imaris/BigDataViewer companion headers (H5 only)"
        )
        self.chk_write_headers.setChecked(True)
        self.chk_write_headers.setToolTip(
            "Copy or rebuild the .ims / *_bdv.h5 / *_bdv.xml headers found "
            "next to the source files so they match the exported data.\n"
            "They reference every channel in the acquisition, so untick when "
            "exporting only some channels."
        )
        lay.addWidget(self.chk_write_headers)

        row_btns = QHBoxLayout()
        self.btn_export = QPushButton("Export")
        self.btn_export.clicked.connect(self._on_export)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_cancel.setVisible(False)
        row_btns.addWidget(self.btn_export)
        row_btns.addWidget(self.btn_cancel)
        lay.addLayout(row_btns)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        lay.addWidget(self.progress_bar)

        self.lbl_progress = QLabel("")
        self.lbl_progress.setVisible(False)
        lay.addWidget(self.lbl_progress)

        grp.setLayout(lay)
        return grp

    # ------------------------------------------------------------------ #
    # Layer selection
    # ------------------------------------------------------------------ #

    def _current_layer(self) -> Any | None:
        layer = self.combo_layer.currentData()
        if layer is None or layer not in self.viewer.layers:
            return None
        return layer

    def _refresh_layers(self, *_args: Any) -> None:
        previous = self._current_layer()
        self.combo_layer.blockSignals(True)
        self.combo_layer.clear()
        for layer in self.viewer.layers:
            if _is_luxendo_layer(layer):
                self.combo_layer.addItem(layer.name, layer)
        idx = self.combo_layer.findData(previous) if previous else -1
        if idx < 0 and self.combo_layer.count():
            # Default to the most recently added Luxendo layer.
            idx = self.combo_layer.count() - 1
        self.combo_layer.setCurrentIndex(idx)
        self.combo_layer.blockSignals(False)
        if self._current_layer() is not previous:
            self._on_layer_changed()
        else:
            self._update_export_enabled()

    def _source_paths(self) -> list[Path]:
        layer = self._current_layer()
        if layer is None:
            return []
        return [Path(p) for p in layer.metadata.get(SOURCE_PATHS_KEY, [])]

    def _on_layer_changed(self, *_args: Any) -> None:
        layer = self._current_layer()
        if layer is None:
            self.lbl_source_info.setText(
                "Open a .lux.h5 file with this plugin to export it."
            )
            for start, stop in self.roi_spins.values():
                start.setRange(0, 0)
                stop.setRange(0, 0)
        else:
            nz, ny, nx = _layer_shape_zyx(layer)
            paths = self._source_paths()
            info = [
                f"{len(paths)} channel file(s), {nx} x {ny} x {nz} (X x Y x Z)"
            ]
            try:
                with h5py.File(paths[0], "r") as f:
                    levels = detect_pyramid_levels(f)
                info.append(
                    "Pyramid levels: "
                    + (", ".join(lvl[0] for lvl in levels) or "none")
                )
            except OSError as exc:
                info.append(f"Cannot read source file: {exc}")
            headers = find_companion_header_files(paths[0].parent)
            if headers:
                info.append(
                    "Companion headers: " + ", ".join(f.name for f in headers)
                )
            self.lbl_source_info.setText("\n".join(info))
            for axis, size in zip("ZYX", (nz, ny, nx), strict=True):
                start, stop = self.roi_spins[axis]
                start.setRange(0, max(0, size - 1))
                stop.setRange(1, size)
            self._reset_roi_spins()
        self._update_export_enabled()

    # ------------------------------------------------------------------ #
    # Format / region
    # ------------------------------------------------------------------ #

    def _format(self) -> str:
        return self.combo_format.currentData()

    def _on_format_changed(self, *_args: Any) -> None:
        is_h5 = self._format() == FORMAT_LUX_H5
        self.chk_write_pyramids.setEnabled(is_h5)
        self.chk_write_headers.setEnabled(is_h5)

    def _on_region_changed(self, *_args: Any) -> None:
        self.roi_box.setEnabled(self.radio_roi.isChecked())

    def _update_export_enabled(self) -> None:
        idle = self._worker is None
        self.btn_export.setEnabled(idle and self._current_layer() is not None)

    # ------------------------------------------------------------------ #
    # ROI
    # ------------------------------------------------------------------ #

    def _reset_roi_spins(self) -> None:
        for start, stop in self.roi_spins.values():
            start.setValue(0)
            stop.setValue(stop.maximum())

    def _roi_from_spins(self) -> Roi:
        values: list[int] = []
        for axis in "ZYX":
            start, stop = self.roi_spins[axis]
            values += [start.value(), stop.value()]
        return tuple(values)

    def _on_draw_roi(self) -> None:
        layer = self._current_layer()
        if layer is None:
            return
        if (
            self._roi_layer is None
            or self._roi_layer not in self.viewer.layers
        ):
            # 2-D so the rectangle applies to every Z-slice, and scaled like
            # the image so its vertices are in the image's voxel units.
            self._roi_layer = self.viewer.add_shapes(
                ndim=2,
                name=ROI_LAYER_NAME,
                scale=layer.scale[-2:],
                translate=layer.translate[-2:],
                edge_color="yellow",
                face_color="transparent",
                edge_width=2,
            )
            self._roi_layer.events.data.connect(self._on_roi_shape_changed)
        self.viewer.layers.selection.active = self._roi_layer
        self._roi_layer.mode = "add_rectangle"
        self.radio_roi.setChecked(True)

    def _on_roi_shape_changed(self, *_args: Any) -> None:
        shapes = self._roi_layer
        layer = self._current_layer()
        if shapes is None or layer is None or len(shapes.data) == 0:
            return
        # Shape vertices -> world -> image voxel coordinates (Y, X).
        verts = np.asarray(shapes.data[-1])[:, -2:]
        world = verts * np.asarray(shapes.scale[-2:]) + np.asarray(
            shapes.translate[-2:]
        )
        vox = (world - np.asarray(layer.translate[-2:])) / np.asarray(
            layer.scale[-2:]
        )
        _, ny, nx = _layer_shape_zyx(layer)
        for axis, col, size in (("Y", 0, ny), ("X", 1, nx)):
            lo = int(np.clip(math.floor(vox[:, col].min()), 0, size - 1))
            hi = int(np.clip(math.ceil(vox[:, col].max()), lo + 1, size))
            start, stop = self.roi_spins[axis]
            start.setValue(lo)
            stop.setValue(hi)

    def _set_z_from_slice(self, stop: bool) -> None:
        layer = self._current_layer()
        if layer is None:
            return
        nz = _layer_shape_zyx(layer)[0]
        z_world = self.viewer.dims.point[-3]
        z = round((z_world - layer.translate[-3]) / layer.scale[-3])
        z = int(np.clip(z, 0, nz - 1))
        start_spin, stop_spin = self.roi_spins["Z"]
        if stop:
            stop_spin.setValue(z + 1)  # include the current slice
        else:
            start_spin.setValue(z)

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def _on_select_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.edit_outdir.text()
        )
        if path:
            self.edit_outdir.setText(path)

    def _confirm_text(self, plan: ExportPlan) -> str:
        fmt_name = self.combo_format.currentText()
        lines = [f"Format: {fmt_name}", f"Output directory: {plan.output_dir}"]
        if plan.roi is not None:
            z0, z1, y0, y1, x0, x1 = plan.roi
            lines.append(
                f"Region: ROI  X=[{x0},{x1})  Y=[{y0},{y1})  Z=[{z0},{z1})"
            )
        else:
            lines.append("Region: full volume")
        nz, ny, nx = plan.channels[0].output_shape
        lines.append(f"Output dimensions: {nx} x {ny} x {nz} (X x Y x Z)")
        lines.append("\nFiles:")
        lines += [f"  {ch.output.name}" for ch in plan.channels]
        if plan.fmt == FORMAT_LUX_H5:
            levels = {
                lvl[0] for ch in plan.channels for lvl in ch.pyramid_levels
            }
            if plan.write_pyramids:
                lines.append(
                    "Pyramid levels: " + (", ".join(sorted(levels)) or "none")
                )
            else:
                lines.append("Pyramid levels: disabled")
        if plan.header_files:
            lines.append(
                f"Companion headers{_HEADER_NOTES[plan.header_action]}: "
                + ", ".join(f.name for f in plan.header_files)
            )
        lines.append(
            f"\nEstimated output size: {plan.total_bytes / 1024**3:.2f} GB"
        )
        lines.append(f"RAM allocation: {self.slider_ram.value()}%")
        return "\n".join(lines)

    def _on_export(self) -> None:
        paths = self._source_paths()
        if not paths:
            QMessageBox.warning(self, "No layer", "Select a Luxendo layer.")
            return
        outdir = self.edit_outdir.text().strip()
        if not outdir:
            QMessageBox.warning(
                self, "No output directory", "Select an output directory."
            )
            return

        roi = self._roi_from_spins() if self.radio_roi.isChecked() else None
        try:
            plan = plan_export(
                paths,
                Path(outdir),
                fmt=self._format(),
                roi=roi,
                write_pyramids=self.chk_write_pyramids.isChecked(),
                write_headers=self.chk_write_headers.isChecked(),
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot export", str(exc))
            return

        existing = [p.name for p in plan.output_paths if p.exists()]
        if existing:
            ans = QMessageBox.question(
                self,
                "Overwrite?",
                "The following files already exist and will be overwritten:\n"
                + "\n".join(f"  {f}" for f in existing)
                + "\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return

        ans = QMessageBox.question(
            self,
            "Confirm Export",
            self._confirm_text(plan),
            QMessageBox.Ok | QMessageBox.Cancel,
        )
        if ans != QMessageBox.Ok:
            return

        self._start_worker(plan)

    def _start_worker(self, plan: ExportPlan) -> None:
        self._set_running(True)
        self.lbl_progress.setText("Starting export...")
        self._worker = ExportWorker(plan, self.slider_ram.value())
        self._worker.progress.connect(self._on_progress)
        self._worker.export_done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.btn_cancel.setEnabled(False)
            self.lbl_progress.setText("Cancelling...")

    def _set_running(self, running: bool) -> None:
        for w in (
            self.combo_layer,
            self.combo_format,
            self.radio_full,
            self.radio_roi,
            self.roi_box,
            self.btn_select_outdir,
            self.edit_outdir,
            self.slider_ram,
        ):
            w.setEnabled(not running)
        if not running:
            self._on_region_changed()
            self._on_format_changed()
        else:
            self.chk_write_pyramids.setEnabled(False)
            self.chk_write_headers.setEnabled(False)
        self.btn_cancel.setVisible(running)
        self.btn_cancel.setEnabled(running)
        self.progress_bar.setVisible(running)
        self.progress_bar.setValue(0)
        self.lbl_progress.setVisible(running)
        self.btn_export.setEnabled(not running)

    def _finish_worker(self) -> None:
        if self._worker is not None:
            self._worker.wait()
            self._worker = None
        self._set_running(False)
        self._update_export_enabled()

    def _on_progress(self, done: int, total: int) -> None:
        if total > 0:
            pct = int(100 * done / total)
            self.progress_bar.setValue(pct)
            self.lbl_progress.setText(
                f"{done / 1024**3:.2f} / {total / 1024**3:.2f} GB written "
                f"({pct}%)"
            )

    def _on_done(self, summary: Path | None) -> None:
        self._finish_worker()
        if summary is None:
            QMessageBox.information(
                self,
                "Cancelled",
                "Export was cancelled. The partially written file was "
                "removed.",
            )
            return
        out_dir = Path(summary).parent
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Export Complete")
        box.setText(f"Export finished.\nOutput folder: {out_dir}")
        open_btn = box.addButton("Open folder", QMessageBox.ActionRole)
        close_btn = box.addButton("Close", QMessageBox.RejectRole)
        box.setDefaultButton(close_btn)
        box.exec_() if hasattr(box, "exec_") else box.exec()
        if box.clickedButton() is open_btn:
            _open_in_file_browser(out_dir)

    def _on_failed(self, message: str) -> None:
        self._finish_worker()
        QMessageBox.critical(self, "Export Error", message)


def _open_in_file_browser(path: Path) -> None:
    """Reveal *path* in the OS file browser."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError as exc:
        logger.warning("Could not open %s in file browser: %s", path, exc)
