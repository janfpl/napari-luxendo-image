import h5py
import numpy as np
import pytest

pytest.importorskip("napari")

from napari_luxendo_image._export import (
    FORMAT_LUX_H5,
    plan_export,
)  # noqa: E402
from napari_luxendo_image._widget import LuxendoExportWidget  # noqa: E402


@pytest.fixture
def viewer_with_layer(make_napari_viewer, lux_dataset):
    viewer = make_napari_viewer()
    viewer.open(str(lux_dataset["paths"][0]), plugin="napari-luxendo-image")
    return viewer


def test_widget_lists_luxendo_layers(make_napari_viewer, lux_dataset):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 4, 4)), name="not luxendo")
    widget = LuxendoExportWidget(viewer)
    assert widget.combo_layer.count() == 0
    assert not widget.btn_export.isEnabled()

    viewer.open(str(lux_dataset["paths"][0]), plugin="napari-luxendo-image")
    assert widget.combo_layer.count() == 1
    assert widget.btn_export.isEnabled()
    assert "Data_2_2_2" in widget.lbl_source_info.text()
    nz, ny, nx = lux_dataset["shape"]
    assert widget.roi_spins["Z"][1].value() == nz
    assert widget.roi_spins["X"][1].value() == nx


def test_drawn_rectangle_sets_yx_bounds(viewer_with_layer):
    widget = LuxendoExportWidget(viewer_with_layer)
    widget._on_draw_roi()
    assert widget.radio_roi.isChecked()

    # The ROI layer shares the image's scale, so vertices are voxel indices.
    widget._roi_layer.add(
        np.array([[5.0, 2.0], [5.0, 31.0], [27.0, 31.0], [27.0, 2.0]]),
        shape_type="rectangle",
    )
    assert widget.roi_spins["Y"][0].value() == 5
    assert widget.roi_spins["Y"][1].value() == 27
    assert widget.roi_spins["X"][0].value() == 2
    assert widget.roi_spins["X"][1].value() == 31

    viewer_with_layer.dims.set_point(0, 8 * 2.0)  # Z voxel 8 (scale 2 um)
    widget._set_z_from_slice(stop=False)
    viewer_with_layer.dims.set_point(0, 15 * 2.0)
    widget._set_z_from_slice(stop=True)
    assert widget._roi_from_spins() == (8, 16, 5, 27, 2, 31)


def test_worker_exports_roi(
    viewer_with_layer, lux_dataset, tmp_path, qtbot, monkeypatch
):
    from qtpy.QtWidgets import QMessageBox

    # The completion dialog is modal; record it instead of blocking.
    shown = []
    for name in ("exec", "exec_"):
        monkeypatch.setattr(
            QMessageBox,
            name,
            lambda box: shown.append(box.text()),
            raising=False,
        )

    widget = LuxendoExportWidget(viewer_with_layer)
    roi = (2, 10, 3, 20, 4, 30)
    plan = plan_export(
        widget._source_paths(), tmp_path / "out", FORMAT_LUX_H5, roi=roi
    )
    widget._start_worker(plan)
    assert not widget.btn_export.isEnabled()
    assert widget.btn_cancel.isVisible() or not widget.isVisible()

    qtbot.waitUntil(lambda: widget._worker is None, timeout=30000)
    assert widget.btn_export.isEnabled()
    assert shown and "Export finished" in shown[0]

    z0, z1, y0, y1, x0, x1 = roi
    with h5py.File(tmp_path / "out" / "stack_ch0_roi.lux.h5", "r") as f:
        np.testing.assert_array_equal(
            f["Data"][()], lux_dataset["arrays"][0][z0:z1, y0:y1, x0:x1]
        )
