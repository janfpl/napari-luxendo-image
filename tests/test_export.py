import json

import h5py
import numpy as np
import pytest
import tifffile

from napari_luxendo_image import _export
from napari_luxendo_image._export import (
    FORMAT_LUX_H5,
    FORMAT_TIFF,
    SUMMARY_FILENAME,
    plan_export,
    run_export,
)

from .conftest import block_mean

ROI = (3, 18, 5, 27, 2, 31)  # z0, z1, y0, y1, x0, x1


@pytest.fixture
def small_slabs(monkeypatch):
    """Force many 4-plane slabs so pyramid groups straddle slab boundaries."""
    monkeypatch.setattr(_export, "compute_chunk_size", lambda *a, **k: 4)


def _crop(data, roi):
    z0, z1, y0, y1, x0, x1 = roi
    return data[z0:z1, y0:y1, x0:x1]


def _ims_text(value):
    return np.asarray(value).tobytes().decode()


@pytest.mark.parametrize("roi", [None, ROI])
def test_lux_h5_export_with_pyramids(lux_dataset, tmp_path, small_slabs, roi):
    out = tmp_path / "out"
    plan = plan_export(lux_dataset["paths"], out, FORMAT_LUX_H5, roi=roi)
    progress = []
    summary = run_export(
        plan, progress_callback=lambda d, t: progress.append((d, t))
    )

    suffix = "_roi" if roi else ""
    for src, data in zip(
        lux_dataset["paths"], lux_dataset["arrays"], strict=True
    ):
        expected = _crop(data, roi) if roi else data
        out_file = out / src.name.replace(".lux.h5", f"{suffix}.lux.h5")
        with h5py.File(out_file, "r") as f, h5py.File(src, "r") as s:
            np.testing.assert_array_equal(f["Data"][()], expected)
            for name, fw, fh, fd in (
                ("Data_2_2_2", 2, 2, 2),
                ("Data_4_4_4", 4, 4, 4),
            ):
                np.testing.assert_array_equal(
                    f[name][()], block_mean(expected, fw, fh, fd)
                )
            assert f["metadata"][()] == s["metadata"][()]

    assert progress[-1][0] == progress[-1][1] == plan.total_bytes
    meta = json.loads(summary.read_text())
    assert meta["export_region"] == ("roi" if roi else "full_volume")
    assert meta["pyramids_written"] is True


def test_full_export_copies_headers_verbatim(lux_dataset, tmp_path):
    out = tmp_path / "out"
    plan = plan_export(lux_dataset["paths"], out)
    assert plan.header_action == "copy"
    run_export(plan)
    for name in ("stack.ims", "stack_bdv.h5", "stack_bdv.xml"):
        assert (out / name).read_bytes() == (
            lux_dataset["dir"] / name
        ).read_bytes()


def test_no_pyramids_writes_single_resolution_headers(lux_dataset, tmp_path):
    out = tmp_path / "out"
    plan = plan_export(lux_dataset["paths"], out, write_pyramids=False)
    assert plan.header_action == "single_resolution"
    run_export(plan)

    with h5py.File(out / "stack_ch0.lux.h5", "r") as f:
        assert sorted(f.keys()) == ["Data", "metadata"]
    with h5py.File(out / "stack.ims", "r") as f:
        assert list(f["DataSet"].keys()) == ["ResolutionLevel 0"]
        assert "Imaris" not in f["DataSetInfo"]
    with h5py.File(out / "stack_bdv.h5", "r") as f:
        assert list(f["t00000/s00"].keys()) == ["0"]
        assert f["s00/resolutions"].shape == (1, 3)


def test_roi_headers_point_at_roi_files(lux_dataset, tmp_path):
    out = tmp_path / "out"
    plan = plan_export(lux_dataset["paths"], out, roi=ROI)
    assert plan.header_action == "roi"
    run_export(plan)

    z0, z1, y0, y1, x0, x1 = ROI
    with h5py.File(out / "stack.ims", "r") as f:
        assert len(f["DataSet"]) == 3
        link = f["DataSet/ResolutionLevel 1/TimePoint 0/Channel 1"].get(
            "Data", getlink=True
        )
        assert (link.filename, link.path) == (
            "stack_ch1_roi.lux.h5",
            "Data_2_2_2",
        )
        img = f["DataSetInfo/Image"].attrs
        assert _ims_text(img["X"]) == str(x1 - x0)
        assert _ims_text(img["Z"]) == str(z1 - z0)
        assert float(_ims_text(img["ExtMin0"])) == pytest.approx(x0 * 0.5)
        # The links resolve against the exported data.
        np.testing.assert_array_equal(
            f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"][()],
            _crop(lux_dataset["arrays"][0], ROI),
        )
    # The BDV XML can't be regenerated for an ROI and is skipped.
    assert not (out / "stack_bdv.xml").exists()


@pytest.mark.parametrize("roi", [None, ROI])
def test_bigtiff_export(lux_dataset, tmp_path, small_slabs, roi):
    out = tmp_path / "out"
    plan = plan_export(lux_dataset["paths"], out, FORMAT_TIFF, roi=roi)
    assert plan.header_files == []
    assert not plan.write_pyramids
    run_export(plan)

    suffix = "_roi" if roi else ""
    for src, data in zip(
        lux_dataset["paths"], lux_dataset["arrays"], strict=True
    ):
        tif = out / src.name.replace(".lux.h5", f"{suffix}.tif")
        with tifffile.TiffFile(tif) as tf:
            assert tf.is_bigtiff
            got = tf.asarray()
        np.testing.assert_array_equal(got, _crop(data, roi) if roi else data)


def test_headers_can_be_disabled(lux_dataset, tmp_path):
    plan = plan_export(
        lux_dataset["paths"], tmp_path / "out", write_headers=False
    )
    assert plan.header_files == []


def test_refuses_to_write_into_source_dir(lux_dataset):
    with pytest.raises(ValueError, match="different from the directory"):
        plan_export(lux_dataset["paths"], lux_dataset["dir"])


@pytest.mark.parametrize(
    "roi",
    [(0, 22, 0, 30, 0, 34), (5, 5, 0, 30, 0, 34), (0, 21, -1, 30, 0, 34)],
)
def test_invalid_roi(lux_dataset, tmp_path, roi):
    with pytest.raises(ValueError, match="ROI"):
        plan_export(lux_dataset["paths"], tmp_path / "out", roi=roi)


def test_cancel_removes_partial_file(lux_dataset, tmp_path, small_slabs):
    out = tmp_path / "out"
    plan = plan_export(lux_dataset["paths"], out)
    calls = iter([False, False, True])
    assert run_export(plan, cancel_check=lambda: next(calls, True)) is None
    assert not (out / "stack_ch0.lux.h5").exists()
    assert not (out / SUMMARY_FILENAME).exists()
