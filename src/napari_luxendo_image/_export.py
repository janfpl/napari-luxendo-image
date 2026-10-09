"""Chunked export of Luxendo .lux.h5 channels to .lux.h5 or BigTIFF.

Ported from shifter (``shifter/export_engine.py``) without the shift
correction: each channel's full-resolution ``Data`` is streamed in Z-slabs,
optionally cropped to an ROI, and written as either

* a Luxendo ``.lux.h5`` file (``Data`` + the original ``metadata`` dataset,
  plus regenerated ``Data_W_H_D`` pyramid levels when requested), with any
  Imaris/BigDataViewer companion headers reconciled to the output, or
* a BigTIFF, one page per Z-plane.

This module has no Qt/napari dependency so it can be used headless.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import psutil
import tifffile

from ._companion_headers import (
    copy_companion_header_files,
    find_companion_header_files,
    lux_output_filename,
    write_roi_headers,
    write_single_resolution_headers,
)
from ._pyramids import (
    StreamingPyramidWriter,
    compute_pyramid_level_shape,
    detect_pyramid_levels,
    pyramid_backend_status,
)

logger = logging.getLogger(__name__)

FORMAT_LUX_H5 = "lux.h5"
FORMAT_TIFF = "tiff"

ROI_SUFFIX = "_roi"
SUMMARY_FILENAME = "luxendo_export.json"

# (z_start, z_end, y_start, y_end, x_start, x_end), ends exclusive.
Roi = tuple[int, int, int, int, int, int]
ProgressCallback = Callable[[int, int], None]
CancelCheck = Callable[[], bool]


# --- Chunk-size policy ----------------------------------------------------- #
# Reading a slab and writing it (plus the pyramid working buffers) transiently
# holds a few full-size copies of it in RAM; budget for that many copies.
_SLAB_PEAK_COPIES = 3

# Hard cap on a single slab's bytes, independent of installed RAM. Export is
# disk-bound, so larger slabs are not faster -- only a bigger memory footprint.
_MAX_SLAB_BYTES = 4 * 1024**3


def compute_chunk_size(
    xy_shape: tuple[int, int],
    ram_percent: int = 90,
    bytes_per_voxel: int = 2,
) -> int:
    """Number of Z-planes to process per slab.

    Bounded by *ram_percent* of the currently *available* system memory and
    by an absolute per-slab cap (:data:`_MAX_SLAB_BYTES`).
    """
    available = psutil.virtual_memory().available
    budget = int(available * ram_percent / 100)

    ny, nx = xy_shape
    plane_bytes = max(1, ny * nx * bytes_per_voxel)

    chunk_z_by_ram = budget // (plane_bytes * _SLAB_PEAK_COPIES)
    chunk_z_by_cap = _MAX_SLAB_BYTES // plane_bytes
    return max(1, int(min(chunk_z_by_ram, chunk_z_by_cap)))


# --- Planning ------------------------------------------------------------- #


@dataclass
class ChannelPlan:
    """What will be written for one source channel."""

    source: Path
    output: Path
    source_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    dtype: np.dtype
    pyramid_levels: list[tuple[str, int, int, int]]
    output_bytes: int


@dataclass
class ExportPlan:
    """Validated description of an export, computed before any writing.

    The widget uses it to warn about overwrites and to show the estimated
    size; :func:`run_export` executes it.
    """

    fmt: str
    output_dir: Path
    roi: Roi | None
    write_pyramids: bool
    channels: list[ChannelPlan]
    header_files: list[Path] = field(default_factory=list)
    header_action: str = "none"  # none | copy | single_resolution | roi

    @property
    def total_bytes(self) -> int:
        return sum(ch.output_bytes for ch in self.channels)

    @property
    def output_paths(self) -> list[Path]:
        paths = [ch.output for ch in self.channels]
        paths += [self.output_dir / f.name for f in self.header_files]
        paths.append(self.output_dir / SUMMARY_FILENAME)
        return paths


def _stem(filename: str) -> str:
    lower = filename.lower()
    for ext in (".lux.h5", ".h5"):
        if lower.endswith(ext):
            return filename[: -len(ext)]
    return filename


def output_filename(source: Path, fmt: str, roi: Roi | None) -> str:
    """Output filename for *source*.

    A full-volume .lux.h5 export keeps the original filename, so copied
    Imaris/BigDataViewer headers continue to resolve their links; an ROI
    export appends ``_roi``.
    """
    suffix = ROI_SUFFIX if roi is not None else ""
    if fmt == FORMAT_LUX_H5:
        return lux_output_filename(source.name, suffix)
    return f"{_stem(source.name)}{suffix}.tif"


def validate_roi(roi: Roi, shape: tuple[int, int, int]) -> None:
    """Raise ValueError unless *roi* is a non-empty box inside *shape*."""
    for axis, (start, end, size) in zip(
        "ZYX",
        (
            (roi[0], roi[1], shape[0]),
            (roi[2], roi[3], shape[1]),
            (roi[4], roi[5], shape[2]),
        ),
        strict=True,
    ):
        if not 0 <= start < end <= size:
            raise ValueError(
                f"ROI {axis} range [{start}, {end}) is empty or outside the "
                f"volume (size {size})."
            )


def _output_bytes(
    out_shape: tuple[int, int, int],
    itemsize: int,
    pyramid_levels: list[tuple[str, int, int, int]],
) -> int:
    total = int(np.prod(out_shape)) * itemsize
    for _name, fw, fh, fd in pyramid_levels:
        pshape = compute_pyramid_level_shape(out_shape, fw, fh, fd)
        if all(d > 0 for d in pshape):
            total += int(np.prod(pshape)) * 2
    return total


def plan_export(
    sources: list[Path | str],
    output_dir: Path | str,
    fmt: str = FORMAT_LUX_H5,
    roi: Roi | None = None,
    write_pyramids: bool = True,
    write_headers: bool = True,
) -> ExportPlan:
    """Validate an export request and describe what it will write.

    Raises
    ------
    ValueError
        For an unknown format, missing/invalid sources, an ROI outside a
        source volume, or an output directory that contains a source file.
    """
    if fmt not in (FORMAT_LUX_H5, FORMAT_TIFF):
        raise ValueError(f"Unknown export format: {fmt!r}")
    if not sources:
        raise ValueError("No source files to export.")

    output_dir = Path(output_dir)
    source_paths = [Path(s) for s in sources]
    source_dirs = {p.resolve().parent for p in source_paths}
    if output_dir.exists() and output_dir.resolve() in source_dirs:
        # Full-volume H5 export keeps the original filenames and rewritten
        # companion headers replace same-named originals: writing into the
        # input directory would destroy the data being read.
        raise ValueError(
            "Choose an output directory different from the directory "
            "containing the source files."
        )

    pyramids = write_pyramids and fmt == FORMAT_LUX_H5
    channels: list[ChannelPlan] = []
    for src in source_paths:
        if not src.is_file():
            raise ValueError(f"Source file not found: {src}")
        with h5py.File(src, "r") as f:
            if "Data" not in f:
                raise ValueError(f"No 'Data' dataset in {src.name}.")
            data = f["Data"]
            if data.ndim != 3:
                raise ValueError(
                    f"'Data' in {src.name} is {data.ndim}-D; expected 3-D "
                    "(Z, Y, X)."
                )
            shape = tuple(int(d) for d in data.shape)
            dtype = data.dtype
            levels = detect_pyramid_levels(f) if pyramids else []
        if levels and dtype != np.uint16:
            raise ValueError(
                f"Pyramid regeneration supports uint16 data only; "
                f"{src.name} is {dtype}. Disable pyramid layers to export it."
            )
        if roi is not None:
            validate_roi(roi, shape)
            out_shape = (roi[1] - roi[0], roi[3] - roi[2], roi[5] - roi[4])
        else:
            out_shape = shape
        channels.append(
            ChannelPlan(
                source=src,
                output=output_dir / output_filename(src, fmt, roi),
                source_shape=shape,
                output_shape=out_shape,
                dtype=dtype,
                pyramid_levels=levels,
                output_bytes=_output_bytes(out_shape, dtype.itemsize, levels),
            )
        )

    plan = ExportPlan(
        fmt=fmt,
        output_dir=output_dir,
        roi=roi,
        write_pyramids=pyramids,
        channels=channels,
    )

    if fmt == FORMAT_LUX_H5 and write_headers:
        plan.header_files = find_companion_header_files(source_paths[0].parent)
        if plan.header_files:
            if roi is not None:
                plan.header_action = "roi"
            elif not pyramids:
                plan.header_action = "single_resolution"
            else:
                plan.header_action = "copy"
    return plan


# --- Writing -------------------------------------------------------------- #


def _iter_slabs(
    data: Any,
    plan: ChannelPlan,
    roi: Roi | None,
    chunk_z: int,
    cancel_check: CancelCheck | None,
):
    """Yield ``(out_z_start, slab)`` for one channel, cropped to *roi*.

    *cancel_check* is polled before each read; when it returns True the
    generator stops early and :class:`_Cancelled` is raised.
    """
    nz = plan.output_shape[0]
    if roi is None:
        z0, y0, y1, x0, x1 = (
            0,
            0,
            plan.source_shape[1],
            0,
            plan.source_shape[2],
        )
    else:
        z0, _, y0, y1, x0, x1 = roi
    for start in range(0, nz, chunk_z):
        if cancel_check and cancel_check():
            raise _Cancelled
        end = min(start + chunk_z, nz)
        yield start, np.ascontiguousarray(
            data[z0 + start : z0 + end, y0:y1, x0:x1]
        )


class _Cancelled(Exception):
    """Raised inside a channel writer when the user cancels."""


def _export_channel_tiff(
    plan: ChannelPlan,
    roi: Roi | None,
    chunk_z: int,
    progress: ProgressCallback,
    cancel_check: CancelCheck | None,
) -> None:
    """Write one channel as BigTIFF."""
    done = 0
    with (
        h5py.File(plan.source, "r") as src,
        tifffile.TiffWriter(str(plan.output), bigtiff=True) as tw,
    ):
        for _start, slab in _iter_slabs(
            src["Data"], plan, roi, chunk_z, cancel_check
        ):
            for plane in slab:
                tw.write(plane, photometric="minisblack", contiguous=True)
            done += slab.nbytes
            progress(done, plan.output_bytes)


def _export_channel_h5(
    plan: ChannelPlan,
    roi: Roi | None,
    chunk_z: int,
    progress: ProgressCallback,
    cancel_check: CancelCheck | None,
) -> None:
    """Write one channel as .lux.h5."""
    out_shape = plan.output_shape
    with (
        h5py.File(plan.source, "r") as src,
        h5py.File(plan.output, "w") as out,
    ):
        src_data = src["Data"]
        src_chunks = src_data.chunks or (64, 64, 64)
        ds = out.create_dataset(
            "Data",
            shape=out_shape,
            dtype=plan.dtype,
            chunks=tuple(
                min(c, d) for c, d in zip(src_chunks, out_shape, strict=True)
            ),
        )

        pyramid_writer = None
        if plan.pyramid_levels:
            pyramid_writer = StreamingPyramidWriter(
                out,
                plan.pyramid_levels,
                out_shape,
                {
                    name: getattr(src.get(name), "chunks", None)
                    for name, *_ in plan.pyramid_levels
                },
            )

        done = 0
        for start, slab in _iter_slabs(
            src_data, plan, roi, chunk_z, cancel_check
        ):
            ds[start : start + slab.shape[0]] = slab
            done += slab.nbytes
            if pyramid_writer is not None:
                done += pyramid_writer.consume(slab, start)
            del slab  # release before the next (possibly multi-GiB) read
            progress(done, plan.output_bytes)

        if "metadata" in src:
            out.create_dataset("metadata", data=src["metadata"][()])

        if pyramid_writer is not None:
            summary = pyramid_writer.finish()
            logger.info(
                "Pyramids for %s: levels=%s compute=%.1fs write=%.1fs",
                plan.output.name,
                ",".join(summary["levels"]) or "-",
                summary["compute_s"],
                summary["write_s"],
            )


def _write_headers(plan: ExportPlan) -> list[Path]:
    if plan.header_action == "copy":
        return copy_companion_header_files(plan.header_files, plan.output_dir)
    if plan.header_action == "single_resolution":
        return write_single_resolution_headers(
            plan.header_files,
            plan.output_dir,
            output_shape_zyx=plan.channels[0].output_shape,
        )
    if plan.header_action == "roi":
        return write_roi_headers(
            plan.header_files,
            plan.output_dir,
            ROI_SUFFIX,
            plan.roi,
            plan.write_pyramids,
        )
    return []


def _write_summary(plan: ExportPlan, headers: list[Path]) -> Path:
    from . import __version__

    summary: dict[str, Any] = {
        "format": plan.fmt,
        "export_region": "roi" if plan.roi is not None else "full_volume",
        "pyramids_written": plan.write_pyramids,
        "channels": [
            {
                "source": str(ch.source),
                "output": ch.output.name,
                "shape_zyx": list(ch.output_shape),
                "pyramid_levels": [lvl[0] for lvl in ch.pyramid_levels],
            }
            for ch in plan.channels
        ],
        "companion_header_files": [p.name for p in headers],
        "bytes_written": plan.total_bytes,
        "export_date": datetime.now().isoformat(timespec="seconds"),
        "software": f"napari-luxendo-image {__version__}",
    }
    if plan.roi is not None:
        keys = ("z_start", "z_end", "y_start", "y_end", "x_start", "x_end")
        summary["roi_bounds"] = dict(
            zip(keys, (int(v) for v in plan.roi), strict=True)
        )
    path = plan.output_dir / SUMMARY_FILENAME
    path.write_text(json.dumps(summary, indent=2))
    return path


def run_export(
    plan: ExportPlan,
    ram_percent: int = 90,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> Path | None:
    """Execute *plan*, channel by channel.

    Parameters
    ----------
    plan : ExportPlan
        From :func:`plan_export`.
    ram_percent : int
        Percentage of currently available RAM a slab may use.
    progress_callback : callable, optional
        Called with ``(bytes_written, total_bytes)`` across all channels.
    cancel_check : callable, optional
        Polled before each slab; return True to abort. The partially written
        file of the current channel is deleted.

    Returns
    -------
    Path or None
        The JSON export summary, or None if cancelled.
    """
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    if plan.write_pyramids:
        logger.info("Pyramid reduction backend: %s", pyramid_backend_status())

    total = plan.total_bytes
    finished_bytes = 0
    for ch in plan.channels:
        chunk_z = compute_chunk_size(
            ch.output_shape[1:], ram_percent, ch.dtype.itemsize
        )

        def _progress(done: int, _total: int, base: int = finished_bytes):
            if progress_callback:
                progress_callback(base + done, total)

        writer = (
            _export_channel_h5
            if plan.fmt == FORMAT_LUX_H5
            else _export_channel_tiff
        )
        try:
            writer(ch, plan.roi, chunk_z, _progress, cancel_check)
        except _Cancelled:
            # The writer's context managers have closed the file by now.
            ch.output.unlink(missing_ok=True)
            logger.info("Export cancelled during %s", ch.output.name)
            return None
        finished_bytes += ch.output_bytes
        if progress_callback:
            progress_callback(finished_bytes, total)

    headers = _write_headers(plan)
    return _write_summary(plan, headers)
