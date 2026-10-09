"""Imaris / BigDataViewer companion headers for exported Luxendo data.

Ported from shifter (``shifter/h5_utils.py``).

Luxendo acquisitions sometimes ship sidecar headers next to the per-channel
.lux.h5 files: an Imaris ``.ims`` header and a BigDataViewer
``*_bdv.h5`` / ``*_bdv.xml`` pair. They describe a *multi-resolution* dataset
whose levels are HDF5 external links into the per-channel files (``Data``,
``Data_2_2_2``, ...), referenced by their literal filenames. An export must
therefore reconcile the headers with what it actually wrote:

* full volume + pyramids   -> copy verbatim (headers already match)
* full volume, no pyramids -> rebuild with only the full-resolution level,
  otherwise links to the absent pyramids make viewers read the data as corrupt
* ROI                      -> rebuild pointing at the ROI filenames, with the
  cropped dimensions / extent (and a single level when pyramids are off)
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ._pyramids import dataset_path_factors

logger = logging.getLogger(__name__)

_HEADER_FILE_GLOBS = ("*.ims", "*_bdv.h5", "*_bdv.xml")


def lux_output_filename(original_filename: str, suffix: str) -> str:
    """Build the export filename for a Luxendo H5 channel.

    Handles the double ``.lux.h5`` extension (``foo.lux.h5`` + ``_roi`` ->
    ``foo_roi.lux.h5``, not ``foo.lux_roi.h5``).
    """
    if original_filename.lower().endswith(".lux.h5"):
        stem = original_filename[: -len(".lux.h5")]
        return f"{stem}{suffix}.lux.h5"
    if original_filename.lower().endswith(".h5"):
        stem = original_filename[: -len(".h5")]
        return f"{stem}{suffix}.h5"
    return f"{original_filename}{suffix}.lux.h5"


def find_companion_header_files(input_dir: Path | str) -> list[Path]:
    """Return sidecar Imaris/BigDataViewer header files in *input_dir*."""
    input_dir = Path(input_dir)
    found: list[Path] = []
    for pattern in _HEADER_FILE_GLOBS:
        found.extend(sorted(input_dir.glob(pattern)))
    return found


def copy_companion_header_files(
    header_files: list[Path], output_dir: Path | str
) -> list[Path]:
    """Copy *header_files* verbatim into *output_dir*, preserving filenames."""
    output_dir = Path(output_dir)
    copied: list[Path] = []
    for src in header_files:
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def write_single_resolution_headers(
    header_files: list[Path],
    output_dir: Path | str,
    output_shape_zyx: tuple[int, int, int] | None = None,
) -> list[Path]:
    """Write headers into *output_dir* describing full resolution only.

    The .ims and *_bdv.h5 are rebuilt from scratch; a *_bdv.xml is copied
    verbatim (its resolution info lives in the paired .h5).
    """
    output_dir = Path(output_dir)
    written: list[Path] = []
    for src in header_files:
        dst = output_dir / src.name
        name = src.name.lower()
        try:
            if name.endswith(".ims"):
                rebuild_ims_header(
                    src,
                    dst,
                    output_shape_zyx=output_shape_zyx,
                    keep_paths={"Data"},
                )
            elif name.endswith("_bdv.h5"):
                rebuild_bdv_h5_header(src, dst, keep_paths={"Data"})
            else:
                shutil.copy2(src, dst)
            written.append(dst)
        except Exception as exc:  # noqa: BLE001 - headers are best effort
            logger.warning(
                "Could not write single-resolution header %s: %s",
                src.name,
                exc,
            )
    return written


def write_roi_headers(
    header_files: list[Path],
    output_dir: Path | str,
    output_suffix: str,
    roi: tuple[int, int, int, int, int, int],
    write_pyramids: bool,
) -> list[Path]:
    """Write companion headers matching an ROI-cropped export.

    Each ``.ims`` / ``*_bdv.h5`` is rebuilt with its external links pointing
    at the ``output_suffix``-named output files, reduced to a single level
    when *write_pyramids* is False, and (for ``.ims``) given the ROI's
    dimensions and cropped extent. A ``*_bdv.xml`` is skipped: its ViewSetup
    dimensions can't be regenerated reliably here.
    """
    output_dir = Path(output_dir)

    def rename(filename: str) -> str:
        return lux_output_filename(filename, output_suffix)

    z0, z1, y0, y1, x0, x1 = roi
    out_shape = (z1 - z0, y1 - y0, x1 - x0)
    keep_paths = None if write_pyramids else {"Data"}

    written: list[Path] = []
    for src in header_files:
        name = src.name.lower()
        dst = output_dir / src.name
        try:
            if name.endswith(".ims"):
                rebuild_ims_header(
                    src,
                    dst,
                    rename=rename,
                    output_shape_zyx=out_shape,
                    roi=roi,
                    keep_paths=keep_paths,
                )
                written.append(dst)
            elif name.endswith("_bdv.h5"):
                rebuild_bdv_h5_header(
                    src, dst, rename=rename, keep_paths=keep_paths
                )
                written.append(dst)
            else:
                logger.info(
                    "Skipping %s for ROI export (BDV XML dimensions not "
                    "regenerated).",
                    src.name,
                )
        except Exception as exc:  # noqa: BLE001 - headers are best effort
            logger.warning("Could not write ROI header %s: %s", src.name, exc)
    return written


# --------------------------------------------------------------------------- #
# Manufacturer-style header rebuilding
#
# The builders write a NEW header containing exactly the structure the
# manufacturer emits: minimal root markers, the requested resolution levels
# (each channel carrying its own ImageSizeX/Y/Z), and DataSetInfo with only
# Channel/Image/TimeInfo. Imaris text attributes are written as uint8 arrays.
# Output goes to a temporary file that is moved into place, so a failure never
# leaves a half-written header.
# --------------------------------------------------------------------------- #

_IMS_ROOT_ATTR_DEFAULTS: dict[str, str] = {
    "DataSetDirectoryName": "DataSet",
    "DataSetInfoDirectoryName": "DataSetInfo",
    "ImarisDataSet": "ImarisDataSet",
    "ImarisVersion": "5.5.0",
    "ThumbnailDirectoryName": "Thumbnail",
}

# DataSetInfo children the manufacturer emits; anything else is Imaris Viewer
# residue and is dropped.
_IMS_INFO_KEEP_RE = re.compile(r"^(Channel \d+|Image|TimeInfo)$")


def _ims_char_attr(text: object) -> np.ndarray:
    """Encode *text* as Imaris' uint8 character-array attribute format."""
    return np.frombuffer(str(text).encode("ascii"), dtype=np.uint8).copy()


def _ims_read_char_attr(arr: object) -> str:
    """Decode an Imaris char-array attribute back to a string."""
    if isinstance(arr, bytes):
        return arr.decode("ascii", "replace")
    if isinstance(arr, str):
        return arr
    try:
        return arr.tobytes().decode("ascii", "replace")  # type: ignore[attr-defined]
    except AttributeError:
        return str(arr)


def _sorted_resolution_levels(dataset_grp: Any) -> list[str]:
    """Return ``ResolutionLevel N`` names sorted by their numeric index."""

    def _idx(name: str) -> int:
        try:
            return int(name.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            return 0

    return sorted(
        (k for k in dataset_grp if k.startswith("ResolutionLevel")), key=_idx
    )


def _copy_attrs_as_ims_text(src_grp: Any, dst_grp: Any) -> None:
    """Copy *src_grp*'s attributes to *dst_grp* in Imaris uint8 text form."""
    for key, value in src_grp.attrs.items():
        dst_grp.attrs.create(key, _ims_char_attr(_ims_read_char_attr(value)))


def rebuild_ims_header(
    src_path: Path | str,
    dst_path: Path | str,
    *,
    rename: Callable[[str], str] | None = None,
    output_shape_zyx: tuple[int, int, int] | None = None,
    roi: tuple[int, int, int, int, int, int] | None = None,
    keep_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Write a fresh, manufacturer-style Imaris header derived from *src_path*.

    Parameters
    ----------
    rename : callable, optional
        Maps each external-link filename to the exported filename.
    output_shape_zyx : tuple, optional
        ``(nz, ny, nx)`` of the exported full-resolution data. Each channel's
        ``ImageSizeX/Y/Z`` is derived from this and the level's own
        downsample factors.
    roi : tuple, optional
        ``(z0, z1, y0, y1, x0, x1)``; crops ``DataSetInfo/Image``'s physical
        extent while preserving voxel size.
    keep_paths : set[str], optional
        Dataset paths that exist in the output (e.g. ``{"Data"}`` when
        pyramids were skipped). Levels linking elsewhere are omitted.
    """
    import h5py

    src_path = Path(src_path)
    dst_path = Path(dst_path)
    tmp_path = dst_path.with_name(dst_path.name + ".tmp")

    written_levels: list[str] = []
    n_channels = 0

    with (
        h5py.File(str(src_path), "r") as src,
        h5py.File(str(tmp_path), "w") as dst,
    ):
        # --- root markers ------------------------------------------------
        for key, default in _IMS_ROOT_ATTR_DEFAULTS.items():
            value = (
                _ims_read_char_attr(src.attrs[key])
                if key in src.attrs
                else default
            )
            dst.attrs.create(key, _ims_char_attr(value))

        # --- DataSet ------------------------------------------------------
        src_ds = src["DataSet"]
        dst_ds = dst.create_group("DataSet")
        out_index = 0
        for level_name in _sorted_resolution_levels(src_ds):
            src_level = src_ds[level_name]

            # Decide whether this level survives, based on what it links to.
            level_paths: set[str] = set()
            for tp in src_level:
                for ch in src_level[tp]:
                    link = src_level[tp][ch].get("Data", getlink=True)
                    if isinstance(link, h5py.ExternalLink):
                        level_paths.add(link.path.strip("/").split("/")[-1])
            if keep_paths is not None and not (level_paths & keep_paths):
                continue

            dst_level = dst_ds.create_group(f"ResolutionLevel {out_index}")
            out_index += 1
            written_levels.append(level_name)

            for tp_name in sorted(src_level.keys()):
                src_tp = src_level[tp_name]
                dst_tp = dst_level.create_group(tp_name)
                for ch_name in sorted(src_tp.keys()):
                    src_ch = src_tp[ch_name]
                    dst_ch = dst_tp.create_group(ch_name)
                    link = src_ch.get("Data", getlink=True)

                    factors = (1, 1, 1)
                    if isinstance(link, h5py.ExternalLink):
                        factors = dataset_path_factors(link.path)
                        target = link.filename
                        if rename is not None:
                            target = rename(target)
                        dst_ch["Data"] = h5py.ExternalLink(target, link.path)

                    if "Histogram" in src_ch:
                        dst_ch.create_dataset(
                            "Histogram", data=src_ch["Histogram"][()]
                        )
                    for key, default in (
                        ("HistogramMin", "0"),
                        ("HistogramMax", "65535"),
                    ):
                        value = (
                            _ims_read_char_attr(src_ch.attrs[key])
                            if key in src_ch.attrs
                            else default
                        )
                        dst_ch.attrs.create(key, _ims_char_attr(value))

                    # Per-level dimensions, derived from the linked level's
                    # factors so they always agree with the linked dataset.
                    if output_shape_zyx is not None:
                        nz, ny, nx = output_shape_zyx
                        fw, fh, fd = factors
                        dst_ch.attrs.create(
                            "ImageSizeX", _ims_char_attr(nx // fw)
                        )
                        dst_ch.attrs.create(
                            "ImageSizeY", _ims_char_attr(ny // fh)
                        )
                        dst_ch.attrs.create(
                            "ImageSizeZ", _ims_char_attr(nz // fd)
                        )
                    else:
                        for key in ("ImageSizeX", "ImageSizeY", "ImageSizeZ"):
                            if key in src_ch.attrs:
                                dst_ch.attrs.create(
                                    key,
                                    _ims_char_attr(
                                        _ims_read_char_attr(src_ch.attrs[key])
                                    ),
                                )
                    n_channels = max(n_channels, len(src_tp.keys()))

        # --- DataSetInfo (Channel N / Image / TimeInfo only) --------------
        dst_info = dst.create_group("DataSetInfo")
        if "DataSetInfo" in src:
            src_info = src["DataSetInfo"]
            for name in src_info:
                if _IMS_INFO_KEEP_RE.match(name):
                    _copy_attrs_as_ims_text(
                        src_info[name], dst_info.create_group(name)
                    )

            if "Image" in dst_info and "Image" in src_info:
                dst_image = dst_info["Image"]
                src_image = src_info["Image"]

                def _num(key: str) -> float | None:
                    if key not in src_image.attrs:
                        return None
                    try:
                        return float(_ims_read_char_attr(src_image.attrs[key]))
                    except ValueError:
                        return None

                if roi is not None:
                    z0, z1, y0, y1, x0, x1 = roi
                    for axis, dim_key, start, end in (
                        (0, "X", x0, x1),
                        (1, "Y", y0, y1),
                        (2, "Z", z0, z1),
                    ):
                        n = _num(dim_key)
                        mn = _num(f"ExtMin{axis}")
                        mx = _num(f"ExtMax{axis}")
                        if None in (n, mn, mx) or not n:
                            continue
                        vox = (mx - mn) / n
                        dst_image.attrs.create(
                            f"ExtMin{axis}",
                            _ims_char_attr(f"{mn + start * vox:.6g}"),
                        )
                        dst_image.attrs.create(
                            f"ExtMax{axis}",
                            _ims_char_attr(f"{mn + end * vox:.6g}"),
                        )

                if output_shape_zyx is not None:
                    nz, ny, nx = output_shape_zyx
                    dst_image.attrs.create("X", _ims_char_attr(nx))
                    dst_image.attrs.create("Y", _ims_char_attr(ny))
                    dst_image.attrs.create("Z", _ims_char_attr(nz))

    os.replace(str(tmp_path), str(dst_path))
    return {"levels": written_levels, "channels": n_channels}


def rebuild_bdv_h5_header(
    src_path: Path | str,
    dst_path: Path | str,
    *,
    rename: Callable[[str], str] | None = None,
    keep_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Write a fresh BigDataViewer HDF5 header derived from *src_path*.

    Keeps only the levels whose ``cells`` link targets are in *keep_paths*
    (all levels when it is None), truncating each setup's ``resolutions`` /
    ``subdivisions`` tables to match, and remapping link filenames via
    *rename*.
    """
    import h5py

    src_path = Path(src_path)
    dst_path = Path(dst_path)
    tmp_path = dst_path.with_name(dst_path.name + ".tmp")

    kept_per_setup: dict[str, list[str]] = {}

    with (
        h5py.File(str(src_path), "r") as src,
        h5py.File(str(tmp_path), "w") as dst,
    ):
        timepoints = [k for k in src if re.fullmatch(r"t\d+", k)]

        for t in sorted(timepoints):
            for s in sorted(src[t].keys()):
                src_grp = src[f"{t}/{s}"]
                keep: list[str] = []
                for lvl in sorted(
                    src_grp.keys(), key=lambda v: int(v) if v.isdigit() else 0
                ):
                    link = src_grp[lvl].get("cells", getlink=True)
                    if not isinstance(link, h5py.ExternalLink):
                        continue
                    leaf = link.path.strip("/").split("/")[-1]
                    if keep_paths is not None and leaf not in keep_paths:
                        continue
                    target = link.filename
                    if rename is not None:
                        target = rename(target)
                    dst.create_group(f"{t}/{s}/{len(keep)}")
                    dst[f"{t}/{s}/{len(keep)}/cells"] = h5py.ExternalLink(
                        target, link.path
                    )
                    keep.append(lvl)
                kept_per_setup.setdefault(s, keep)

        # Resolution tables truncated to the number of surviving levels.
        for s in [k for k in src if re.fullmatch(r"s\d+", k)]:
            n_keep = len(kept_per_setup.get(s, []))
            for arr in ("resolutions", "subdivisions"):
                key = f"{s}/{arr}"
                if key not in src:
                    continue
                data = src[key][:]
                if n_keep:
                    data = data[:n_keep]
                dst.create_dataset(key, data=data)

    os.replace(str(tmp_path), str(dst_path))
    return {"levels_per_setup": kept_per_setup}
