"""Synthetic Luxendo .lux.h5 files and companion headers for tests."""

import json

import h5py
import numpy as np
import pytest

VOXEL_UM = {"width": 0.5, "height": 0.5, "depth": 2.0}


def block_mean(data, fw, fh, fd):
    """Reference pyramid level: floor of the (fd, fh, fw) block mean."""
    nz, ny, nx = data.shape
    oz, oy, ox = nz // fd, ny // fh, nx // fw
    trimmed = data[: oz * fd, : oy * fh, : ox * fw].astype(np.uint64)
    sums = trimmed.reshape(oz, fd, oy, fh, ox, fw).sum(axis=(1, 3, 5))
    return (sums // (fw * fh * fd)).astype(np.uint16)


def write_lux_h5(path, data, levels=((2, 2, 2), (4, 4, 4))):
    meta = {
        "processingInformation": {
            "voxel_size_um": VOXEL_UM,
            "image_size_vx": {
                "width": data.shape[2],
                "height": data.shape[1],
                "depth": data.shape[0],
            },
        }
    }
    with h5py.File(path, "w") as f:
        f.create_dataset("Data", data=data, chunks=(4, 8, 8))
        for fw, fh, fd in levels:
            f.create_dataset(
                f"Data_{fw}_{fh}_{fd}",
                data=block_mean(data, fw, fh, fd),
                chunks=(2, 4, 4),
            )
        f.create_dataset("metadata", data=json.dumps(meta).encode("ascii"))


def _char(text):
    return np.frombuffer(str(text).encode("ascii"), dtype=np.uint8)


def write_ims_header(
    path, channel_files, shape, levels=((2, 2, 2), (4, 4, 4))
):
    """Minimal Imaris header linking to each channel's Data/Data_W_H_D."""
    nz, ny, nx = shape
    paths = ["Data"] + [f"Data_{w}_{h}_{d}" for w, h, d in levels]
    with h5py.File(path, "w") as f:
        f.attrs.create("ImarisDataSet", _char("ImarisDataSet"))
        for lvl, link_path in enumerate(paths):
            for c, ch_file in enumerate(channel_files):
                grp = f.create_group(
                    f"DataSet/ResolutionLevel {lvl}/TimePoint 0/Channel {c}"
                )
                grp["Data"] = h5py.ExternalLink(ch_file, link_path)
        img = f.create_group("DataSetInfo/Image")
        for key, val in (("X", nx), ("Y", ny), ("Z", nz)):
            img.attrs.create(key, _char(val))
        for axis, n, vox in ((0, nx, 0.5), (1, ny, 0.5), (2, nz, 2.0)):
            img.attrs.create(f"ExtMin{axis}", _char(0))
            img.attrs.create(f"ExtMax{axis}", _char(n * vox))
        f.create_group("DataSetInfo/Channel 0")
        f.create_group("DataSetInfo/Imaris")  # viewer residue, dropped


def write_bdv_header(path, channel_files, levels=((2, 2, 2), (4, 4, 4))):
    paths = ["Data"] + [f"Data_{w}_{h}_{d}" for w, h, d in levels]
    factors = [(1, 1, 1), *levels]
    with h5py.File(path, "w") as f:
        for c, ch_file in enumerate(channel_files):
            for lvl, link_path in enumerate(paths):
                f[f"t00000/s{c:02d}/{lvl}/cells"] = h5py.ExternalLink(
                    ch_file, link_path
                )
            f.create_dataset(
                f"s{c:02d}/resolutions", data=np.array(factors, float)
            )
            f.create_dataset(
                f"s{c:02d}/subdivisions", data=np.full((len(paths), 3), 8)
            )


@pytest.fixture
def lux_dataset(tmp_path):
    """Two channel .lux.h5 files (with pyramids) plus .ims/_bdv headers."""
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(0)
    shape = (21, 30, 34)  # deliberately not divisible by the factors
    names = ["stack_ch0.lux.h5", "stack_ch1.lux.h5"]
    arrays = []
    for name in names:
        data = rng.integers(0, 60000, size=shape, dtype=np.uint16)
        write_lux_h5(src / name, data)
        arrays.append(data)
    write_ims_header(src / "stack.ims", names, shape)
    write_bdv_header(src / "stack_bdv.h5", names)
    (src / "stack_bdv.xml").write_text("<SpimData/>")
    return {
        "dir": src,
        "paths": [src / n for n in names],
        "arrays": arrays,
        "shape": shape,
    }
