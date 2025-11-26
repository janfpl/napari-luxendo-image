import json

import h5py
import numpy as np
from dask import array as da


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        if any(not str(p).lower().endswith(".lux.h5") for p in path):
            return None
    else:
        if not str(path).lower().endswith(".lux.h5"):
            return None

    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    if isinstance(path, str):
        data, metadata = _load_lux_h5(path)
        metadata = json.loads(metadata)
    elif isinstance(path, list):
        _data, _metadata = zip(*[_load_lux_h5(p) for p in path], strict=True)
        n_scales = len(_data[0])
        data = [
            da.stack([d[s] for d in _data], axis=0) for s in range(n_scales)
        ]
        metadata = json.loads(_metadata[0])

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    voxel_size = metadata.get("processingInformation", {}).get(
        "voxel_size_um", None
    )
    if voxel_size is not None:
        add_kwargs["scale"] = (
            voxel_size["depth"],
            voxel_size["height"],
            voxel_size["width"],
        )

    layer_type = "image"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]


def _load_lux_h5(path):
    """Load a single .lux.h5 file.

    Parameters
    ----------
    path : str
        Path to the .lux.h5 file.

    Returns
    -------
    data : list of dask.array.Array
        The image data contained in the .lux.h5 file as a list of dask arrays.
    metadata : str
        The metadata contained in the .lux.h5 file.
    """
    file = h5py.File(path, "r")
    data_keys = sorted([key for key in file if key.startswith("Data")])
    # TODO: handle nested dataset structure
    if "Data" not in data_keys:
        raise ValueError("No 'Data' datasets found in .lux.h5 file.")
    if "metadata" not in file:
        raise ValueError("No 'metadata' key found in .lux.h5 file.")

    data = []
    for key in data_keys:
        dataset = file[key]
        dask_array = da.from_array(
            # TODO: make chunk size multiplier configurable
            dataset,
            chunks=tuple(x * 4 for x in dataset.chunks),
        )
        data.append(dask_array)

    metadata = np.array(file["metadata"]).item().decode("ascii")

    return data, metadata
