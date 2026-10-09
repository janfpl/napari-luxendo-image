import numpy as np

from napari_luxendo_image import napari_get_reader
from napari_luxendo_image._reader import SOURCE_PATHS_KEY


def test_reader_single_file(lux_dataset):
    path = str(lux_dataset["paths"][0])
    reader = napari_get_reader(path)
    assert callable(reader)

    layer_data_list = reader(path)
    assert len(layer_data_list) == 1
    data, add_kwargs, layer_type = layer_data_list[0]

    assert layer_type == "image"
    np.testing.assert_array_equal(
        np.asarray(data[0]), lux_dataset["arrays"][0]
    )
    assert add_kwargs["scale"] == (2.0, 0.5, 0.5)
    assert add_kwargs["metadata"][SOURCE_PATHS_KEY] == [path]


def test_reader_stacked_files(lux_dataset):
    paths = [str(p) for p in lux_dataset["paths"]]
    data, add_kwargs, _ = napari_get_reader(paths)(paths)[0]

    assert data[0].shape == (2, *lux_dataset["shape"])
    assert add_kwargs["metadata"][SOURCE_PATHS_KEY] == paths


def test_get_reader_pass():
    assert napari_get_reader("fake.file") is None
    assert napari_get_reader(["a.lux.h5", "b.tif"]) is None
