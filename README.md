# napari-luxendo-image

[![License MIT](https://img.shields.io/pypi/l/napari-luxendo-image.svg?color=green)](https://github.com/jason-yang-0/napari-luxendo-image/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-luxendo-image.svg?color=green)](https://pypi.org/project/napari-luxendo-image)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-luxendo-image.svg?color=green)](https://python.org)
[![tests](https://github.com/jason-yang-0/napari-luxendo-image/workflows/tests/badge.svg)](https://github.com/jason-yang-0/napari-luxendo-image/actions)
[![codecov](https://codecov.io/gh/jason-yang-0/napari-luxendo-image/branch/main/graph/badge.svg)](https://codecov.io/gh/jason-yang-0/napari-luxendo-image)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-luxendo-image)](https://napari-hub.org/plugins/napari-luxendo-image)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

Unofficial reader for Luxendo Image format

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template] (None).

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-luxendo-image` via [pip]:

```
pip install napari-luxendo-image
```

If napari is not already installed, you can install `napari-luxendo-image` with napari and Qt via:

```
pip install "napari-luxendo-image[all]"
```

For faster pyramid generation during export, also install numba:

```
pip install "napari-luxendo-image[numba]"
```

## Exporting

**Plugins > Luxendo Image > Export (.lux.h5 / BigTIFF)** opens a panel that
writes a layer opened with this plugin back to disk:

- **Output format**: Luxendo `.lux.h5` (full-resolution `Data` plus the
  original `metadata`) or BigTIFF (one page per Z-plane). A layer stacked from
  several files is exported as one file per channel.
- **Export region**: the full volume, or an ROI. Click *Draw ROI rectangle* to
  set the Y/X bounds from a rectangle in the viewer, use the *Z start/stop =
  current slice* buttons, or type any of the six bounds directly.
- **Pyramid layers** (`.lux.h5` only): regenerate the source file's
  `Data_W_H_D` resolution levels, or untick for the fastest export.
- **Companion headers** (`.lux.h5` only): Imaris `.ims` and BigDataViewer
  `*_bdv.h5`/`*_bdv.xml` headers next to the source files are copied, or
  rebuilt to match an ROI or a pyramid-less export.

Full-volume `.lux.h5` exports keep the original filenames so the headers keep
working; ROI exports get a `_roi` suffix. A `luxendo_export.json` summary is
written alongside. The output directory must differ from the source directory.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-luxendo-image" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
