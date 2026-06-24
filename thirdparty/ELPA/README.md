# Bundled ELPA

This directory contains bundled source releases of
[ELPA](https://elpa.mpcdf.mpg.de/), used when LibRPA is configured with
`LIBRPA_USE_BUNDLED_ELPA=ON`.

Bundled versions:

- `elpa-2026.02.001`: latest ELPA release listed in the official tarball
  archive as of 2026-05-30.

The CMake wrapper in this directory builds ELPA through its upstream autotools
build system and exposes an `elpa` CMake target for LibRPA.

ELPA is licensed under LGPL-3.0. The upstream license text is included in each
bundled source tree, `elpa-2026.02.001/LICENSE`, with full license copies under each release's
`COPYING/` directory.
