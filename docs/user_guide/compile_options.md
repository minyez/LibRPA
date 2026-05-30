# Compile Options

## Overview

| Option                                                      | Type              | Default    |
|-------------------------------------------------------------|-------------------|------------|
| [`LIBRPA_USE_LIBRI`](#librpa-use-libri)                     | bool              | `OFF`      |
| [`LIBRPA_USE_CMAKE_INC`](#librpa-use-cmake-inc)             | bool              | `OFF`      |
| [`LIBRPA_USE_EXTERNAL_GREENX`](#librpa-use-external-greenx) | bool              | `OFF`      |
| [`LIBRPA_ENABLE_FORTRAN_BIND`](#librpa-enable-fortran-bind) | bool              | `OFF`      |
| [`LIBRPA_FORTRAN_DP`](#librpa-fortran-dp)                   | string or integer | `c_double` |
| [`LIBRPA_ENABLE_DRIVER`](#librpa-enable-driver)             | bool              | `ON`       |
| [`LIBRPA_ENABLE_TEST`](#librpa-enable-test)                 | bool              | `ON`       |
| [`LIBRPA_ENABLE_CPP_TEST`](#librpa-enable-cpp-test)         | bool              | `ON`       |
| [`LIBRPA_ENABLE_FORTRAN_TEST`](#librpa-enable-fortran-test) | bool              | `ON`       |
| [`LIBRI_INCLUDE_DIR`](#libri-include-dir)                   | string            | empty      |
| [`LIBCOMM_INCLUDE_DIR`](#libcomm-include-dir)               | string            | empty      |
| [`CEREAL_INCLUDE_DIR`](#cereal-include-dir)                 | string            | empty      |
| [`SCALAPACK_DIR`](#scalapack-dir)                           | string            | empty      |
| [`LIBRPA_USE_EXTERNAL_ELPA`](#librpa-use-external-elpa)     | bool              | `OFF`      |
| [`EXTERNAL_ELPA_DIR`](#external-elpa-dir)                   | string            | empty      |
| [`LIBRPA_USE_BUNDLED_ELPA`](#librpa-use-bundled-elpa)       | bool              | `OFF`      |
| [`LIBRPA_BUNDLED_ELPA_VERSION`](#librpa-bundled-elpa-version) | string          | `2021.11.002` |
| [`LIBRPA_BUNDLED_ELPA_KERNEL`](#librpa-bundled-elpa-kernel) | string            | empty      |
| [`LIBRPA_BUNDLED_ELPA_OPENMP`](#librpa-bundled-elpa-openmp) | bool              | `OFF`      |
| [`LIBRPA_BUNDLED_ELPA_CONFIGURE_ARGS`](#librpa-bundled-elpa-configure-args) | string | empty |
| [`LIBRPA_BUNDLED_ELPA_LIBS`](#librpa-bundled-elpa-libs)     | string            | empty      |

These options can be parsed on the CMake command line, for example:

```sh
cmake -DLIBRPA_USE_LIBRI=ON
```

(librpa-use-libri)=
## `LIBRPA_USE_LIBRI`

When enabled, LibRPA is compiled with [LibRI](https://github.com/abacusmodeling/LibRI)
for RI tensor contractions.

The *GW* and EXX functionalities require LibRPA to be compiled with LibRI, i.e. `-DLIBRPA_USE_LIBRI=ON`.
By contrast, the RPA correlation energy can also be computed without this option.

(librpa-use-external-elpa)=
## `LIBRPA_USE_EXTERNAL_ELPA`

When enabled, LibRPA is linked against an external
[ELPA](https://elpa.mpcdf.mpg.de/) installation.

ELPA support is intended for optimized linear algebra subroutines, such as
ELPA-provided dense eigensolver routines, in ELPA-backed implementations. This
option provides the build interface for those code paths.

Set [`EXTERNAL_ELPA_DIR`](#external-elpa-dir) to the ELPA installation prefix
so CMake can find the ELPA headers, Fortran module directory, and library.

Example:
```sh
cmake -DLIBRPA_USE_EXTERNAL_ELPA=ON -DEXTERNAL_ELPA_DIR=/path/to/elpa
```

(librpa-use-bundled-elpa)=
## `LIBRPA_USE_BUNDLED_ELPA`

When enabled, LibRPA builds and links against a bundled ELPA source release
under `thirdparty/ELPA`.

This option is mutually exclusive with
[`LIBRPA_USE_EXTERNAL_ELPA`](#librpa-use-external-elpa).

The bundled ELPA build is managed through CMake's `ExternalProject` mechanism.
After ELPA has been built in an existing build directory, changing compiler
flags or `CMAKE_BUILD_TYPE` may not automatically reconfigure and rebuild ELPA.
Use a fresh build directory, or clean the bundled ELPA sub-build, when those
settings need to be applied to ELPA itself.

Example:
```sh
cmake -DLIBRPA_USE_BUNDLED_ELPA=ON
```

(librpa-bundled-elpa-version)=
## `LIBRPA_BUNDLED_ELPA_VERSION`

Selects which bundled ELPA release is built when
[`LIBRPA_USE_BUNDLED_ELPA`](#librpa-use-bundled-elpa) is enabled.

Supported values are:

- `2021.11.002`
- `2026.02.001`

(librpa-bundled-elpa-kernel)=
## `LIBRPA_BUNDLED_ELPA_KERNEL`

Selects an x86 SIMD kernel family for the bundled ELPA build.

By default, this option is empty. In that case, LibRPA disables ELPA's
x86-specific SIMD kernels and lets ELPA build portable generic kernels. Set this
option on compatible x86 systems when an optimized kernel family is desired.

Supported values are:

- empty
- `SSE`
- `SSE_ASSEMBLY`
- `AVX`
- `AVX2`
- `AVX512`

Example:
```sh
cmake -DLIBRPA_USE_BUNDLED_ELPA=ON \
      -DLIBRPA_BUNDLED_ELPA_KERNEL=AVX512
```

(librpa-bundled-elpa-openmp)=
## `LIBRPA_BUNDLED_ELPA_OPENMP`

Controls whether the bundled ELPA library is built with ELPA's own OpenMP
support.

The default is `OFF`. In that case, ELPA is built as an MPI-only static library,
even if LibRPA itself is compiled with OpenMP. Set this option to `ON` only when
the runtime process and thread layout is chosen with ELPA threading in mind.
For example, with MPI ranks, OpenMP regions in LibRPA, threaded BLAS, and
OpenMP-enabled ELPA all active at the same time, the total number of runnable
threads can exceed the available cores unless `OMP_NUM_THREADS`,
BLAS-specific thread controls, and the MPI rank count are coordinated.

When this option is `ON`, LibRPA also enables ELPA's runtime MPI threading
support checks and allows ELPA to limit its OpenMP thread count when the MPI
library does not provide the thread level ELPA needs.

Example:
```sh
cmake -DLIBRPA_USE_BUNDLED_ELPA=ON \
      -DLIBRPA_BUNDLED_ELPA_OPENMP=ON
```

(librpa-bundled-elpa-configure-args)=
## `LIBRPA_BUNDLED_ELPA_CONFIGURE_ARGS`

Additional arguments passed to the bundled ELPA `configure` script.

Arguments passed through this option are appended after LibRPA's defaults,
including [`LIBRPA_BUNDLED_ELPA_KERNEL`](#librpa-bundled-elpa-kernel), so they
can override the default kernel selection when a specific ELPA setup is needed.

Example:
```sh
cmake -DLIBRPA_USE_BUNDLED_ELPA=ON \
      -DLIBRPA_BUNDLED_ELPA_CONFIGURE_ARGS="--enable-store-build-config"
```

(librpa-bundled-elpa-libs)=
## `LIBRPA_BUNDLED_ELPA_LIBS`

Linker flags passed to the bundled ELPA `configure` script through its `LIBS`
environment variable.

By default, LibRPA forwards the detected LAPACK and ScaLAPACK libraries to the
bundled ELPA build. Set this option only when the autodetected flags are not
suitable for a particular compiler or math library setup.

When static math libraries are used, ELPA's libtool build may try to include
those static archives inside `libelpa.a`. LibRPA removes such nested archive
members after the bundled ELPA install step and links the math libraries
separately through CMake.

Example:
```sh
cmake -DLIBRPA_USE_BUNDLED_ELPA=ON \
      -DLIBRPA_BUNDLED_ELPA_LIBS="-L/path/to/lib -lscalapack -llapack -lblas"
```

(librpa-use-cmake-inc)=
## `LIBRPA_USE_CMAKE_INC`

When enabled, the `cmake.inc` file is used to initialize compilers and other build options.

**Deprecated**. It is recommended to use standard CMake command-line options such as `-C` or `-D` to specify custom variables.

(librpa-use-external-greenx)=
## `LIBRPA_USE_EXTERNAL_GREENX`

Controls whether LibRPA uses the bundled GreenX library or an external one.

The minimax grids used by LibRPA are provided through the
[GreenX](https://nomad-coe.github.io/greenX/) library.

When this option is `OFF` (default), LibRPA builds and links against the bundled GreenX source distributed with LibRPA under `thirdparty/greenX`.

When this option is `ON`, LibRPA does not build the bundled GreenX copy.
Instead, it expects an external GreenX library to be provided by the parent or higher-level CMake project.
In particular, the CMake target `LibGXMiniMax` must already be defined and available for linking.

This option is mainly intended for developer workflows or project setups in which GreenX is managed outside LibRPA.

(librpa-enable-fortran-bind)=
## `LIBRPA_ENABLE_FORTRAN_BIND`

When enabled, the Fortran bindings of LibRPA are built.

(librpa-fortran-dp)=
## `LIBRPA_FORTRAN_DP`

Specifies the Fortran kind used for double-precision real and complex data in the Fortran bindings.

The default value is `c_double`, which is suitable when interoperability with C is desired.
This option may also be set to an integer kind value if needed by the calling code.

This option is meaningful only if `LIBRPA_ENABLE_FORTRAN_BIND=ON`.

(librpa-enable-driver)=
## `LIBRPA_ENABLE_DRIVER`

When enabled, the LibRPA driver executable is built.

(librpa-enable-test)=
## `LIBRPA_ENABLE_TEST`

When enabled, the unit tests of LibRPA are built.

After LibRPA has been compiled successfully, the tests can be run from the build directory with:
```sh
ctest
```
or equivalently
```sh
make test
```

```{note}
At present, the unit tests do not cover the entire code base.
Test coverage is still being expanded.
```

(librpa-enable-cpp-test)=
## `LIBRPA_ENABLE_CPP_TEST`

When enabled, the C++ unit tests are built.

This option is meaningful only if `LIBRPA_ENABLE_TEST=ON`.

(librpa-enable-fortran-test)=
## `LIBRPA_ENABLE_FORTRAN_TEST`

When enabled, the Fortran unit tests are built.

This option is meaningful only if both `LIBRPA_ENABLE_TEST=ON` and `LIBRPA_ENABLE_FORTRAN_BIND=ON`.

(libri-include-dir)=
## `LIBRI_INCLUDE_DIR`

Specifies the path to the LibRI include directory.

If this variable is empty, the internal LibRI copy is used.
Otherwise, CMake searches for `RI/ri/RI_Tools.h` under the specified directory.
An error is raised if the file cannot be found.

Example:
```sh
cmake -DLIBRI_INCLUDE_DIR=/path/to/LibRI/include
```

(libcomm-include-dir)=
## `LIBCOMM_INCLUDE_DIR`

Specifies the path to the LibComm include directory.

If this variable is empty, the internal LibComm copy is used.
Otherwise, CMake searches for `Comm/Comm_Tools.h` under the specified directory.
An error is raised if the file cannot be found.

Example:
```sh
cmake -DLIBCOMM_INCLUDE_DIR=/path/to/LibComm/include
```

(cereal-include-dir)=
## `CEREAL_INCLUDE_DIR`

Specifies the path to the cereal include directory.

If this variable is empty, the bundled cereal copy is used.
Otherwise, CMake searches for `cereal/cereal.hpp` under the specified directory.
An error is raised if the file cannot be found.

Example:
```sh
cmake -DCEREAL_INCLUDE_DIR=/path/to/cereal/include
```

(scalapack-dir)=
## `SCALAPACK_DIR`

`SCALAPACK_DIR` specifies the installation path of ScaLAPACK and is used to
locate the ScaLAPACK libraries when `MKLROOT` is not defined.

This variable can be provided in two ways:

- as a CMake option:

  ```bash
  cmake -DSCALAPACK_DIR=/path/to/scalapack
  ```

- or as an environment variable:

  ```bash
  export SCALAPACK_DIR=/path/to/scalapack
  cmake
  ```

This option is intended for environments where ScaLAPACK is provided as a
standalone installation rather than through Intel MKL.

(external-elpa-dir)=
## `EXTERNAL_ELPA_DIR`

`EXTERNAL_ELPA_DIR` specifies the installation prefix of an external ELPA
library. It is used when
[`LIBRPA_USE_EXTERNAL_ELPA`](#librpa-use-external-elpa) is enabled.

CMake searches below this prefix for:

- headers such as `include/elpa-*/elpa/elpa.h`
- Fortran modules such as `include/elpa-*/modules/elpa.mod`
- libraries such as `lib/libelpa.so` or `lib/libelpa_openmp.so`

This variable can be provided as a CMake option:

```bash
cmake -DLIBRPA_USE_EXTERNAL_ELPA=ON -DEXTERNAL_ELPA_DIR=/path/to/elpa
```

or as an environment variable:

```bash
export EXTERNAL_ELPA_DIR=/path/to/elpa
cmake -DLIBRPA_USE_EXTERNAL_ELPA=ON
```
