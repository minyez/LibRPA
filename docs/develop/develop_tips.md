# Development Tips

## Git collaborative workflow

The recommended workflow is:

1. Fork the repository.
2. Create a topic branch for your changes.
3. Rebase your branch on `master` when it is ready for review.
4. Resolve any conflicts locally.
5. Open a merge request.
6. Address review comments.
7. Merge with a fast-forward merge.

## Fortran binding

When updating the Fortran binding, make sure that the `LibrpaOptions` derived
type in `binding/fortran/librpa_f03.f90` matches the definition in
`include/librpa_options.h`.

After modifying `librpa_f03.f90`, run the stub-conversion script so that the
stub module stays synchronized with the main binding module.

```bash
cd binding/fortran
../../utilities/convert_fortran_module_to_stub.py librpa_f03.f90 librpa_f03_stubs.f90
```

## Code style

LibRPA does not enforce a strict project-wide code style. However, the repository
does provide a `.clang-format` file for formatting C and C++ code with
`clang-format`.

The goal is to avoid spending review time on personal formatting preferences,
such as whether an opening brace should be placed on a new line. The
`.clang-format` settings naturally encode some preferences, but the main purpose
is to make formatting mechanical and consistent.

Use `clang-format` for new C and C++ code. Avoid reformatting old code unless you
are making substantial changes in the same area, or unless the commit is
explicitly dedicated to formatting cleanup. This keeps functional diffs focused
and easier to review. Modern editors with language-server support should be able
to pick up the repository formatting rules automatically.

For naming, follow these general conventions:

- Use `UpperCamelCase` for classes and structs.
- Use `snake_case` for variables, object instances, functions, and namespaces.
- Use `SCREAMING_SNAKE_CASE` for constants.

## A few C++ guidelines

- Avoid forward declarations for concrete classes and structs; include the
  header that provides the necessary definition.
- Prefer RAII, standard containers, references, and `const` correctness; make
  ownership and mutation explicit.
- Treat MPI layout, rank ownership, and collectives as part of the program
  behavior; check both serial and representative multi-rank cases when they may
  be affected.
- Use clear, explicit control flow for indexing, basis mappings, matrix layouts,
  and symmetry logic. Add a short comment when the physical or parallel
  assumption is not obvious from the code.
- When adding code in `src/core`, consider whether reusable pieces belong in
  utility components such as `src/math`, `src/mpi`, `src/io`, or `src/utils`;
  implement those pieces in the appropriate place and assemble the core
  algorithm under `src/core`.
- Source files outside `src/core` should not include headers from `src/core`.
- Files in `src/interface` should not include internal headers outside
  `src/interface`.
- Use `src/io/stl_io_helper.h` when printing STL containers.

## Adding new runtime options

- `LibrpaSwitch`-type runtime options must begin with `output_` or `use_`, and vice versa.
- Option with a few available values can either be an `int` type starting with `option_`,
  or a dedicated `enum` type defined in `librpa_enums.h`.
