# Optimized LDpred2 source

This is a clean bigsnpr source checkout with the best accepted LDpred2
optimization integrated into its normal package build. Extract this
directory and upload its contents to a GitHub repository, or initialize
a new Git repository here. Git history is not included.

## Install

Install R and the normal bigsnpr dependencies listed in DESCRIPTION.
The tested bigsparser version is 0.7.3. One way to obtain the dependencies
is to run, from this directory, in R:

```r
install.packages("remotes")
remotes::install_deps(".", dependencies = NA)
```

Then install this source package:

```sh
R CMD INSTALL .
```

No external header overlay, custom user Makevars, absolute paths, or
separately patched bigsparser installation is required. The relative
`-Ivendor` include path in `src/Makevars` selects the bundled header.
The bigsparser dependency itself is still required.

## Changes

- `src/vendor/bigsparser/SFBM.h`: evaluated optimized compact sparse AXPY.
- `src/Makevars`: relative include-path wiring for that header.
- This document, header attribution, and SOURCE_PROVENANCE.json.
The LDpred2 sampler sources and public R API are unchanged.

## Validation and scope

The evaluated build produced bitwise-identical posterior-mean beta vectors
for all nine paired real-data comparisons: GIANT height, BMI, and WHRadjBMI,
three seeds each, 957,587 matched variants on a shared HapMap3/UKB reference.
Measured speedups were 1.85–1.87x on the evaluation machine; they are not
promises for other processors, compilers, data, or sampler settings.
Benchmark outputs and large evaluation inputs are intentionally excluded.

The evaluated environment was Linux x86-64 with GCC and AVX-512F support.
ISA selection is performed at runtime; AVX2 and scalar paths are included.
Other platforms have not been independently validated. Do not enable
`-ffast-math` or floating-point reassociation if exact reproducibility
is required. The AVX-512 path retains its FMA-blocking assembly barrier.

## Licensing

bigsnpr and the bundled bigsparser header are GPL-3; upstream authorship
and package metadata are retained. The GPL v3 text is in COPYING.
