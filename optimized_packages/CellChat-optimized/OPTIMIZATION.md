# Optimized CellChat source

This tree contains the current accepted optimization of `computeCommunProb`.
It is based on `jinworks/CellChat` commit
`75253cd0c9e68410e6e721a6d3a0419a1d7e358f`, with the cumulative iter16 patch
already applied. No additional patch application is needed.

## Build and install

The package name remains **CellChat**, and its existing API is unchanged.
Install the dependencies listed in `DESCRIPTION`, including those supplied
through Bioconductor, and use an R-compatible C++ toolchain. From the parent
directory of this extracted folder:

```sh
R CMD INSTALL CellChat-optimized
```

Alternatively, from R, with `remotes` installed:

```r
remotes::install_local("CellChat-optimized", dependencies = NA, upgrade = "never")
library(CellChat)
```

To publish to GitHub, upload the **contents** of this directory so that
`DESCRIPTION`, `R/`, and `src/` are at the repository root. After publication:

```r
remotes::install_github("YOUR_ACCOUNT/YOUR_REPOSITORY", upgrade = "never")
```

This ZIP contains source, not an installed R library. Native code is rebuilt
on installation; no machine-specific compiled objects are shipped.

## Optimization and validation scope

Changes reduce permutation-loop overhead, reuse invariant lookups, restrict
aggregation to referenced genes, and use exact C++ triMean/order-statistic
kernels. The accepted implementation is serial; the experimental OpenMP
candidate is not included.

The final evaluated build produced identical `net$prob` and `net$pval`
outputs on the tested human skin, mouse embryonic skin, and mouse wound
datasets. This is evidence for those tested inputs and environment, not a
guarantee for every dataset or platform. This source-only ZIP does not include
evaluation datasets, logs, installed dependencies, or the evaluation archive.

## Upstream attribution and license

Original authorship, package metadata, documentation, bundled reference
databases, and GPL-3 licensing are retained. This is a modified source tree,
not an official upstream release. See the original README below and LICENSE.
