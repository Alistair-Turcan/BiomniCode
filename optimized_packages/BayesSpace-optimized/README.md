# BayesSpace — optimized source

Current-best optimization revision **ext45**, based on BayesSpace **1.21.2**,
upstream commit `ba8b42211ef1581d31a659766559bd6e42be3acc`.
This is an optimized fork, not an official upstream release. The package name,
version field, public API, license and original authorship are retained.

## Install this version

Extract the ZIP and open a terminal in this directory. Use an R installation
with a compatible C++17 build toolchain and the package dependencies listed in
`DESCRIPTION`. To resolve CRAN/Bioconductor dependencies and install this local
source, run in R:

```r
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
if (!requireNamespace("remotes", quietly = TRUE))
    install.packages("remotes")
options(repos = BiocManager::repositories())
remotes::install_local(".", dependencies = NA, upgrade = "never",
                      build_vignettes = FALSE)
library(BayesSpace)
```

If dependencies are already installed, a direct source install is:

```sh
R CMD INSTALL --no-multiarch --no-docs --no-help .
```

Do not install `BayesSpace` from Bioconductor or the upstream GitHub URL to obtain
these optimizations: those commands install a different source revision.
Fresh dependency installation and cross-platform installation have not been
retested for this packaging step. The recorded evaluation used R 4.5.3; exact
runtime/linking package versions are in `optimization/R_dependency_manifest.tsv`.
Changing compiler, BLAS or dependencies requires revalidation of exact outputs.

## Use

The existing BayesSpace API is unchanged. Original usage documentation is in
`README_UPSTREAM.md`, `man/` and `vignettes/BayesSpace.Rmd`. Native package examples
and test fixtures are included; large external evaluation datasets, chains,
run artifacts, installed libraries and compiled binaries are not.

## Publish to GitHub

Create an empty GitHub repository, then run from this directory, replacing the
remote URL with your own:

```sh
git init
git add .
git commit -m "Import optimized BayesSpace ext45"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

This ZIP contains no existing `.git` history or remote configuration. The
`.gitignore` includes exceptions so native RDS fixtures are committed.

## Validation and provenance

See `optimization/VALIDATION.md`, `optimization/provenance.json` and the
included summary tables for the completed evaluation and its limitations.
`optimization/current_best.patch` is the cumulative patch against the pinned
upstream commit; it is **already applied** to this source tree.
`optimization/source_optimized.json` records the exact evaluated source tree.
Only README and packaging-ignore files differ from that evaluated tree.
`optimization/repository_SHA256SUMS.json` covers the delivered repository files,
excluding that checksum inventory itself.

Original attribution and licensing are retained in `DESCRIPTION`, `LICENSE`,
`LICENSE.md` and `inst/CITATION`.
