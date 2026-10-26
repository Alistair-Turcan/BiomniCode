# Optimized LAVA source snapshot

This is the optimized source tree used in the latest three-configuration,
three-replicate evaluation. It includes all changes present in that evaluated
tree, including changes that did not individually pass the speedup gate.
It is not an upstream release or a claim of validation on arbitrary inputs.

## Install from this directory

Install R, a C++ compiler, and the required R dependencies first:

```r
install.packages(c("cpp11", "data.table", "matrixsampling"))
```

Then, in a terminal from the extracted repository directory:

```sh
R CMD INSTALL .
```

The native code will be compiled locally. The evaluated environment was Linux
x86-64 with R 4.4.2. Some optimized kernels use AVX2/FMA-targeted code;
other operating systems, CPU architectures, and BLAS implementations have
not been validated for this snapshot. Do not assume cross-platform bitwise
equivalence.

## Upload to a GitHub repository

Unzip the archive and upload the contents of `LAVA-optimized/` as the repository
root, including hidden files. No Git history or remote is embedded.
After uploading to a repository you control, install that fork with:

```r
remotes::install_github("YOUR_ACCOUNT/YOUR_REPOSITORY", build_vignettes = FALSE)
```

The original README follows the snapshot notice and contains the LAVA tutorial.
Its upstream installation example installs upstream LAVA, not this snapshot.

## Included and excluded

R/C++ sources, package metadata, manual pages, upstream documentation,
support data, and the original tutorial datasets are included. Tutorial data
are retained so the documented examples remain usable.

Compiled objects/libraries, benchmark outputs, evaluation archives, local
environments, and temporary files are excluded. SOURCE_MANIFEST.json records
SHA-256 hashes of the packaged files, excluding the manifest itself.

## Attribution and licensing

Based on josefin-werme/LAVA, commit
e729a245f7b6923967a96804fbf5246eadf2d6c6.
Original authorship and package metadata are preserved.

The upstream DESCRIPTION declares `License: All rights reserved`.
No new license or permission to publish or redistribute is granted by this
archive. Confirm the applicable permissions before publishing a public fork.
