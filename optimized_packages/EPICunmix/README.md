# EPICunmix — optimized source

This repository contains the current best optimized EPICunmix implementation
from the completed optimization and subsequent real-data evaluation.
The package name remains **EPICunmix** and the upstream version remains **0.0.1**.
This is an optimized source snapshot, not an upstream release or an official fork.

All files in `R/`, `man/`, and `data/`, as well as `DESCRIPTION`, `NAMESPACE`, and
`LICENSE`, are byte-identical to the frozen optimized version used in the evaluation.
The optimization patch against upstream commit
`b512206e3e0bcb9f0720423293094935861fdcde` is in `docs/optimized.patch`.

## Install from this checkout

Unzip the download and work inside the `EPICunmix` folder.
Install R and the package dependencies, then install the local source:

```r
install.packages(c(
  "remotes", "BiocManager", "nnls", "doParallel", "foreach",
  "MCMCglmm", "Matrix", "matrixcalc", "BisqueRNA"
))
BiocManager::install(c("edgeR", "Biobase"), ask = FALSE, update = FALSE)
remotes::install_local(".", dependencies = FALSE, upgrade = "never")
library(EPICunmix)
```

Alternatively, with dependencies already available:

```sh
R CMD INSTALL .
```

After uploading this folder's contents to your own GitHub repository, you can
install that repository with `remotes::install_github("YOUR_ACCOUNT/YOUR_REPOSITORY")`.
Do not use the historical upstream installation command if you want this optimization.

## Usage and documentation

The public API is unchanged:

```r
library(EPICunmix)
help("get_prior", package = "EPICunmix")
help("bMIND", package = "EPICunmix")
help("bmind_de", package = "EPICunmix")
help("run_epic_unmix", package = "EPICunmix")
data("example", package = "EPICunmix")
str(example)
```

Supply your own bulk expression, single-cell reference, metadata, and cell-type
fractions for analyses. The small upstream package datasets are retained for
compatibility; the large external `example/` directory is not included.
Examples in the inherited function documentation that reference that directory
require separately obtaining its inputs from upstream.
Historical method instructions and attribution are in `docs/UPSTREAM_README.md`.
As upstream advises, avoid loading the standalone bMIND package and EPICunmix
at the same time.

## Upload to GitHub

Create an empty repository and upload the contents of this folder, or run:

```sh
git init
git add .
git commit -m "Add evaluated optimized EPICunmix source"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

No Git history, credentials, benchmark data, fitted outputs, installed libraries,
or machine-specific paths are bundled. The original automatic website-deployment
workflow is omitted, so uploading does not attempt to deploy a site.

## Validation and limitations

The completed evaluation compared this exact code with the original over three
real bulk-expression datasets and three paired repetitions per dataset.
All 68,891,028 finite numeric pairs were exactly equal. This is equivalence
evidence for the tested inputs, not a guarantee for all possible inputs.
The separate evaluation archives contain the complete results and reproduction records.

`SOURCE_MANIFEST.json` identifies retained upstream files and their hashes.
`SHA256SUMS` inventories this repository snapshot, excluding itself.
`RELEASE_NOTES.md` describes packaging choices and inherited package-check limitations.
No scientific benchmark is rerun merely to create this source archive.

## License and attribution

Original author information and the upstream GPL license are retained unchanged
in `DESCRIPTION` and `LICENSE`. EPICunmix builds upon bMIND; see the historical
README for attribution and upstream links.
