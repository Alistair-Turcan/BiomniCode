# Optimized LDSC — Python 3

Python 3 port of the optimized LDSC fork based on
[bulik/ldsc](https://github.com/bulik/ldsc), version 1.0.1.
This is the optimized Python 3 source evaluated against both the Python 2
original and the CBIIT Python 3 fork in the accompanying project evaluation.

## Installation

Use Python 3.9, the evaluated interpreter version, and the pinned dependencies:

```bash
conda env create -f environment.yml
conda activate ldsc-opt-py3
python ldsc.py --help
python munge_sumstats.py --help
```

Alternatively, with an existing Python 3.9 installation:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python ldsc.py --help
```

The evaluated stack is NumPy 1.26.4, pandas 1.5.3, SciPy 1.10.1,
and bitarray 2.6.0. `make_annot.py` additionally requires pybedtools
and a compatible BEDTools installation; these are not needed for `--l2`
or `--h2`.

## Usage

Estimate LD scores from PLINK inputs:

```bash
python ldsc.py --l2 --bfile /path/to/plink_prefix --ld-wind-cm 1 --out ld_scores
```

Run heritability regression using chromosome-split reference scores:

```bash
python ldsc.py --h2 trait.sumstats.gz \
  --ref-ld-chr /path/to/baselineLD. \
  --w-ld-chr /path/to/weights.hm3_noMHC. \
  --out trait_h2
```

Reference panels and GWAS datasets are not included.
See the [upstream tutorials](https://github.com/bulik/ldsc/wiki)
for input formats, methodological guidance, and additional modes.

## Caching and numerical agreement

The optimized implementation caches parsed and merged intermediate inputs in
`.ldsc_cache` directories beside input files. A first run builds these caches;
later runs with matching inputs can reuse them. Final regression results are
computed again, not cached. Cache directories can be deleted when no runs are
using them; they will be rebuilt when needed. Allow additional disk space and
write permission beside inputs.

In the three-dataset project evaluation, the Python 3 variants agreed with the
original's full-precision regression outputs within approximately 2.5e-15
absolute difference, but were **not bitwise-identical** across the evaluated
software stacks. Chromosome 22 LD-score output was identical after
decompression. Do not assume bitwise identity across BLAS or dependency
versions. Timing depends on data, cache state, BLAS, and thread settings.

## Tests

```bash
python -m nose test
```

Tests and their original bundled fixtures are retained. Packaging validation
does not constitute a fresh execution of the test suite.

## Publishing to GitHub

Extract this archive, enter the `ldsc-optimized` directory, and upload its
contents or initialize your own repository:

```bash
git init
git add .
git commit -m "Add optimized Python 3 LDSC"
```

The archive contains no Git history, runtime caches, external GWAS/reference
panels, local environments, or evaluation-run outputs.
`PACKAGING.md` identifies the evaluated source archive and packaging changes.
`README.upstream.md` preserves the original project documentation and citations;
`PORTING.md` describes the Python 3 conversion.

## License and attribution

GNU GPL v3; see `LICENSE`. Original LDSC authors: Brendan Bulik-Sullivan
and Hilary Finucane. Preserve upstream attribution and cite the relevant LDSC
methods papers listed in `README.upstream.md` when using the software.
