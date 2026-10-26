# Optimized bioinformatics packages

This directory contains 30 source snapshots imported from the supplied
[`30_githubs.zip` Google Drive archive](https://drive.google.com/file/d/1U7ay5vUm3L99w_IXwZNXGLNnqB0rMOuv/view).
They cover single-cell and spatial analysis, statistical genetics, sequence
processing, and protein structure prediction and design. Descriptions below
summarize the included project documentation.

## Single-cell and spatial analysis

| Package | What it contains |
| --- | --- |
| [BayesSpace](BayesSpace-optimized/) | R/C++ tools for Bayesian clustering and resolution enhancement of spatial transcriptomics. Includes package source, vignettes, fixtures, and optimization provenance. |
| [CellChat](CellChat-optimized/) | R tools for inferring and comparing cell-cell communication from single-cell and spatial expression data, with an optimized permutation-testing implementation. |
| [EPICunmix](EPICunmix/) | R implementation for deconvolving bulk expression and estimating cell-type-specific expression using single-cell references; includes bMIND-related functions and package examples. |
| [MAGIC](MAGIC_optimized/) | Gene-expression denoising and imputation through graph diffusion. Contains both the Python MAGIC package and its optimized graphtools dependency, including optional native code. |
| [SCENT](SCENT/) | R tools linking accessible chromatin peaks to target genes using paired single-cell ATAC and RNA measurements, Poisson regression, and bootstrap inference. |
| [SpatialDE](SpatialDE/) | Python implementation for detecting spatially variable genes using Gaussian-process models, with analysis scripts, notebooks, and numerical smoke checks. |
| [cell2location-fast](cell2location-fast/) | GPU-oriented cell2location fork for mapping cell-type abundance in spatial transcriptomics, including a direct PyTorch spatial model, benchmarks, and validation documentation. |
| [pySCENIC](pyscenic-optimized/) | A two-package source tree containing pySCENIC and ctxcore for gene regulatory network inference, regulon analysis, and cell-level activity scoring. |
| [scVelo](scvelo-optimized/) | Python toolkit for RNA velocity, splicing dynamics, latent time, and transcriptional kinetics, with source, documentation, and tests. |
| [Tangram](tangram_optimized/) | Python tools for mapping single-cell expression profiles to spatial measurements and projecting genes or annotations; includes CPU/native and GPU-related optimizations. |

## Statistical genetics and regulatory genomics

| Package | What it contains |
| --- | --- |
| [CIGMA](CIGMA-optimized/) | Python tools for decomposing cell-type-shared and cell-type-specific genetic effects on expression, with Snakemake analysis workflows and R support for REML. |
| [ChromBPNet](chrombpnet-optimized/) | Sequence-based, bias-aware neural models of chromatin accessibility; includes training, inference, interpretation, and evaluation utilities. |
| [GPN-Star](gpn_star_optimized/) | Genomic language-model source and recipes covering GPN, GPN-MSA, PhyloGPN, and GPN-Star, including inference and training support. |
| [LAVA](LAVA-optimized/) | R tools for local genetic heritability and genetic correlation analysis, with tutorial data and packaging notes. |
| [LDSC](ldsc-optimized/) | Optimized Python 3 port of LD Score Regression, including LD-score computation, heritability regression, and summary-statistics preparation. |
| [SAIGE](SAIGE-optimized/) | R/C++ mixed-model association analysis for large cohorts, including single-variant and SAIGE-GENE+ rare-variant set tests. |
| [bigsnpr](bigsnpr-optimized/) | R/C++ tools for large SNP datasets, genotype processing, and statistical genetics, with documentation and vendored components. |
| [susieR](susieR/) | R implementation of Sum of Single Effects sparse regression and genetic fine-mapping, including credible-set inference, examples, and tests. |

## Sequence processing and alignment

| Package | What it contains |
| --- | --- |
| [DIAMOND](diamond-optimized/) | Optimized C++ protein and translated-DNA sequence aligner, with build files, native tests, and optimization provenance. |
| [fastp](fastp-optimized/) | C++ FASTQ preprocessing, read filtering, adapter trimming, quality control, and reporting. |
| [minimap2](minimap2/) | C sequence mapper and aligner for nucleotide reads and assemblies, including command-line source, Python bindings, and helper utilities. |

## Protein prediction and design

| Package | What it contains |
| --- | --- |
| [AlphaFold 2](alphafold-round5-af5-014/) | AlphaFold v2 and AlphaFold-Multimer inference source, supporting scripts, examples, and documentation. The archive's round-five folder name is retained. |
| [AlphaFold 3](alphafold3/) | AlphaFold 3 structure-prediction source snapshot with supporting scripts and verification documentation. |
| [Boltz](boltz-optimized/) | Optimized biomolecular structure-prediction source, CLI, examples, and provenance. |
| [BoltzGen](boltzgen_optimized/) | Biomolecular binder-design and generation toolkit with optimized inference source, examples, and environment requirements. |
| [LigandMPNN](ligandmpnn_optimized/) | Ligand-aware protein sequence design from structural inputs, with inference scripts, helpers, training source, and a model-parameter downloader. |
| [ProteinMPNN](proteinmpnn_optimized/) | Structure-conditioned protein sequence design, with optimized inference, helper scripts, examples, training source, and bundled inference weights. |
| [RFdiffusion3 / Foundry](rfd3-optimized/) | Optimized Foundry source containing RFdiffusion3 and related model components, documentation, and test fixtures. |
| [ThermoMPNN](ThermoMPNN-optimized/) | Protein mutation stability prediction using ProteinMPNN-derived representations, with model source and supporting resources. |
| [xfold](xfold-optimized/) | Optimized xfold adapted to OpenFold3-preview2 weights, including af3_torch, shared kernels, build recipes, and provenance. |

## Working with these snapshots

- Each folder is an independent project. Follow its local README, dependency
  files, and optimization or validation notes; there is no shared installation
  command or environment for all 30 packages.
- Install the local source to use the supplied modifications. Installing a
  similarly named package from PyPI, CRAN, Bioconductor, or upstream GitHub can
  select a different implementation.
- Python/R versions, native libraries, GPU requirements, and external assets
  differ by project. Some model weights, reference databases, and evaluation
  datasets must be obtained separately under their applicable terms.
- Existing licenses, attribution, and citation instructions remain in each
  package. MAGIC's README records an unresolved GPL version discrepancy.
- Included speedup and validation reports describe prior work. This import
  verifies archive/file integrity; it does not rerun scientific benchmarks,
  install dependencies, or establish performance or numerical equivalence.

## Import details

Original top-level package names and file bytes are retained. The minimap2 ZIP
had no enclosing project folder, so its files are placed in `minimap2/`; its
embedded `.git/` metadata is omitted so the sources can be tracked by the parent
repository. Executable permissions and the five relative documentation symlinks
in RFdiffusion3 are preserved. The ZIP files themselves are not duplicated here.

[`IMPORT_MANIFEST.json`](IMPORT_MANIFEST.json) records the source archive,
SHA-256 checksums of the outer and individual ZIPs, destination folders, imported
file counts, and omitted Git metadata entry counts.
