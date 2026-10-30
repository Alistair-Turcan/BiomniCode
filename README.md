# BiomniCode

BiomniCode brings together a reusable **Biomni agent skill for code optimization**
and **30 optimized bioinformatics source packages**. The skill captures the method
used to produce these kinds of optimized packages; the package collection contains
implementations for single-cell and spatial analysis, statistical genetics,
sequence processing, and protein prediction and design.

## Repository layout

```text
BiomniCode/
├── README.md                       # Repository overview and full package catalog
├── skills/
│   ├── IMPORT_MANIFEST.json         # Uploaded skill archive provenance
│   └── biomni-code/
│       ├── SKILL.md                 # Agent entry point and optimization workflow
│       ├── README.md, VERSION       # Supplied package overview and version marker
│       ├── scripts/                 # Benchmark, state, recovery, delivery helpers
│       ├── references/              # Detailed method, languages, strategy library
│       ├── templates/               # Reports, notebooks, records, reproduction
│       ├── tests/                   # Synthetic helper regression tests
│       ├── manifest.json            # Current package file checksums
│       └── <language>/, strategy_library/  # Supplied root resource aliases
└── optimized_packages/
    ├── README.md                    # Package-specific catalog and import notes
    ├── IMPORT_MANIFEST.json         # Original package archive checksums
    └── <30 project directories>/    # Source, local docs, licenses, tests/resources
```

## The biomni-code agent skill

Start with [`skills/biomni-code/SKILL.md`](skills/biomni-code/SKILL.md).
The skill directs an agent to reduce runtime and memory use while preserving
public APIs, file formats, behavior, and scientific meaning. It includes an
executable ledger and evidence gates rather than relying on conversational memory.

Its workflow has seven phases:

| Phase | Purpose |
| --- | --- |
| 0 — Setup | Prepare a reproducible environment, representative workload, protected original source, and run state. |
| 1 — Baseline | Measure original runtime, peak memory, output identity, and repeatability. |
| 2 — Profiling | Diagnose CPU, I/O, and memory costs and rank measured hotspots. |
| 3 — Scaling suite | Define deterministic input sizes and calibrate the per-iteration benchmark. |
| 4 — Correctness suite | Establish the behavior checks that gate changes. |
| 5 — Optimization | Reassess, implement, test, measure, review, and accept or revert one candidate at a time. |
| 6 — Delivery | Run final correctness and scaling comparisons, produce figures and reports, and verify delivery artifacts. |

The default budget is **50 substantive attempts**, with a user-requested override.
Accepted and rejected candidates count; setup and recovery do not. The method
requires a justified numerical tolerance policy frozen before optimization,
protected tests and reference outputs, measured runtime/memory acceptance,
a diverse idea backlog, feedback-driven selection, and separately verified
checkpoints. Its stop gate distinguishes completed work from interrupted or blocked
runs. These are instructions for a future optimization task, not evidence that
this repository import performed those experiments.

### Skill resources

| Resource | Contents |
| --- | --- |
| [`scripts/`](skills/biomni-code/scripts/) | State and history (`biomni_code_state.py`), measurement (`benchmark_runner.py`), acceptance (`acceptance.py`), numerical policy, scaling, plotting, resume, run control, checkpoint, integrity, packaging, and delivery helpers. |
| [`references/`](skills/biomni-code/references/) | Optimization loop, correctness, numerical tolerance, calibration, state, recovery, parallelism, profiler tools, delivery requirements, and helper latency guidance. |
| [`references/languages/`](skills/biomni-code/references/languages/) | Hints for Python, R, C, C++, Rust, Go, Java, Julia, JavaScript, and assembly. Root language folders preserve the supplied aliases. |
| [`references/strategy_library/`](skills/biomni-code/references/strategy_library/) | Eleven optimization families, intervention weights, and hotspot taxonomy. A root strategy-library copy is also supplied. |
| [`templates/`](skills/biomni-code/templates/) | Benchmark cards, candidate and feedback records, iteration records, optimization notebook, status, final report, run manifest, and reproduction script. |
| [`tests/`](skills/biomni-code/tests/) | Synthetic regressions for source identity, candidate admission, portfolio bookkeeping, promotion, recovery, and checkpoint behavior. |

### Using the skill in Biomni

Make the complete `skills/biomni-code/` directory available to the Biomni agent's
skill-loading mechanism and select `SKILL.md` as its entry point. Keep `scripts/`,
`references/`, and `templates/` alongside it so relative resources resolve.
This repository supplies the skill files; it does not include the Biomni runtime
or an automatic skill-registration adapter.

Give the agent a target repository and revision, a representative command and
input workload, correctness requirements, available hardware, and any desired
iteration-budget override. For example:

> Use the biomni-code skill to optimize this repository's inference workload.
> Measure the supplied representative inputs, preserve the public interface and
> scientific outputs under a justified numerical policy, and deliver the optimized
> source with reproducible correctness and performance evidence.

The supplied workflow uses `/workspace/biomnicode/<repo>/` for run state,
`/mnt/shared-workspace/biomnicode/<repo>/checkpoints` for checkpoint storage, and
`/mnt/results/` for final artifacts. Provision suitable writable storage when
running it. Dependencies, compilers, profilers, model assets, and GPU requirements
come from the selected target; the skill is not a shared environment for all packages.
See the [skill overview](skills/biomni-code/README.md) and
[recovery guide](skills/biomni-code/references/interruption_recovery.md) before a run.

### Validating the skill package

From the repository root:

```bash
cd skills/biomni-code
python scripts/verify_skill_package.py --skill-root . --compile
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

These validate package structure, Python compilation, and helper regressions.
They do not benchmark the 30 scientific packages or establish their numerical
correctness. The supplied archive has mixed version metadata: root `VERSION` is
`2.5.3`, while `references/VERSION` and the updated skill describe helper version
`2.5.4` with run-policy compatibility `2.5.3`. Both markers are preserved.

## Optimized package catalog

Each linked folder is an independent source project. Its local README, dependency
files, licensing, and validation notes are authoritative for that snapshot.

### Single-cell and spatial analysis

| Package | What it contains |
| --- | --- |
| [BayesSpace](optimized_packages/BayesSpace-optimized/) | R/C++ tools for Bayesian clustering and resolution enhancement of spatial transcriptomics. Includes package source, vignettes, fixtures, and optimization provenance. |
| [CellChat](optimized_packages/CellChat-optimized/) | R tools for inferring and comparing cell-cell communication from single-cell and spatial expression data, with an optimized permutation-testing implementation. |
| [EPICunmix](optimized_packages/EPICunmix/) | R implementation for deconvolving bulk expression and estimating cell-type-specific expression using single-cell references; includes bMIND-related functions and package examples. |
| [MAGIC](optimized_packages/MAGIC_optimized/) | Gene-expression denoising and imputation through graph diffusion. Contains both the Python MAGIC package and its optimized graphtools dependency, including optional native code. |
| [SCENT](optimized_packages/SCENT/) | R tools linking accessible chromatin peaks to target genes using paired single-cell ATAC and RNA measurements, Poisson regression, and bootstrap inference. |
| [SpatialDE](optimized_packages/SpatialDE/) | Python implementation for detecting spatially variable genes using Gaussian-process models, with analysis scripts, notebooks, and numerical smoke checks. |
| [cell2location-fast](optimized_packages/cell2location-fast/) | GPU-oriented cell2location fork for mapping cell-type abundance in spatial transcriptomics, including a direct PyTorch spatial model, benchmarks, and validation documentation. |
| [pySCENIC](optimized_packages/pyscenic-optimized/) | A two-package source tree containing pySCENIC and ctxcore for gene regulatory network inference, regulon analysis, and cell-level activity scoring. |
| [scVelo](optimized_packages/scvelo-optimized/) | Python toolkit for RNA velocity, splicing dynamics, latent time, and transcriptional kinetics, with source, documentation, and tests. |
| [Tangram](optimized_packages/tangram_optimized/) | Python tools for mapping single-cell expression profiles to spatial measurements and projecting genes or annotations; includes CPU/native and GPU-related optimizations. |

### Statistical genetics and regulatory genomics

| Package | What it contains |
| --- | --- |
| [CIGMA](optimized_packages/CIGMA-optimized/) | Python tools for decomposing cell-type-shared and cell-type-specific genetic effects on expression, with Snakemake analysis workflows and R support for REML. |
| [ChromBPNet](optimized_packages/chrombpnet-optimized/) | Sequence-based, bias-aware neural models of chromatin accessibility; includes training, inference, interpretation, and evaluation utilities. |
| [GPN-Star](optimized_packages/gpn_star_optimized/) | Genomic language-model source and recipes covering GPN, GPN-MSA, PhyloGPN, and GPN-Star, including inference and training support. |
| [LAVA](optimized_packages/LAVA-optimized/) | R tools for local genetic heritability and genetic correlation analysis, with tutorial data and packaging notes. |
| [LDSC](optimized_packages/ldsc-optimized/) | Optimized Python 3 port of LD Score Regression, including LD-score computation, heritability regression, and summary-statistics preparation. |
| [SAIGE](optimized_packages/SAIGE-optimized/) | R/C++ mixed-model association analysis for large cohorts, including single-variant and SAIGE-GENE+ rare-variant set tests. |
| [bigsnpr](optimized_packages/bigsnpr-optimized/) | R/C++ tools for large SNP datasets, genotype processing, and statistical genetics, with documentation and vendored components. |
| [susieR](optimized_packages/susieR/) | R implementation of Sum of Single Effects sparse regression and genetic fine-mapping, including credible-set inference, examples, and tests. |

### Sequence processing and alignment

| Package | What it contains |
| --- | --- |
| [DIAMOND](optimized_packages/diamond-optimized/) | Optimized C++ protein and translated-DNA sequence aligner, with build files, native tests, and optimization provenance. |
| [fastp](optimized_packages/fastp-optimized/) | C++ FASTQ preprocessing, read filtering, adapter trimming, quality control, and reporting. |
| [minimap2](optimized_packages/minimap2/) | C sequence mapper and aligner for nucleotide reads and assemblies, including command-line source, Python bindings, and helper utilities. |

### Protein prediction and design

| Package | What it contains |
| --- | --- |
| [AlphaFold 2](optimized_packages/alphafold-round5-af5-014/) | AlphaFold v2 and AlphaFold-Multimer inference source, supporting scripts, examples, and documentation. The archive's round-five folder name is retained. |
| [AlphaFold 3](optimized_packages/alphafold3/) | AlphaFold 3 structure-prediction source snapshot with supporting scripts and verification documentation. |
| [Boltz](optimized_packages/boltz-optimized/) | Optimized biomolecular structure-prediction source, CLI, examples, and provenance. |
| [BoltzGen](optimized_packages/boltzgen_optimized/) | Biomolecular binder-design and generation toolkit with optimized inference source, examples, and environment requirements. |
| [LigandMPNN](optimized_packages/ligandmpnn_optimized/) | Ligand-aware protein sequence design from structural inputs, with inference scripts, helpers, training source, and a model-parameter downloader. |
| [ProteinMPNN](optimized_packages/proteinmpnn_optimized/) | Structure-conditioned protein sequence design, with optimized inference, helper scripts, examples, training source, and bundled inference weights. |
| [RFdiffusion3 / Foundry](optimized_packages/rfd3-optimized/) | Optimized Foundry source containing RFdiffusion3 and related model components, documentation, and test fixtures. |
| [ThermoMPNN](optimized_packages/ThermoMPNN-optimized/) | Protein mutation stability prediction using ProteinMPNN-derived representations, with model source and supporting resources. |
| [xfold](optimized_packages/xfold-optimized/) | Optimized xfold adapted to OpenFold3-preview2 weights, including af3_torch, shared kernels, build recipes, and provenance. |

## Installing and using the scientific packages

Choose the package for your task and follow its local installation instructions.
There is no single install command, shared dependency lock, or common runtime
for this collection. Several snapshots require different Python or R versions,
native compilers, system libraries, or compatible CUDA environments. Install the
local source when you want the included optimizations: the similarly named
upstream PyPI, CRAN, Bioconductor, or GitHub project may contain different code.

Source, tests, examples, weights, and datasets vary by package. Some resources are
bundled; others require separate downloads and access agreements. In particular,
large model weights, reference databases, and evaluation inputs must not be
assumed present. Included benchmark results describe historical evaluations, not
a new comparison of all packages on one machine. Consult each project's recorded
scope before applying speedup or output-equivalence claims to a new workload.

## Provenance and repository validation

The 30 packages came from the user-supplied `30_githubs.zip` archive. Their
[import manifest](optimized_packages/IMPORT_MANIFEST.json) records individual
archive hashes, folders, file counts, and omitted Git metadata. The import
verified 8,506 files and preserved five internal documentation symlinks.
The minimap2 archive was placed inside `minimap2/` and its embedded `.git/`
metadata omitted. See the [package import notes](optimized_packages/README.md#import-details).

The skill came from the user-supplied `biomni-code.zip` archive, imported under
`skills/biomni-code/`. Its [import record](skills/IMPORT_MANIFEST.json) records
the archive SHA-256 and original stale checksum entries. The two package checksum
manifests were regenerated to match the delivered files, including the newer
helper-latency reference; all other uploaded skill files retain their original
bytes. The supplied root and reference READMEs differ in their packaging and
version descriptions; both are retained as source documentation.

## Licenses and attribution

The scientific packages retain their individual licenses, notices, and citation
requirements; there is no repository-wide license replacing them. MAGIC's local
README documents an unresolved GPL version discrepancy. Model weights and
external datasets can have separate terms. The supplied skill archive contains
no standalone license file; this repository does not assign it a new license.
