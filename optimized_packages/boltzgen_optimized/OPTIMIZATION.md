# Optimized-version provenance

## Exact retained implementation

- Upstream repository: https://github.com/HannesStark/boltzgen
- Upstream commit: `31d9d9b9c72245b4ed6fe8742d6fbf4e1a3552a0`.
- Upstream package version: `0.3.2` (unchanged; this is not a new upstream release).
- Accepted optimized candidate: **retained iter021**.
- Delivered cumulative patch SHA-256: `ccda3e001951cdfe16a13657367ca6803e7b1f1b22c0a4b86a3c42c4d13f7643`.
- Accepted source archive SHA-256: `76d59d819a75b3d95dc230f0a3cedb7ce240d3a7047eea0a048376869075c358`.

Every implementation file in this package is byte-identical to the accepted
optimized source archive. Only the README was prefixed with fork-specific
installation guidance; this note, the environment lock and the provenance
manifests are additions. No later experimental candidate is included.

`provenance/accepted_source_manifest.json` records the accepted archive's file
identities. `provenance/package_sha256.json` records the files in this GitHub-ready
distribution (excluding that inventory itself).

## Independent evaluation summary

The independently completed evaluation used three real targets (2A1X, 1JQD and
2PNY), three paired process replicates per version, 30 designs per process,
140-residue binders, 500 diffusion steps and three recycling steps on one NVIDIA
A10. Whole-process timing included imports, initialization/checkpoint loading,
inference, complete output writes and identical final-coordinate capture.

| Target | Original seconds, mean ± sample SD | Optimized seconds, mean ± sample SD | Speedup |
| --- | ---: | ---: | ---: |
| 2A1X | 1051.3 ± 15.6 | 805.0 ± 5.3 | 1.31× |
| 1JQD | 1022.4 ± 13.8 | 790.2 ± 9.1 | 1.29× |
| 2PNY | 807.1 ± 10.5 | 636.5 ± 3.5 | 1.27× |

All 270 paired designs were bitwise exact in captured coordinates, metadata and
RNG checks; all 540 paired application-file comparisons were exact. Across
3,395,520 coordinate components, MAE/RMSE/max absolute error were zero.
GPU reservation increased approximately 32.5–33.4%; host peak RSS was approximately
6.5–6.7 GiB. These results apply to the tested environment and workload, not all
hardware or target classes. Biological binding quality was not evaluated.

The separate `boltzgen_evaluation.zip` contains the full evidence;
`boltzgen_optimized_lite.zip` contains its data-free evaluation companion.
Neither evaluation archive is embedded in this source-only repository.

## Installation and runtime notes

- Tested Python: 3.11.13; Torch: 2.7.1+cu126; Torch CUDA: 12.6;
  NumPy: 2.0.2. The exact evaluated dependency versions are preserved in
  `requirements-evaluation.lock`; that lock is a tested environment, not a
  cross-platform dependency promise.
- Install Torch first from its CUDA 12.6 index, then install the lock and the
  local package as shown in the README. Install a C++ compiler and `ninja`
  separately for the optional native initializer accelerator.
- Use a local editable install (`-e .`) to retain access to its adjacent C++
  source. Building/publishing binary wheels has not been validated by this
  packaging task.
- The native initializer loader is guarded to the validated Torch version/ABI.
  Unsupported or failed native builds fall back to eager behavior; the measured
  speedup must not be assumed when the accelerator is unavailable.
- Production entropy behavior remains unchanged. Common-random-number controls
  used for the evaluation were harness-only, not production changes.
- Checkpoints, molecule libraries and pretrained model downloads follow the
  upstream usage documentation; no weights or generated designs are included.

## Uploading to GitHub

Extract the ZIP and use the contents of `boltzgen_optimized/` as your repository
root. It already contains the source layout, `pyproject.toml`, `.gitignore`,
tests, upstream example inputs, assets, Dockerfile and license. No `.git` history,
environment, compiled cache, benchmark bundle or generated design output is included.

The upstream `.github/workflows/python-publish.yml` is preserved unchanged.
Review or disable it before creating releases, and configure your own package
identity and publishing credentials before attempting a PyPI release. This ZIP
does not publish anything or claim ownership of the upstream package name.

Upstream attribution and license are retained in `LICENSE` and the original
documentation below the README banner.
