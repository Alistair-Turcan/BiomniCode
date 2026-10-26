# Finalized optimization and evaluation

## Source identity and accepted changes

- Upstream: ChromBPNet v1.0.1, commit `eaa0fe58b6a43da62ea23b75cfc2bef4ecd3550c`.
- Frozen optimized source ZIP SHA-256: `ffe98410ab9312aa91c182f0c18057a1806e5932c0ea0df5c2234394c0ec916b`.
- Dense fixedStep NumPy bigWig writes replace per-base Python lists for nonempty slices, preserving interval ownership and the original empty/tied-slice fallback.
- Reuse no-bias predictions only when inputs, supported model configuration and weights match exactly; arbitrary models retain the original inference path.
- Eligible float32 GPU dilated Conv1D layers use a height-one NHWC Conv2D operation; unsupported cases retain the original operation.

No new optimization, retraining, weight changes, or precision changes were made for this release. Changes apply to inference/export, not a validated training-speed improvement.

## Completed independent large-input evaluation

Three technical repetitions per original/optimized version on three real human ATAC-seq workloads: 18 successful fresh-process runs. All released model-filtered eligible peaks were used; no region cap. Runtime includes imports, models, input reads/encoding, inference, normal outputs, observed-track metrics and their normal plots. Comparison and evaluation-plot generation are excluded from timing.

| Dataset | Regions | Speedup (ratio of mean runtimes) | Host RSS change | TF allocator peak change |
|---|---:|---:|---:|---:|
| GM12878 | 275,542 | 1.806× | +6.33% | -10.87% |
| HepG2 | 215,398 | 1.817× | +6.13% | +6.27% |
| IMR-90 | 262,876 | 1.823× | +6.39% | +13.23% |

**Memory trade-off:** peak host RSS increased on all three large workloads. TensorFlow allocator behavior was dataset-dependent; sampled driver device peak was 9,333 MiB for both versions in all runs. RSS, allocator memory and sampled device memory are distinct domains.

Exhaustive emitted HDF5 and decoded bigWig comparisons found maximum absolute and relative errors of zero, with no numerical violations across 243 diagnostic rows. Protected coordinates/order/schema/storage, decoded PNG pixels and final RNG states matched. Decoded outputs and RNG states matched within-version repetitions. Raw logits are not emitted by this CLI and were not directly compared.

Hardware/controls: NVIDIA A10 (23,028 MiB), driver 610.57.04, batch size 64, float32, seeds 0, deterministic TensorFlow/cuDNN, TF32 disabled, memory growth, TF/OMP intra-op 4, inter-op 1, BLAS 1. Runs were serial, fresh-process and warm-filesystem. Host OS process thread count was 36 in every full run. Exact stack versions are summarized in README.

These are descriptive, device/stack/workload-specific technical measurements, not biological replicates, a cross-hardware guarantee, or held-out biological accuracy. Workloads include training regions. Do not pool these timings with the earlier small benchmark, which included comparison/canonicalization overhead.

## Packaging-only changes

Relative to the frozen source, only three original files differ:
1. README.md: optimized-version introduction prepended; original body retained.
2. setup.py: `license_files` points to the existing `LICENSE`, not `LICENSE.txt`.
3. .github/workflows/latest-docker-image.yml: upstream automatic DockerHub publishing replaced with manual build-only automation, without secrets/login/push.

All other original files, especially every scientific implementation file under `chrombpnet/`, are byte-identical. Upstream MIT license and CITATION.cff are unchanged. No GitHub account/repository URL or new package/version identity was invented.

The tested Python lock is an environment record, not evidence of a fresh-machine rebuild. Release checks parse/compile Python and build sdist/wheel without installing dependencies or rerunning GPU inference. Full evaluation bundles and raw inputs/outputs are separate deliverables.
