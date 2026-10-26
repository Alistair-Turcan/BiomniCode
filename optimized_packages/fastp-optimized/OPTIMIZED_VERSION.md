# Optimized fastp: current best delivered version

This repository contains the final optimized fastp implementation evaluated in the completed
optimization and subsequent three-full-dataset evaluation. This is a source release, not an
upstream official release. The MIT license and upstream attribution are preserved in `LICENSE`.
See `README.md` for the complete upstream usage documentation.

## Source identity

- Base: fastp 1.4.0, upstream commit `ee125a52fddae9d8189e13cd8a33ad04988a1351`.
- Cumulative optimization patch SHA-256:
  `61d4281f997c623dcb41e4d86f940c9c9ed9636c968d4c21441e1276cdf3caf9`.
- The optimized C++ sources, Makefile, helper scripts and upstream test assets are unchanged from
  the verified delivered version. The optimization is already applied; do not apply the patch again.
- Packaging-only changes: `.gitignore` additionally excludes local environments/build/data
  directories; the inherited GitHub Actions workflow accepts both `main` and `master`.
  This note, build recipes, patch and checksum manifests are additions, not algorithm changes.
- `BENCHMARKED_SOURCE.sha256` identifies all 75 files of the benchmarked optimized tree.
  The README and all scientific implementation files retain their original bytes.
- `PACKAGE_MANIFEST.sha256` identifies the files in this source release, excluding itself.

## Build with the benchmarked dependency versions

The preserved explicit Conda lock is Linux x86_64 specific. It pins ISA-L 2.32.1,
libdeflate 1.26 and Highway 1.3.0, together with supporting runtime/tool packages.
GNU make, Conda and a C++ compiler must already be installed. The evaluated compiler was GCC 13.3.0;
the lock does not install or pin the host compiler. A different host/compiler is a new build.

From this repository's root:

```bash
bash recipes/environment.sh
bash recipes/build.sh "$PWD"
./env/build/fastp-optimized/fastp --version
./env/build/fastp-optimized/fastp test
```

The final two paths assume the extracted directory is named `fastp-optimized`.
If renamed, `build.sh` prints the actual binary path; its final component uses the source directory name.
The recipe uses release `-O3` flags, four make jobs, and an rpath to the local dependency environment.
Move or rebuild the executable together with its libraries; it is not a standalone static distribution.
For normal system-wide dependency installation and a direct `make`, see the upstream README.

## GitHub

Unzip the archive, enter `fastp-optimized`, then initialize a new repository:

```bash
git init -b main
git add .
git commit -m "Import optimized fastp source"
# Create an empty repository in your own GitHub account, then:
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

No `.git` history, remote configuration, credentials, large sequencing datasets, evaluation reports,
raw outputs, local dependencies, or compiled binaries are included.
The tiny upstream `testdata/` fixtures are retained because they are part of the original source tree.
The workflow is inherited and its hosted Linux/macOS runs have not been newly executed for this packaging
request. Its dependency recipe differs from the pinned benchmark build above; CI is not a reproduction
of the performance evaluation. `scripts/bench_e2e.sh` is an inherited synthetic benchmark helper,
not the independent real-data evaluation harness and not the source of the reported results.

## Prior validation scope

This unchanged implementation completed an independent evaluation on three full public sequencing runs,
three measured repetitions per version, four processing workers and gzip level four. All nine paired
full-stream FASTQ comparisons and all 74,217 paired scientific numeric values were exact; the measured
runtime reductions were 21.97%, 18.31% and 32.86%, with essentially unchanged native peak memory.
These are selected-workload results, not a guarantee for other CPUs, thread counts or options.
The full and lite evaluation archives are separate deliverables; none of their datasets or run outputs
are bundled here. No new optimization, benchmarking or upstream publication was performed to make this ZIP.
