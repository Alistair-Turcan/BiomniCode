# DIAMOND — optimized fork

This repository packages the **current retained optimization (iter006)** of
DIAMOND v2.2.8, based on upstream commit
`745cd3b275da3dfe0e4721b75420122daa7e4d07`.

The optimization is already applied to the source. **Do not apply
`current_best.patch` again.** That patch is included for review and provenance.
The historical optimized workspace commit was `e098397`; Git history is not
included in this clean source package.

## Build and test

Linux/x86-64 prerequisites: a C++ compiler, CMake, zlib and SQLite development
libraries. The configuration below also enables zstd support.

For Ubuntu/Debian:

```bash
sudo apt-get install build-essential cmake zlib1g-dev libsqlite3-dev libzstd-dev
```

From this repository's root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWITH_ZSTD=ON
cmake --build build --parallel 4
./build/diamond version
./build/diamond test
```

The existing command-line interface and binary name are unchanged. Normal
DIAMOND `blastp` and `blastx` commands use the optimized source automatically
when its guarded SIMD path is applicable; other cases retain the original path.

## What changed

The retained change accelerates exact reduced-seed encoding for supported
eight-letter seeds using existing SIMD build/dispatch infrastructure. It
preserves the generic implementation and tail handling. No new dependency,
search option or intentional numerical/output change was introduced.

See [OPTIMIZATION.md](OPTIMIZATION.md) for the change and validation summary,
[current_best.patch](current_best.patch) for the exact code diff, and
[PACKAGE_PROVENANCE.json](PACKAGE_PROVENANCE.json) for pinned identities.

## Publish to your GitHub repository

Unzip the package and use the contents of `diamond-optimized/` as the repository
root. Hidden files such as `.github/` and `.gitignore` are included.

To initialize a new repository before building:

```bash
git init -b main
git add .
git commit -m "Import retained DIAMOND optimization"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your repository. No remote is configured and nothing
is uploaded by this package. Original upstream workflow definitions are retained;
review them for your repository's CI settings.

## Upstream, license and scope

This is an optimized fork, **not an official upstream release**. The original
project documentation and attribution are preserved in
[README_UPSTREAM.md](README_UPSTREAM.md), with its original
[LICENSE](LICENSE), [CITATION.cff](CITATION.cff), and [TRADEMARKS](TRADEMARKS).

Bundled upstream test fixtures and expected outputs are preserved. Large
evaluation datasets, external evaluation reference databases, benchmark run
outputs, compiled binaries, build directories, and `.git` history are not
included. This is a source package, not the evaluation ZIP.
