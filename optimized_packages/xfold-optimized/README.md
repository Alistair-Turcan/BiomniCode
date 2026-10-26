# xfold-optimized

The accepted optimized xfold implementation from this project, packaged as a
standalone repository. The complete source tree is preserved byte-for-byte from
the evaluated version: **accepted changes `iter002` and `iter012` only**.

This is xfold ported to the **OpenFold3-preview2 weights**, with the corrected
stock/off implementation as its baseline. It is **not** an unmodified checkout
of upstream xfold, and it does not use DeepMind's restricted AlphaFold 3 weights.

## Start here

- [Setup and usage](docs/USAGE.md): Docker setup, standard prediction entry point,
  and the direct API profile used in evaluation.
- [Original component documentation](af3_torch/README.md) and
  [pinned software/build instructions](af3_torch/STOCK.md).
- [Source pins](provenance/source_pins.json),
  [accepted patch](provenance/accepted.patch), and
  [original source inventory](provenance/source_inventory.json).
- [License index](LICENSE) and [attribution notes](NOTICE.md).

**Important:** the inherited component README advertises several modes and
defaults to `fast`. Those are original kit options, not interchangeable names
for this project's evaluated optimization. To avoid enabling additional kit
optimizations, select **`--mode off` explicitly** when using the kit CLI.
The accepted changes are already present in this tree's model/API code;
`off` does not revert them. The original unpatched comparator is not bundled.
The benchmark used the direct API profile documented in [USAGE](docs/USAGE.md),
not a new timing run of the CLI or this packaged Docker build.

## What was preserved

- All **1,461 original source files**, including the two accepted modified files.
- The original `af3_torch/` and `common/` layout, APIs, model defaults, kernel
  providers, source-bundled example inputs, kernel test vectors, and tests.
- All component licenses, source headers, and third-party notices.
- Existing dependency locks, Docker/Apptainer recipes, and pinned source-build
  setup instructions.

The accepted evaluation profile uses fp32 weights, bf16 autocast, the original
fastnn Triton implementations, five ordered samples, 200 diffusion steps per
sample, and 10 configured recycles plus the corrected initial pass. It does not
enable the kit's separate wholesale `exact`, `fast`, or `big` lever sets.

## External assets

Model weights, installed Python environments, compiled runtime caches, and
benchmark inputs/predictions/results are **not included**. Obtain weights
separately using the component's pinned fetch/conversion instructions, subject
to their applicable terms. Original source-bundled examples and kernel test
fixtures are retained.

The historical prebuilt featurizer wheel is also not bundled. The default
Docker route builds the pinned featurizer source. This is a setup route, not a
promise of bitwise replay of the historical wheel or its chemical-component
data; see [setup caveats](docs/USAGE.md#reproducibility-and-validation).

## Put this repository on GitHub

Extract the ZIP and run from this folder:

```bash
git init
git add .
git commit -m "Import accepted optimized xfold source"
```

Create an empty GitHub repository, then follow its commands to add a remote and
push. No Git history, account information, access tokens, or remote configuration
is embedded in this package. No existing file requires Git LFS on size grounds;
the bundled provider assets make the initial repository larger than a minimal
Python-only source checkout.

The `.gitignore` deliberately preserves vendored binaries, test vectors, and
source `build/` paths; do not replace it with blanket `*.so`, `*.pt`, or `build/`
ignores. `.gitattributes` disables automatic line-ending conversion to retain
the frozen source bytes.

## Integrity

On Linux, from the repository root:

```bash
sha256sum --check SHA256SUMS
```

The inventory records every original source member's path, size, mode, and
SHA256. The accepted patch is **against the corrected stock/off preview2 port**,
not pristine upstream xfold; it is retained for review, not for applying again
to this already-patched tree.

This packaging pass checked source identity, patch applicability in reverse,
syntax, documentation links, Git inclusion, member sizes, and archive integrity.
It did **not** rerun a clean software installation, GPU inference, or benchmarks.

## Licensing

Retain the [license index](LICENSE), [attribution notes](NOTICE.md), component
licenses/notices, and per-file headers when redistributing. The inherited
notices document a mismatch between xfold's repository-level Apache license and
some carried AlphaFold-derived per-file CC BY-NC-SA headers. This package
preserves those notices; it does not resolve that upstream issue or grant a
new blanket license over third-party material.
