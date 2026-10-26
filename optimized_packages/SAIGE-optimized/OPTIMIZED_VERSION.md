# Optimized SAIGE source snapshot

This folder is the current best optimized SAIGE source, ready to initialize as
a GitHub repository. Upload this folder's contents, not the enclosing ZIP.

- Upstream project: https://github.com/saigegit/SAIGE
- Source revision: `e113e4449995dfee3120ce127d849ebd2afcd274`
- Package version: SAIGE 1.5.2
- Optimization scope evaluated: Step 1 null-model fitting and Step 2 single-variant
  tests with PLINK BED/BIM/FAM input, one thread.

## Contents

The committed R/C++ source, package metadata, original README, LICENSE,
documentation, dependency manifest, Docker files and distributed example
fixtures are retained. No optimization benchmark datasets/results are included.
The embedded installed package `src/SAIGE/` is omitted because it is a build
artifact, not source. Local `.pixi/` environments, `libplink2_includes.a`, Git
history, caches and untracked files are not included.

This is a source distribution, not a precompiled installation. Follow the
upstream installation documentation linked in README.md to obtain the required
R/native dependencies and PLINK2 build inputs. Machine-specific dependencies
and libraries must be installed/built on the destination system.

All retained source files are unchanged from the revision above. The
SOURCE_SNAPSHOT.json file identifies this snapshot; this note and that JSON
are the only added files. Keep the original license and attribution when
publishing.

## Publish with Git

From this extracted folder:

```sh
git init
git add .
git commit -m "Import optimized SAIGE source"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```
