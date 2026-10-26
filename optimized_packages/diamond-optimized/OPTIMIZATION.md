# Retained optimization and validation

## Exact retained source change

- Upstream DIAMOND version: **2.2.8**
- Pinned upstream commit: `745cd3b275da3dfe0e4721b75420122daa7e4d07`
- Accepted optimization: **iter006**
- Historical optimized local commit: `e098397`
- Patch SHA-256:
  `634dd1d224977d57616f6f0852145f52b1516dd27b12848deef9e73ece416a2c`

Only these two implementation paths differ from the pinned original:

1. `src/search/seed_array/enum_seeds.h`
2. `src/search/seed_array/reduced_seed_simd.h` (new)

The code introduces an exact SIMD reduced-seed encoder, guarded by the existing
SSE4.1/sequence-mask compilation conditions. Its fast path requires seed weight
eight, shape length at most sixteen, a ten-letter reduction, and sixteen readable
input bytes. Unsupported configurations and sequence tails use the original
iterator. Masked-letter rejection and downstream filtering remain in place.
No approximate arithmetic or relaxed comparison tolerance is introduced.

## Existing validation record

This is the same retained implementation used in the completed optimization and
independent large-data evaluation, not a new optimization attempt.

- Both implementations passed the previously recorded 40 upstream self-tests.
- The independent evaluation used complete human, mouse and maize protein/CDS
  collections, both `blastp` and `blastx`, and three repetitions per version:
  **36 successful timed searches** on one eight-thread worker.
- All **18 paired output comparisons** were byte-for-byte identical across
  **30,587,517 matched hit pairs**. Within-version repeated outputs were also
  byte-identical.
- Mean runtime reductions across those six workloads ranged from **5.06% to
  15.71%**; peak-memory changes were mixed, not uniformly improvements.
- These are workload-specific regression/performance results, not proof of
  universal speedup or independent biological accuracy.

The full measurements, raw outputs and validation details are supplied separately
in the completed evaluation packages. They are intentionally not duplicated here.
No benchmark or source build was rerun merely to assemble this source ZIP.

## Packaging checks

Before adding packaging documentation, the delivered optimized source was
reverse-patched and every resulting file was compared with the pinned original
source archive. The complete file sets and contents matched. The only changed
implementation paths are the two listed above.

Packaging adds documentation, a copy of the retained patch and provenance; it
does not modify the evaluated implementation or its CMake/build configuration.
The original README is preserved verbatim as `README_UPSTREAM.md`. Original
source test fixtures, workflow definitions, licensing and attribution are retained.
