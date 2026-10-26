# Optimized source snapshot

## Identity

- Upstream base commit: `b512206e3e0bcb9f0720423293094935861fdcde`.
- The evaluated winning patch is included at `docs/optimized.patch`.
- All retained original package files are unchanged and checked against both the
  optimization delivery and subsequent evaluation source manifests.
- Packaging adds documentation, ignore rules, and provenance only.

## Included and omitted

Included: all four R implementation files, all seven Rd help files, the original
three small bundled package datasets, package metadata, namespace, GPL license,
historical README, and the winning optimization patch.

Omitted: the large external `example/` directory (including cached fitted
outputs), downloaded evaluation datasets, benchmark outputs and logs, dependency
installations, Git history, and the upstream automatic pkgdown deployment workflow.
The separate full and lite evaluation archives remain the evaluation deliverables.

## Known inherited limitations

The original `DESCRIPTION` lacks a `Maintainer` field. The previous package checks
for both original and optimized versions stopped at that metadata error.
This snapshot preserves author/contact metadata rather than assigning a new
maintainer without authorization. Set the correct maintainer for your fork before
expecting a clean `R CMD check`; this is not a CRAN-ready release.

Some inherited documentation examples refer to the omitted external `example/`
directory. The bundled `data/` directory is separate and retained.
The default dependency installation commands are not an environment lockfile.
No new dependency upgrades, algorithm changes, or numerical claims are introduced.
