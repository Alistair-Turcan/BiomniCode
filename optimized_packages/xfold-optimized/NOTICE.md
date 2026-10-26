# Provenance and attribution

This package preserves the complete accepted source inventory. Its component
documentation and original copyright/license headers are unchanged.

## Upstream identities

- xfold: `Shenggan/xfold`,
  commit `22bdeedfa309ef4ff6f9199910d8403915de69d6`.
- Corrected preview2 port and provider reference:
  `anthropics/uplifting-biomolecular-modeling`,
  commit `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`.
- Converter/featurizer: `sokrypton/alphafold3`,
  commit `bc32b22ff5902e3daffd5d1f7203d7f2ab6cb997`.

## Accepted changes in this project

Only `iter002` and `iter012` were accepted. The cumulative
[patch](provenance/accepted.patch) modifies two files:

- `af3_torch/opt/forward/af3t/af3_torch/af3_torch_api.py`:
  inference-scoped use and cleanup of the inherited graph/conditioning
  mechanism, with safety guards for static state and hooks.
- `af3_torch/opt/forward/af3t/af3_torch/xfold/nn/diffusion_transformer.py`:
  guarded selective reuse of the existing `af3t_glu_proj` mechanism in the
  eligible diffusion transition, preserving the conditioning operation.

The graph/conditioning and `af3t_glu_proj` mechanisms were inherited and
selectively reused; this project does not claim authorship of those original
components. The patch is relative to the corrected stock/off preview2 port and
must not be applied to this already-patched source tree.

This packaging adds repository documentation, ignore/attribute files, and
compact integrity/provenance metadata only. It does not modify the original
scientific source files.

## Licenses

Read the [license index](LICENSE), the
[AF3-torch notices](af3_torch/THIRD_PARTY_NOTICES.md), and the
[shared-core notices](common/opt_core/THIRD_PARTY_NOTICES.md).
Keep the inherited per-file licensing caveat described there. No upstream
notice, attribution, or license has been replaced by this packaging.
