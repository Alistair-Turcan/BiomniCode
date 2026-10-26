# cell2location-fast

A GPU-optimized, drop-in fork of [cell2location](https://github.com/BayraktarLab/cell2location) 0.1.5 with a corrected variational initialization. Same model, same API, same results — roughly **30× faster** training on GPU with **2–3× lower** peak memory.

> **This is a fork.** All scientific credit for the model belongs to the original cell2location authors. If you use this software, **cite the original paper**:
>
> Kleshchevnikov, V., Shmatko, A., Dann, E. et al. *Cell2location maps fine-grained cell types in spatial transcriptomics.* Nat Biotechnol 40, 661–671 (2022). https://doi.org/10.1038/s41587-021-01139-4

## What is different from upstream cell2location?

Two things, and nothing else:

### 1. GPU performance optimizations (from the fork this repo is based on)

The spatial-mapping model (`Cell2location`) was re-implemented as a Pyro-free, pure-PyTorch module (`cell2location/models/_cell2location_torch_module.py`) that computes the exact same ELBO as the original Pyro model:

1. **Pyro-free direct PyTorch ELBO** — eliminates Pyro tracing/dispatch overhead per minibatch.
2. **`torch.compile(mode="reduce-overhead")`** — fuses elementwise ops into Triton kernels.
3. **bfloat16 data preloading** — the count matrix lives on the GPU in bf16 (halves data memory; note bf16 has an 8-bit mantissa, so counts >256 are rounded — a deliberate speed/precision tradeoff).
4. **Fused Adam optimizer** — single CUDA kernel for the optimizer step.
5. **`init_to_mean` initialization** — variational distributions initialized at prior means, matching Pyro's AutoNormal `init_to_mean`.

The original Pyro model classes are untouched and remain available.

### 2. Initialization bug fix (this repo)

The optimized torch module shipped with an initialization bug: variational scale parameters were created as raw `nn.Parameter(0.1)` and then passed through a `softplus` transform, giving an effective initial posterior scale of `softplus(0.1) ≈ 0.744` instead of the intended `0.1` used by Pyro's AutoNormal (which stores the *unconstrained* value `softplus⁻¹(0.1) ≈ -2.252`). That is **7.4× excess initial posterior noise on all 17 latent sites**, which visibly changes early training dynamics.

The fix stores the correct unconstrained value, `log(expm1(0.1))`, so the constrained initial scale is exactly `0.1`, matching Pyro. See `validation/torch_module_init_fix.patch` for the exact diff.

## Performance

Measured on an NVIDIA A10G (this repo, seed-42 synthetic workload: 15,000 genes, 30 factors, 20 epochs, minibatch 256 — see `validation/bench_scaling.py`):

| Spots | Pyro original | cell2location-fast | Speedup | Pyro peak mem | Fast peak mem |
|---|---|---|---|---|---|
| 5,000 | 26.9 s | 0.82 s | **33×** | 590 MB | 199 MB |
| 10,000 | 47.6 s | 1.66 s | **29×** | 910 MB | 372 MB |
| 20,000 | 93.4 s | 3.32 s | **28×** | 1,537 MB | 710 MB |

The original optimization work reports 37×/45×/48× on an A100 (see `bench/README.md`); the gap versus our A10G numbers is expected because the optimized path is memory-bandwidth-bound. Runtime scales linearly in the number of spots for both implementations, so the speedup is roughly constant with input size.

## Correctness verification

The fixed fast module was validated against the original Pyro implementation (full details and scripts in `validation/`):

- **Initialization (exact)**: all 17 latent sites' initial locations and scales match Pyro's AutoNormal guide and the analytic prior means to float32 precision (max scale error 1.5e-8), at 5k/10k/20k spots.
- **ELBO at init**: relative difference vs. Pyro `Trace_ELBO` ≈ 2e-5 at all three sizes.
- **End-to-end training equivalence** (5k spots, 200 epochs): torch-vs-Pyro posterior means for cell abundances `w_sf` agree at Pearson 0.9967, within the Pyro-vs-Pyro seed noise floor (0.9979); ground-truth abundance recovery identical to 3 decimals (0.8618 vs 0.8614/0.8616).
- **Unfixed module shown to deviate**: with the original `0.744` init scale the same checks fall well outside the noise floor.

## Installation

```bash
pip install git+https://github.com/<your-username>/cell2location-fast.git
```

Requires Python ≥3.10 and a CUDA-enabled PyTorch for the GPU fast path (CPU works but the speedup is GPU-specific). Dependencies are unchanged from upstream (`pyro-ppl`, `scvi-tools>=1.3`, `torch`, `numpy`, `pandas`, `scanpy`, `opencv-python`).

**Do not install alongside the original `cell2location` package** — this fork is a drop-in replacement and uses the same `cell2location` import namespace.

## Usage

Identical to upstream cell2location — existing scripts and tutorials work unchanged:

```python
import cell2location
from cell2location.models import Cell2location

# ... setup_anndata, then:
mod = Cell2location(adata_vis, cell_state_df=inf_aver,
                    N_cells_per_location=30, detection_alpha=200)
mod.train(max_epochs=250, batch_size=2500)   # uses the fast torch path on GPU
adata_vis = mod.export_posterior(adata_vis)
```

Upstream tutorials: https://cell2location.readthedocs.io/en/latest/

Remember the two user-provided hyperparameters (`N_cells_per_location`, `detection_alpha`) — see the [upstream guidance](https://github.com/BayraktarLab/cell2location/blob/master/docs/images/Note_on_selecting_hyperparameters.pdf): try `detection_alpha=200` for low and `detection_alpha=20` for high within-slide technical variability.

## Repository layout

- `cell2location/` — the package (import name unchanged: `cell2location`)
- `bench/` — the original optimization benchmark scripts and A100 results
- `validation/` — initialization-fix patch, validation/benchmark scripts, and all result CSVs/figures backing the numbers above
- `docs/`, `tests/` — upstream documentation and tests

## License and attribution

Apache License 2.0, same as upstream — see `LICENSE` and `NOTICE`. Original work Copyright the cell2location authors (Wellcome Sanger Institute, Bayraktar Lab). Modifications in this fork are documented in `NOTICE`.

## Reporting issues

- Bugs in the **fast path / this fork**: open an issue on this repository.
- Bugs in the **model or upstream package**: https://github.com/BayraktarLab/cell2location/issues
- Usage questions: scverse community discourse ([cell2location](https://discourse.scverse.org/c/ecosytem/cell2location/42)).
