# Validation evidence

Scripts and results backing the correctness and performance claims in the top-level README. All comparisons are the fixed Pyro-free torch module vs. the original Pyro `Cell2location` model (upstream algorithm), run with scvi-tools 1.3.3, pyro-ppl 1.9.1, torch 2.7.1.

## The initialization fix

- `torch_module_init_fix.patch` — the complete diff applied on top of the `cell2location_optimized` snapshot. Variational scales are initialized to `log(expm1(0.1))` (the unconstrained value) so the post-softplus initial scale is exactly `0.1`, matching Pyro AutoNormal's `PyroParam(0.1, softplus_positive)`. Before the fix the effective initial scale was `softplus(0.1) ≈ 0.744` on all 17 latent sites.

## Scripts

- `validate_c2l.py` — three-part equivalence check: (A) exact initialization of all 17 latent sites vs. Pyro guide + analytic prior means; (B) ELBO-at-init match vs. Pyro `Trace_ELBO`; (C) end-to-end training equivalence at small scale (1000 spots × 1000 genes, full-batch) including Pyro-vs-Pyro seed noise floor and ground-truth recovery.
- `bench_scaling.py` — runtime/memory scaling benchmark (5k/10k/20k spots × 15k genes, 30 factors, 20 epochs, minibatch 256, seed-42 synthetic workload) plus correctness-at-scale checks (init + ELBO at every size) and a 200-epoch training-equivalence run at 5k. Requires a CUDA GPU.

## Results

- `validation_metrics.csv`, `validation_ground_truth_recovery.csv` — small-scale (1k spots) equivalence: fixed-vs-Pyro `w_sf` Pearson 0.9974 vs. noise floor 0.9987; ground-truth recovery 0.7762 (fixed) vs. 0.7764/0.7777 (Pyro seeds); unfixed module far outside (0.7381 / truth 0.5537).
- `runtime.csv` — per-rep timings and peak GPU memory (A10G): 33×/29×/28× speedups at 5k/10k/20k spots, 2–3× lower memory.
- `correctness_at_scale.csv` — init exact (max scale error 1.5e-8) and ELBO relative difference ≈ 2e-5 at all three sizes.
- `equivalence_5k.csv` — 200-epoch training equivalence at 5k spots: fixed-vs-Pyro `w_sf` Pearson 0.9967 vs. noise floor 0.9979; ground-truth recovery 0.8618 (fixed) vs. 0.8614/0.8616 (Pyro seeds).
- `runtime_scaling.png/.svg` — runtime and memory vs. input size figure.
- `validation_wsf_scatter.png/.svg` — posterior-mean `w_sf` scatter, fixed torch vs. Pyro (small scale).

Note: the 20-epoch setting in `bench_scaling.py` is a *timing* protocol, not a convergence protocol — at 20 epochs both implementations sit at the prior. The equivalence run uses 200 epochs.
