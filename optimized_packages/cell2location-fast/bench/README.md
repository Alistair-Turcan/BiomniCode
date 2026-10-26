# Cell2Location Optimization Benchmarks

## Files
- `workload_gen.py` — synthetic spatial transcriptomics data generator
- `bench_final.py` — benchmark script for the Pyro-free torch module

## Usage
```bash
# From the bench/ directory:
python bench_final.py 10000 20 3  # n_obs=10000, epochs=20, reps=3
```

## Optimizations Applied
1. Pyro-free direct PyTorch ELBO (eliminates Pyro tracing overhead)
2. torch.compile mode="reduce-overhead" (fuses elementwise ops into Triton kernels)
3. bfloat16 data preloading (halves data memory, same range as float32)
4. Fused Adam optimizer (single CUDA kernel for optimizer step)
5. init_to_mean initialization (matches Pyro's AutoNormal init_to_mean)

## Results (A100 GPU)
| Size | Pyro baseline | Pyro-free optimized | Speedup | Memory reduction |
|------|--------------|--------------------|---------|-----------------|
| 5k spots | 26.4s, 303MB | 0.72s, 200MB | 37x | 34% |
| 10k spots | 62.2s, 893MB | 1.38s, 355MB | 45x | 60% |
| 20k spots | 69.1s, 1493MB | 1.44s, 666MB | 48x | 55% |
